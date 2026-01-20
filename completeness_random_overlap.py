#!/usr/bin/env python3
"""
completeness_random_overlap.py

Goal:
  Heuristically test "completeness" of an eigenstate database up to an energy cutoff E_CUT
  by generating random *low-energy* wavefunctions and measuring how much of their norm lies
  in the span of stored eigenmodes with energy <= E_CUT.

How it works (high level):
  1) Load eigenpairs from an HDF5 database (evals/evecs).
  2) Select the subset of modes with Re(E) <= E_CUT.
  3) Build the Dirichlet Laplacian Hamiltonian H from the saved geometry (phi/allowed).
  4) Generate random vectors and "diffusion-filter" them with (I - dt H)^N to bias toward low energies.
  5) For each filtered test wavefunction psi, compute:
       - Rayleigh quotient E_psi = <psi|H|psi>/<psi|psi>
       - overlap fraction f = ||P psi||^2 / ||psi||^2, where P projects onto the selected eigenmodes
  6) Report statistics and save a histogram plot of overlap fractions.

Notes:
  - This is a *practical diagnostic*, not a rigorous proof. If your random test psi still has
    appreciable high-energy components, overlap can be < 1 even if the basis is complete.
    Increase FILTER_STEPS or decrease FILTER_DT to make psi more band-limited.
"""

import os
import json

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix


# ========= USER CONFIG (edit these) =========
H5_PATH = r"eigensdf/doublewell/final2/eigenpairs.h5"   # eigenpairs.h5 or eigenpairs_complex.h5

# If None: will look for geometry_used.npz in the same folder as H5_PATH.
GEOM_NPZ_PATH = None
# Fallback if geometry_used.npz isn't found and GEOM_NPZ_PATH is None.
SDF_FALLBACK_PATH = r"potentials/final_potential.npz"

# Completeness test cutoff (uses Re(E) if energies are complex)
E_CUT = 4

# Test type:
#   - "random": random vectors + diffusion filter (older behavior)
#   - "wavepacket": Gaussian wavepackets with a tunable carrier momentum (recommended)
TEST_TYPE = "wavepacket"

# How many test wavefunctions
N_TRIALS = 1
RANDOM_SEED = 1234

# Low-pass / diffusion filtering: psi <- (I - dt H)^N psi
# (Used for TEST_TYPE="random", and optionally as a light “clean-up” for wavepackets.)
FILTER_STEPS = 80
FILTER_DT = 0.04

# Acceptance / energy targeting
# For wavepackets, you typically want E_psi near E_TARGET and below E_CUT.
E_TARGET = 1.0             # target Rayleigh energy for generated test packets
E_TOL = .9               # accept if |E_psi - E_TARGET| <= E_TOL
MAX_RAYLEIGH = 2.5        # optional hard ceiling (e.g. 1.0). If None, uses E_CUT.
MAX_TRIES_PER_TRIAL = 1    # retries per trial if energy targeting fails

# Projection computation
CHUNK_EVECS = 256           # how many eigenvectors to pull from HDF5 at once
RENORMALIZE_EVECS = True    # safe even if your store isn't perfectly normalized
MAX_MODES_UNDER_CUT = None  # optional cap for speed; e.g. 5000 (uses lowest energies only)
SHOW_PROJECTION_PROGRESS = True
PROGRESS_EVERY_CHUNKS = 1   # print every N chunks (per trial)

# Wavepacket parameters (TEST_TYPE="wavepacket")
WP_SIGMA_PIX = 32.0         # Gaussian width in pixels
WP_WALL_MARGIN_PHI = 30    # avoid centers too close to wall (requires phi)
WP_CENTER_MODE = "fixed"   # "random" or "fixed"
WP_CENTER_X = 110         # used if WP_CENTER_MODE="fixed" (pixel x)
WP_CENTER_Y = 515.0         # used if WP_CENTER_MODE="fixed" (pixel y)
WP_K_DIR_MODE = "y"    # "random" or "x" or "y"
WP_K_TUNE_ITERS = 6         # adjust |k| to hit E_TARGET (Rayleigh)
WP_POST_DIFFUSE_STEPS = 0   # optional small smoothing (0-10)

# Preview rendering of a sample wavepacket (saved as PNG)
PREVIEW_WAVEPACKET = True
PREVIEW_SHOW = True          # show interactively with plt.show()
PREVIEW_SAVE = False         # optionally also save to file
PREVIEW_FILENAME = "wavepacket_preview.png"  # used only if PREVIEW_SAVE=True
PREVIEW_PIXELS = 1024
PREVIEW_DPI = 256
PREVIEW_BG_OUT_RGB = (0, 0, 0)
PREVIEW_BG_IN_RGB = (70, 80, 95)   # lighter slate (not white)
PREVIEW_PROB_CMAP = "turbo"
PREVIEW_PROB_PCT_HI = 99.5
PREVIEW_PROB_GAMMA = 0.6
PREVIEW_PROB_SHOW_MIN = 0.08
PREVIEW_PROB_ALPHA_GAMMA = 2.0
PREVIEW_DRAW_BOUNDARY = True
PREVIEW_BOUNDARY_COLOR = "#0b0b0b"
PREVIEW_BOUNDARY_LW = 1.0

# Output
OUT_DIR = None  # None -> folder next to H5_PATH
OUT_PREFIX = "completeness_test"
MAKE_PLOT = True
# ===========================================


def load_geometry(geom_npz_path, h5_path, sdf_fallback):
    if geom_npz_path is None:
        cand = os.path.join(os.path.dirname(os.path.abspath(h5_path)), "geometry_used.npz")
        geom_npz_path = cand if os.path.exists(cand) else None

    if geom_npz_path is not None and os.path.exists(geom_npz_path):
        g = np.load(geom_npz_path)
        phi = g["phi"].astype(np.float32) if "phi" in g.files else None
        allowed = g["allowed"].astype(bool) if "allowed" in g.files else None
        if allowed is None:
            if phi is None:
                raise ValueError("geometry_used.npz missing both 'allowed' and 'phi'.")
            allowed = phi > 0.0
        if phi is None:
            # phi is needed for smooth-wall stencil below; fallback to a crude phi if absent
            phi = allowed.astype(np.float32) - 0.5
        return phi, allowed, geom_npz_path

    # fallback to SDF NPZ
    g = np.load(sdf_fallback)
    phi = g["phi"].astype(np.float32) if "phi" in g.files else None
    if "allowed" in g.files:
        allowed = g["allowed"].astype(bool)
    else:
        if phi is None:
            raise ValueError("SDF_FALLBACK_PATH must contain 'phi' if 'allowed' is not present.")
        allowed = phi > 0.0
    if phi is None:
        phi = allowed.astype(np.float32) - 0.5
    return phi, allowed, sdf_fallback


def build_dirichlet_sw(phi, allowed, alpha_clip=5e-3) -> csr_matrix:
    """
    Same smooth-wall Dirichlet stencil as in lowband_complete_solver.py (real symmetric).
    Returns sparse H on the packed allowed DOFs.
    """
    Ny, Nx = phi.shape
    idx = -np.ones((Ny, Nx), np.int64)
    idx[allowed] = np.arange(allowed.sum())
    n = int(allowed.sum())

    rows, cols, vals = [], [], []
    diag = np.zeros(n, np.float64)

    def add(a, b, v):
        rows.append(a)
        cols.append(b)
        vals.append(v)

    def link(a, y, x, yn, xn):
        if 0 <= yn < Ny and 0 <= xn < Nx and allowed[yn, xn]:
            b = idx[yn, xn]
            add(a, b, -1.0)
            diag[a] += 1.0
        else:
            phi_i = phi[y, x]
            phi_o = phi[yn, xn] if (0 <= yn < Ny and 0 <= xn < Nx) else -phi_i
            denom = (phi_i - phi_o)
            alpha = 0.5 if denom <= 0 else np.clip(phi_i / denom, alpha_clip, 1.0)
            diag[a] += 1.0 / alpha

    for y in range(Ny):
        for x in range(Nx):
            a = idx[y, x]
            if a < 0:
                continue
            link(a, y, x, y, x + 1)
            link(a, y, x, y, x - 1)
            link(a, y, x, y + 1, x)
            link(a, y, x, y - 1, x)

    rows += list(range(n))
    cols += list(range(n))
    vals += list(diag)
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def diffusion_filter(H: csr_matrix, psi: np.ndarray, steps: int, dt: float) -> np.ndarray:
    """
    psi <- (I - dt H)^steps psi, with renormalization each step.
    """
    psi = psi.astype(np.complex128, copy=False)
    for _ in range(int(steps)):
        psi = psi - dt * (H @ psi)
        nrm = np.linalg.norm(psi) + 1e-30
        psi = psi / nrm
    return psi


def rayleigh_quotient(H: csr_matrix, psi: np.ndarray) -> float:
    Hpsi = H @ psi
    num = np.vdot(psi, Hpsi)
    den = np.vdot(psi, psi)
    if den == 0:
        return np.inf
    return float(np.real(num / den))


def proj_norm_sq(
    h5: h5py.File,
    eig_idxs: np.ndarray,
    psi: np.ndarray,
    chunk: int,
    renorm: bool,
    show_progress: bool = False,
    progress_every_chunks: int = 5,
) -> float:
    """
    Compute ||P psi||^2 where P projects onto span{v_k} for k in eig_idxs.
    Reads eigenvectors in chunks to avoid loading everything at once.
    """
    evecs_ds = h5["evecs"]
    total = 0.0
    psi = psi.astype(np.complex128, copy=False)

    # Fast path: if indices are exactly [0,1,2,...,K-1], we can slice contiguously.
    contiguous = (eig_idxs.size > 0) and np.array_equal(eig_idxs, np.arange(eig_idxs.size, dtype=np.int64))

    chunk = int(chunk)
    progress_every_chunks = max(int(progress_every_chunks), 1)
    chunks_total = int((eig_idxs.size + chunk - 1) // chunk)
    chunk_i = 0

    for start in range(0, eig_idxs.size, chunk):
        stop = min(start + int(chunk), eig_idxs.size)
        if contiguous:
            V = np.array(evecs_ds[:, start:stop], dtype=np.complex128)
        else:
            idxs = np.asarray(eig_idxs[start:stop], dtype=np.int64)
            idxs_sorted = np.sort(idxs)  # h5py requires increasing order
            V = np.array(evecs_ds[:, idxs_sorted], dtype=np.complex128)  # (n, m)

        if renorm:
            norms = np.sqrt(np.sum(np.abs(V) ** 2, axis=0)) + 1e-30
            V = V / norms

        c = V.conj().T @ psi  # (m,)
        total += float(np.sum(np.abs(c) ** 2))

        chunk_i += 1
        if show_progress and (chunk_i % progress_every_chunks == 0 or chunk_i == chunks_total):
            done = min(stop, eig_idxs.size)
            print(f"    [proj] {done}/{eig_idxs.size} modes processed ({chunk_i}/{chunks_total} chunks)")

    return total


def packed_xy_from_allowed(allowed: np.ndarray):
    """
    For each packed DOF index i (where allowed is True), return its (x,y) coordinates.
    """
    ys, xs = np.where(allowed)
    # matches packing order used throughout this repo: img2d[allowed].reshape(-1)
    return xs.astype(np.float64), ys.astype(np.float64)


def make_wavepacket(xp, yp, x0, y0, sigma, kx, ky):
    dx = xp - float(x0)
    dy = yp - float(y0)
    env = np.exp(-(dx * dx + dy * dy) / (2.0 * float(sigma) * float(sigma)))
    phase = np.exp(1j * (float(kx) * dx + float(ky) * dy))
    psi = (env * phase).astype(np.complex128, copy=False)
    psi /= (np.linalg.norm(psi) + 1e-30)
    return psi


def nearest_allowed_center(cx: np.ndarray, cy: np.ndarray, x0: float, y0: float):
    """
    Find nearest allowed center pixel (cx,cy arrays) to a desired (x0,y0).
    """
    dx = cx.astype(np.float64) - float(x0)
    dy = cy.astype(np.float64) - float(y0)
    j = int(np.argmin(dx * dx + dy * dy))
    return float(cx[j]), float(cy[j]), j


def make_background_from_phi(phi, allowed, bg_out_rgb, bg_in_rgb):
    """
    Create an RGB background using SDF smoothing if phi is present:
      outside -> bg_out_rgb, inside -> bg_in_rgb
    """
    out_rgb = np.array(bg_out_rgb, dtype=np.float32).reshape(1, 1, 3)
    in_rgb = np.array(bg_in_rgb, dtype=np.float32).reshape(1, 1, 3)

    if phi is not None:
        # ~1 px smooth transition across boundary
        w = 1.0
        a = np.clip(0.5 + phi.astype(np.float32) / (2.0 * w), 0.0, 1.0)[..., None]
    else:
        a = allowed.astype(np.float32)[..., None]

    bg = (1.0 - a) * out_rgb + a * in_rgb
    return np.clip(bg, 0, 255).astype(np.uint8)


def render_wavepacket_preview(out_path, psi, allowed, phi):
    """
    Render |psi|^2 overlayed on the geometry background (saved as PNG).
    """
    Ny, Nx = allowed.shape
    img2 = np.zeros((Ny, Nx), dtype=np.complex128)
    img2[allowed] = psi.reshape(-1)
    p = np.abs(img2) ** 2
    p /= (p[allowed].sum() + 1e-30)

    hi = float(np.percentile(p[allowed], PREVIEW_PROB_PCT_HI)) if allowed.any() else float(np.percentile(p, PREVIEW_PROB_PCT_HI))
    scale = hi if hi > 1e-20 else float(p.max() + 1e-20)
    pn = np.clip(p / scale, 0.0, 1.0)
    if PREVIEW_PROB_GAMMA != 1.0:
        pn = pn ** float(PREVIEW_PROB_GAMMA)

    try:
        cmap = plt.get_cmap(PREVIEW_PROB_CMAP)
    except Exception:
        cmap = plt.get_cmap("turbo")
    color = (cmap(pn)[:, :, :3] * 255).astype(np.uint8)

    t0 = float(np.clip(PREVIEW_PROB_SHOW_MIN, 0.0, 0.999999))
    a = np.clip((pn - t0) / (1.0 - t0), 0.0, 1.0)
    if PREVIEW_PROB_ALPHA_GAMMA != 1.0:
        a = a ** float(PREVIEW_PROB_ALPHA_GAMMA)
    a = a.astype(np.float32)

    bg = make_background_from_phi(phi, allowed, PREVIEW_BG_OUT_RGB, PREVIEW_BG_IN_RGB).astype(np.float32)
    blend = ((1.0 - a)[..., None] * bg + a[..., None] * color.astype(np.float32)).astype(np.uint8)

    figsize = (float(PREVIEW_PIXELS) / float(PREVIEW_DPI), float(PREVIEW_PIXELS) / float(PREVIEW_DPI))
    fig, ax = plt.subplots(figsize=figsize, dpi=PREVIEW_DPI)
    ax.imshow(blend, origin="lower", interpolation="bilinear")
    if PREVIEW_DRAW_BOUNDARY and (phi is not None):
        ax.contour(phi, levels=[0.0], colors=[PREVIEW_BOUNDARY_COLOR], linewidths=PREVIEW_BOUNDARY_LW, origin="lower")
    ax.axis("off")

    if PREVIEW_SAVE and out_path is not None:
        fig.savefig(out_path, bbox_inches=None)

    if PREVIEW_SHOW:
        # Blocking show so you can inspect before the completeness run continues.
        plt.show()

    plt.close(fig)


def main():
    os.makedirs(os.path.dirname(H5_PATH) if OUT_DIR is None else OUT_DIR, exist_ok=True)
    out_dir = os.path.dirname(os.path.abspath(H5_PATH)) if OUT_DIR is None else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # --- load eigenpairs ---
    with h5py.File(H5_PATH, "r") as h5:
        evals = np.array(h5["evals"][:])
        if evals.size == 0:
            raise RuntimeError("No eigenvalues found in H5_PATH.")
        E_key = np.real(evals) if np.iscomplexobj(evals) else evals.astype(np.float64, copy=False)
        finite = np.isfinite(E_key)
        eig_idxs = np.where(finite & (E_key <= float(E_CUT)))[0].astype(np.int64)
        if eig_idxs.size == 0:
            raise RuntimeError(f"No eigenvalues with Re(E) <= {E_CUT}.")
        eig_idxs = np.sort(eig_idxs)

        if MAX_MODES_UNDER_CUT is not None:
            eig_idxs = eig_idxs[: int(MAX_MODES_UNDER_CUT)]

        n = int(h5["evecs"].shape[0])
        print(f"[store] {H5_PATH}")
        print(f"[store] total modes: {evals.size}, DOFs n={n}")
        print(f"[select] modes with Re(E) <= {E_CUT}: {eig_idxs.size}")

        # --- build geometry & H ---
        phi, allowed, geom_path_used = load_geometry(GEOM_NPZ_PATH, H5_PATH, SDF_FALLBACK_PATH)
        n_geom = int(allowed.sum())
        if n_geom != n:
            raise ValueError(f"Geometry DOFs {n_geom} != eigenvector DOFs {n}. Check matching geometry_used.npz.")
        H = build_dirichlet_sw(phi, allowed)
        print(f"[geom] using: {geom_path_used}")
        print(f"[ham] built H with nnz={H.nnz}")

        # --- trials ---
        rng = np.random.default_rng(int(RANDOM_SEED))
        overlaps = []
        rayleighs = []
        accepted = 0
        rejected = 0

        # Precompute packed coordinates for wavepacket generation
        xp, yp = packed_xy_from_allowed(allowed)
        # Candidate centers (avoid wall if phi is meaningful)
        if phi is not None and np.isfinite(phi).any():
            center_mask = (allowed & (phi > float(WP_WALL_MARGIN_PHI)))
        else:
            center_mask = allowed
        cy, cx = np.where(center_mask)
        if cx.size == 0:
            cy, cx = np.where(allowed)
        if cx.size == 0:
            raise RuntimeError("Allowed region is empty.")

        # Preview a sample wavepacket so you can visually confirm the test state
        if TEST_TYPE == "wavepacket" and PREVIEW_WAVEPACKET:
            if WP_CENTER_MODE == "fixed":
                x0p, y0p, _ = nearest_allowed_center(cx, cy, float(WP_CENTER_X), float(WP_CENTER_Y))
            else:
                j0 = int(rng.integers(0, cx.size))
                x0p = float(cx[j0]); y0p = float(cy[j0])

            if WP_K_DIR_MODE == "x":
                theta = 0.0
            elif WP_K_DIR_MODE == "y":
                theta = 0.5 * np.pi
            else:
                theta = float(2.0 * np.pi * rng.random())

            k = float(np.sqrt(max(float(E_TARGET), 1e-12)))
            kx = k * np.cos(theta)
            ky = k * np.sin(theta)
            psi_prev = make_wavepacket(xp, yp, x0p, y0p, float(WP_SIGMA_PIX), kx, ky)
            for _it in range(int(WP_K_TUNE_ITERS)):
                Epsi0 = rayleigh_quotient(H, psi_prev)
                if not np.isfinite(Epsi0) or Epsi0 <= 1e-15:
                    break
                scale = np.sqrt(max(float(E_TARGET), 1e-12) / Epsi0)
                kx *= scale; ky *= scale
                psi_prev = make_wavepacket(xp, yp, x0p, y0p, float(WP_SIGMA_PIX), kx, ky)
            if WP_POST_DIFFUSE_STEPS and int(WP_POST_DIFFUSE_STEPS) > 0:
                psi_prev = diffusion_filter(H, psi_prev, steps=int(WP_POST_DIFFUSE_STEPS), dt=float(FILTER_DT))
            Eprev = rayleigh_quotient(H, psi_prev)
            out_prev = os.path.join(out_dir, PREVIEW_FILENAME) if PREVIEW_SAVE else None
            render_wavepacket_preview(out_prev, psi_prev, allowed, phi)
            msg = f"[preview] center≈({x0p:.1f},{y0p:.1f})  E_ray≈{Eprev:.6g}"
            if PREVIEW_SAVE:
                msg += f"  saved {out_prev}"
            print(msg)

        for t in range(int(N_TRIALS)):
            ok = False
            for _ in range(int(MAX_TRIES_PER_TRIAL)):
                if TEST_TYPE == "random":
                    psi0 = rng.normal(size=n) + 1j * rng.normal(size=n)
                    psi0 = psi0.astype(np.complex128, copy=False)
                    psi0 /= (np.linalg.norm(psi0) + 1e-30)
                    psi = diffusion_filter(H, psi0, steps=FILTER_STEPS, dt=float(FILTER_DT))
                elif TEST_TYPE == "wavepacket":
                    # Choose a random center well inside the domain
                    if WP_CENTER_MODE == "fixed":
                        x0, y0, _ = nearest_allowed_center(cx, cy, float(WP_CENTER_X), float(WP_CENTER_Y))
                    else:
                        j = int(rng.integers(0, cx.size))
                        x0 = float(cx[j])
                        y0 = float(cy[j])

                    # Direction
                    if WP_K_DIR_MODE == "x":
                        theta = 0.0
                    elif WP_K_DIR_MODE == "y":
                        theta = 0.5 * np.pi
                    else:
                        theta = float(2.0 * np.pi * rng.random())

                    # Start with |k| ~ sqrt(E_TARGET) (small-k continuum approx)
                    k = float(np.sqrt(max(float(E_TARGET), 1e-12)))
                    kx = k * np.cos(theta)
                    ky = k * np.sin(theta)

                    psi = make_wavepacket(xp, yp, x0, y0, float(WP_SIGMA_PIX), kx, ky)

                    # Tune |k| to hit Rayleigh energy (quick fixed-point update)
                    for _it in range(int(WP_K_TUNE_ITERS)):
                        Epsi0 = rayleigh_quotient(H, psi)
                        if not np.isfinite(Epsi0) or Epsi0 <= 1e-15:
                            break
                        scale = np.sqrt(max(float(E_TARGET), 1e-12) / Epsi0)
                        kx *= scale
                        ky *= scale
                        psi = make_wavepacket(xp, yp, x0, y0, float(WP_SIGMA_PIX), kx, ky)

                    if WP_POST_DIFFUSE_STEPS and int(WP_POST_DIFFUSE_STEPS) > 0:
                        psi = diffusion_filter(H, psi, steps=int(WP_POST_DIFFUSE_STEPS), dt=float(FILTER_DT))
                else:
                    raise ValueError("TEST_TYPE must be 'random' or 'wavepacket'.")

                Epsi = rayleigh_quotient(H, psi)

                Emax = float(E_CUT) if MAX_RAYLEIGH is None else float(MAX_RAYLEIGH)
                if (Epsi <= Emax) and (abs(Epsi - float(E_TARGET)) <= float(E_TOL)):
                    ok = True
                    break

            if not ok:
                rejected += 1
                continue

            proj2 = proj_norm_sq(
                h5,
                eig_idxs,
                psi,
                chunk=CHUNK_EVECS,
                renorm=RENORMALIZE_EVECS,
                show_progress=bool(SHOW_PROJECTION_PROGRESS),
                progress_every_chunks=int(PROGRESS_EVERY_CHUNKS),
            )
            proj2 = max(0.0, min(1.0, proj2))  # numerical guard
            overlaps.append(proj2)
            rayleighs.append(Epsi)
            accepted += 1

            print(f"[trial {t+1:03d}] E_ray={Epsi:.6g}, overlap={proj2*100:.3f}%")

    if accepted == 0:
        raise RuntimeError("No trials were accepted. Try increasing E_TOL, increasing MAX_RAYLEIGH, or increasing FILTER_STEPS.")

    overlaps = np.array(overlaps, dtype=np.float64)
    rayleighs = np.array(rayleighs, dtype=np.float64)

    stats = {
        "H5_PATH": H5_PATH,
        "GEOM_USED": geom_path_used,
        "E_CUT": float(E_CUT),
        "N_TRIALS_REQUESTED": int(N_TRIALS),
        "N_TRIALS_ACCEPTED": int(accepted),
        "N_TRIALS_REJECTED": int(rejected),
        "FILTER_STEPS": int(FILTER_STEPS),
        "FILTER_DT": float(FILTER_DT),
        "MAX_RAYLEIGH": None if (MAX_RAYLEIGH is None) else float(MAX_RAYLEIGH),
        "K_MODES_UNDER_CUT": int(eig_idxs.size),
        "overlap_min": float(np.min(overlaps)),
        "overlap_mean": float(np.mean(overlaps)),
        "overlap_p01": float(np.quantile(overlaps, 0.01)),
        "overlap_p05": float(np.quantile(overlaps, 0.05)),
        "overlap_p50": float(np.quantile(overlaps, 0.50)),
        "overlap_p95": float(np.quantile(overlaps, 0.95)),
        "overlap_p99": float(np.quantile(overlaps, 0.99)),
        "rayleigh_mean": float(np.mean(rayleighs)),
        "rayleigh_max": float(np.max(rayleighs)),
    }

    print("\n[summary]")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6g}")
        else:
            print(f"  {k}: {v}")

    out_json = os.path.join(out_dir, f"{OUT_PREFIX}_E{float(E_CUT):g}.json")
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[save] {out_json}")

    if MAKE_PLOT:
        plt.figure(figsize=(6.2, 4.0), dpi=160)
        plt.hist(100.0 * overlaps, bins=20, color="#2C7FB8", edgecolor="#1b1b1b", alpha=0.9)
        plt.xlabel("Projection onto stored modes ≤ E_CUT (%)")
        plt.ylabel("Count")
        plt.title(f"Completeness diagnostic (E_CUT={float(E_CUT):g}, trials={accepted})")
        plt.grid(True, axis="y", alpha=0.2)
        out_png = os.path.join(out_dir, f"{OUT_PREFIX}_E{float(E_CUT):g}_hist.png")
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        print(f"[save] {out_png}")


if __name__ == "__main__":
    main()


