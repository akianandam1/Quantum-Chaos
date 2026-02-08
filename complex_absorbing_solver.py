#!/usr/bin/env python3
# complex_absorbing_solver.py
#
# Build a Laplacian Hamiltonian on an SDF geometry (phi, allowed) as in
# lowband_complete_solver.py, but add an *imaginary absorbing potential*
# in a user-labelled red region read from a PNG.
#   - black: disallowed (outside domain)
#   - white: allowed, zero imaginary potential
#   - red:   allowed, with a smoothly ramped +i*W(x) absorber
#
# Output: eigenpairs_complex.h5 (evals [K] complex, evecs [n,K] complex)
#         geometry_used.npz, manifest.json, preview PNGs.

import os
import json
import time

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import eigs, splu, LinearOperator


# ========= USER PARAMS =========
SDF_PATH       = "potentials/isolated_small.npz"         # (phi, allowed) from png_to_sdf
ABSORBER_PNG   = "potentials/isolated_small_red.png"  # PNG with red absorber region
OUT_DIR        = "eigensdf/singleresonances/final_small"

N_MODES        = 1000          # how many lowest modes to compute (by Re(E))
IMAG_MAX       = 4.0          # maximum strength of imaginary potential W_max
RAMP_POWER     = 2.0          # ramp ~ (s^RAMP_POWER), s in [0,1]
# Ramp direction inside the red absorber region:
#   - "auto": infer white-side vs black-side per row (current behavior)
#   - "right": W increases as x increases (toward +x)
#   - "left":  W increases as x decreases (toward -x)
RAMP_DIRECTION = "right"

PREVIEW_GEOM_PNG     = os.path.join(OUT_DIR, "geom_allowed.png")
PREVIEW_ABS_PNG      = os.path.join(OUT_DIR, "absorber_mask.png")
PREVIEW_IMAG_PNG     = os.path.join(OUT_DIR, "imag_profile.png")
# =================================


# ======= GEOMETRY / LOADING =======
def load_phi_allowed(sdf_path):
    g = np.load(sdf_path)
    phi = g["phi"].astype(np.float32)
    if "allowed" in g:
        allowed = g["allowed"].astype(bool)
    else:
        allowed = phi > 0.0
    return phi, allowed


def load_absorber_from_png(png_path, shape):
    """
    Read an RGB PNG and extract a boolean mask for the red region.

    Convention:
      - red   (R>200, G<50, B<50) -> absorber region (True)
      - white (R,G,B all high)    -> allowed / no absorber
      - black (all low)           -> disallowed
    """
    try:
        from PIL import Image
        arr = np.array(Image.open(png_path))
    except Exception:
        import matplotlib.image as mpimg
        arr = mpimg.imread(png_path)

    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3]
        if rgb.dtype != np.uint8:
            rgb = (255 * rgb).astype(np.uint8)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        absorber = (r > 200) & (g < 50) & (b < 50)
    else:
        # grayscale; treat any positive value as non-absorber
        absorber = np.zeros(arr.shape, dtype=bool)

    if absorber.shape != shape:
        raise ValueError(f"Absorber PNG shape {absorber.shape} does not match phi shape {shape}.")

    return absorber


def save_preview_images(phi, allowed, absorber, imag_profile,
                        geom_path=None, abs_path=None, imag_path=None):
    Ny, Nx = phi.shape

    if geom_path is not None:
        plt.figure(figsize=(4, 4), dpi=130)
        plt.imshow(allowed, origin="lower", cmap="gray")
        plt.title("Allowed mask")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(geom_path, bbox_inches="tight")
        plt.close()

    if abs_path is not None:
        plt.figure(figsize=(4, 4), dpi=130)
        plt.imshow(absorber, origin="lower", cmap="Reds")
        plt.title("Absorber (red region)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(abs_path, bbox_inches="tight")
        plt.close()

    if imag_path is not None and imag_profile is not None:
        plt.figure(figsize=(4, 4), dpi=130)
        plt.imshow(imag_profile, origin="lower", cmap="inferno")
        plt.title("Imaginary potential W(x)")
        plt.colorbar(label="W")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(imag_path, bbox_inches="tight")
        plt.close()


# ======= IMAGINARY POTENTIAL PROFILE =======
def build_imag_profile(absorber_mask, allowed_mask, W_max=1.0, ramp_power=2.0, ramp_direction="auto"):
    """
    Build a smooth absorber profile W(y,x) on the *allowed* region that
    ramps horizontally across the red region.

    Picture (per row y):

        white  |  red .............. red  |  black
                x_start            x_end

    - At the white–red interface (left edge of red), W ≈ 0.
    - It increases monotonically as you move horizontally through red,
      reaching W_max near the red–black interface (right edge of red).

    Implementation:
      - For each row y, find all x where absorber_mask & allowed_mask is True.
      - Let x_start = min(x), x_end = max(x).
      - Define s = (x - x_start) / max(x_end - x_start, 1) in [0,1].
      - Use W = W_max * (s ** ramp_power).
    """
    Ny, Nx = allowed_mask.shape
    mask = absorber_mask & allowed_mask
    if not mask.any():
        return np.zeros_like(allowed_mask, dtype=np.float64)

    W = np.zeros_like(allowed_mask, dtype=np.float64)

    ramp_direction = str(ramp_direction).lower().strip()
    if ramp_direction not in ("auto", "left", "right"):
        raise ValueError("ramp_direction must be one of: 'auto', 'left', 'right'")

    for y in range(Ny):
        row = mask[y, :]
        if not row.any():
            continue

        xs_all = np.where(row)[0]
        # walk contiguous segments of red pixels in this row
        seg_start = xs_all[0]
        prev_x = xs_all[0]

        def process_segment(x0, x1):
            # x0..x1 inclusive are absorber pixels in this row
            # determine which side is white and which is black
            def classify_boundary(x_edge, direction):
                # direction: -1 for checking left neighbor, +1 for right neighbor
                x_out = x_edge + direction
                if x_out < 0 or x_out >= Nx:
                    # treat outside image as black wall
                    return "black"
                if not allowed_mask[y, x_out]:
                    return "black"
                if allowed_mask[y, x_out] and not absorber_mask[y, x_out]:
                    # allowed but not absorber -> white region
                    return "white"
                return None

            # Explicit direction override:
            # - right: ramp 0->1 from left edge to right edge
            # - left:  ramp 0->1 from right edge to left edge
            if ramp_direction == "right":
                white_x = x0
                black_x = x1
            elif ramp_direction == "left":
                white_x = x1
                black_x = x0
            else:
                # auto: infer white-side vs black-side from neighbors
                # Left boundary at x0, look to the left
                left_label = classify_boundary(x0, -1)
                # Right boundary at x1, look to the right
                right_label = classify_boundary(x1, +1)

                white_x = None
                black_x = None
                if left_label == "white":
                    white_x = x0
                elif left_label == "black":
                    black_x = x0

                if right_label == "white":
                    white_x = x1 if white_x is None else white_x
                elif right_label == "black":
                    black_x = x1 if black_x is None else black_x

                # Fallback: if we didn't clearly identify both sides,
                # just ramp from x0 to x1 (increasing to the right).
                if white_x is None or black_x is None or white_x == black_x:
                    white_x = x0
                    black_x = x1

            denom = abs(black_x - white_x)
            if denom == 0:
                # single-pixel absorber in this row
                if mask[y, white_x]:
                    W[y, white_x] = W_max
                return

            for x in range(x0, x1 + 1):
                if not mask[y, x]:
                    continue
                if white_x < black_x:
                    s = (x - white_x) / denom
                else:
                    s = (white_x - x) / denom
                s = float(np.clip(s, 0.0, 1.0))
                W[y, x] = W_max * (s ** ramp_power)

        for x in xs_all[1:]:
            if x == prev_x + 1:
                prev_x = x
                continue
            # close previous segment
            process_segment(seg_start, prev_x)
            seg_start = x
            prev_x = x

        # last segment in this row
        process_segment(seg_start, prev_x)

    return W


# ======= HAMILTONIAN WITH ABSORBER =======
def build_dirichlet_with_absorber(phi, allowed, imag_profile):
    """
    Build sparse Hamiltonian H with Dirichlet walls using the same
    finite-difference stencil as build_dirichlet_sw, but add a purely
    imaginary diagonal potential +i * W where imag_profile = W(x,y).
    """
    Ny, Nx = phi.shape
    idx = -np.ones((Ny, Nx), np.int64)
    idx[allowed] = np.arange(allowed.sum())
    n = int(allowed.sum())

    rows, cols, vals = [], [], []
    diag = np.zeros(n, np.complex128)

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
            # same smooth-wall correction as in build_dirichlet_sw
            phi_i = phi[y, x]
            phi_o = phi[yn, xn] if (0 <= yn < Ny and 0 <= xn < Nx) else -phi_i
            denom = (phi_i - phi_o)
            alpha = 0.5 if denom <= 0 else np.clip(phi_i / denom, 5e-3, 1.0)
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

            # add imaginary absorber on the diagonal if present
            if imag_profile is not None and allowed[y, x]:
                W = float(imag_profile[y, x])
                if W != 0.0:
                    diag[a] += 1j * W

    rows += list(range(n))
    cols += list(range(n))
    vals += list(diag)

    H = coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.complex128).tocsr()

    def pack(img2d):
        return img2d[allowed].reshape(-1)

    def unpack(vec):
        out = np.zeros((Ny, Nx), dtype=np.complex128)
        out[allowed] = vec.reshape(-1)
        return out

    return H, n, pack, unpack, allowed, (Ny, Nx)


# ======= STORAGE =======
def h5_open_complex(out_dir, n, mode="w"):
    os.makedirs(out_dir, exist_ok=True)
    h5p = os.path.join(out_dir, "eigenpairs_complex.h5")
    if (mode == "w") or (not os.path.exists(h5p)):
        with h5py.File(h5p, "w") as f:
            f.create_dataset("evals", shape=(0,), maxshape=(None,), dtype="c16")
            f.create_dataset(
                "evecs",
                shape=(n, 0),
                maxshape=(n, None),
                dtype="c16",
                chunks=(min(n, 8192), 1),
            )
    return h5py.File(h5p, "a")


def h5_append_complex(f, evals_new, evecs_new):
    K0 = f["evals"].shape[0]
    K1 = K0 + evals_new.shape[0]
    f["evals"].resize((K1,))
    f["evecs"].resize((f["evecs"].shape[0], K1))
    f["evals"][K0:K1] = np.asarray(evals_new, dtype=np.complex128)
    f["evecs"][:, K0:K1] = np.asarray(evecs_new, dtype=np.complex128)


# ======= EIGEN SOLVER (NON-HERMITIAN) =======
def solve_low_modes(H, k, sigma=0.0, tol=1e-8):
    """
    Compute k modes of complex, non-Hermitian H.

    We use a shift-invert strategy around a (real) sigma, so ARPACK
    effectively returns modes whose eigenvalues are *closest* to sigma.
    By default sigma=0 to get low-real-part modes.
    """
    k = min(k, H.shape[0] - 2)  # ARPACK requires k < N-1
    if k <= 0:
        raise ValueError("Requested N_MODES too small for matrix size.")

    n = H.shape[0]
    # (H - sigma I) factorization for shift-invert
    A = (H - sigma * eye(n, format="csr", dtype=np.complex128)).tocsc()
    LU = splu(A)

    def op(x):
        return LU.solve(x)

    OPinv = LinearOperator((n, n), matvec=op, dtype=np.complex128)

    # 'which=LM' on OPinv => eigenvalues of H nearest sigma
    evals, evecs = eigs(H, k=k, which="LM", sigma=sigma, OPinv=OPinv, tol=tol)
    # sort by real part (ascending)
    order = np.argsort(evals.real)
    return evals[order], evecs[:, order]


# ======= MAIN DRIVER =======
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Geometry
    phi, allowed = load_phi_allowed(SDF_PATH)
    Ny, Nx = phi.shape
    print(f"[geom] phi shape={phi.shape}, allowed cells={allowed.sum()}")

    absorber_raw = load_absorber_from_png(ABSORBER_PNG, phi.shape)
    absorber = absorber_raw & allowed  # ensure inside domain
    print(f"[geom] absorber pixels (inside allowed) = {absorber.sum()}")

    imag_profile = build_imag_profile(
        absorber, allowed, W_max=IMAG_MAX, ramp_power=RAMP_POWER, ramp_direction=RAMP_DIRECTION
    )

    save_preview_images(
        phi,
        allowed,
        absorber,
        imag_profile,
        geom_path=PREVIEW_GEOM_PNG,
        abs_path=PREVIEW_ABS_PNG,
        imag_path=PREVIEW_IMAG_PNG,
    )

    # Build Hamiltonian
    H, n, pack, unpack, allowed_mask, (Ny, Nx) = build_dirichlet_with_absorber(
        phi, allowed, imag_profile
    )
    print(f"[ham] built H of size {n}x{n} (complex)")

    # Save geometry snapshot
    np.savez_compressed(
        os.path.join(OUT_DIR, "geometry_used.npz"),
        phi=phi,
        allowed=allowed_mask,
        absorber=absorber,
        imag_profile=imag_profile,
        shape=np.array([Ny, Nx], np.int32),
    )

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(
            {
                "note": "complex_absorbing_solver",
                "SDF_PATH": SDF_PATH,
                "ABSORBER_PNG": ABSORBER_PNG,
                "N_MODES": N_MODES,
                "IMAG_MAX": IMAG_MAX,
                "RAMP_POWER": RAMP_POWER,
                "RAMP_DIRECTION": RAMP_DIRECTION,
            },
            f,
            indent=2,
        )

    # Solve eigenproblem
    print(f"[solve] computing {N_MODES} modes of complex H …")
    evals, evecs = solve_low_modes(H, N_MODES)
    K_total = evals.size
    print(f"[solve] finished eigensolve, got {K_total} modes; writing to disk in chunks …")

    # Store with simple progress updates
    f = h5_open_complex(OUT_DIR, n, mode="w")
    BATCH_SAVE = 50
    written = 0
    for start in range(0, K_total, BATCH_SAVE):
        stop = min(start + BATCH_SAVE, K_total)
        h5_append_complex(f, evals[start:stop], evecs[:, start:stop])
        written = stop
        print(f"[store] saved {written}/{K_total} eigenstates so far")
    f.flush()
    f.close()

    # Quick diagnostic plot: spectrum in complex plane
    plt.figure(figsize=(5, 4), dpi=130)
    plt.scatter(evals.real, evals.imag, s=20)
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("Re(E)")
    plt.ylabel("Im(E)")
    plt.title("Complex spectrum with absorber")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "spectrum_complex_plane.png"), bbox_inches="tight")
    plt.close()

    print(
        f"[done] stored {evals.size} modes; "
        f"Re(E) in [{evals.real.min():.6g}, {evals.real.max():.6g}], "
        f"Im(E) in [{evals.imag.min():.6g}, {evals.imag.max():.6g}]"
    )
    print(f"Elapsed {time.time() - t0:.1f} s -> {OUT_DIR}")


if __name__ == "__main__":
    main()


