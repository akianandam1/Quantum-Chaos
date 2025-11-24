#!/usr/bin/env python3
# standing_wave_polygon_energy.py
#
# Build a standing wave localized inside a user-specified quadrilateral
# (4 points in pixel coordinates), SHOW a preview over the potential,
# then compute the expectation value <H> using a finite-difference Laplacian.
#
# Optional: compute overlaps with your eigenbasis (set DO_OVERLAPS=True).
#
# ------------------------- HOW TO USE -------------------------
# 1) Edit the CONFIG block below (paths, polygon points, modes, etc.).
# 2) Run the file directly (no CLI flags needed).
#    A matplotlib preview window appears first; close it to continue.
#
# Notes:
# - Coordinates are in *pixel units* on the SDF grid: x = column, y = row.
# - The standing wave uses sin(kx*(x-xmin)) * sin(ky*(y-ymin)) inside the polygon,
#   with kx = MX*pi/Lx, ky = MY*pi/Ly where Lx, Ly are the polygon's bbox extent.
# - Outside the polygon (or outside the allowed region) ψ = 0.
# - The Laplacian uses 5-point stencil with Dirichlet zero outside the grid/allowed,
#   so it is consistent with infinite-well boundaries.
#
# --------------------------------------------------------------

CONFIG = {
    # -------- Files --------
    "NPZ_PATH":  r"potentials/potential7.npz",            # contains 'phi' and 'allowed' (or allowed = phi>0)
    "H5_PATH":   r"eigensdf/doublewell/trial1/eigenpairs.h5",  # used only if DO_OVERLAPS=True
    "OUT_DIR":   r"eigensdf/doublewell/trial1",
    "SAVE_CSV":  True,          # if DO_OVERLAPS
    "TOP_K":     20,            # if DO_OVERLAPS

    # -------- Preview --------
    "SHOW_PREVIEW": True,       # blocking matplotlib window
    "PREVIEW_STYLE": "filled",  # "filled" or "outline" for allowed region
    "ALLOWED_ALPHA": 0.22,      # opacity for allowed fill
    "OUTLINE_WIDTH": 5,         # white outline thickness (px)
    "PREVIEW_CMAP": "magma",    # colormap for |psi0|

    # -------- Standing wave in polygon --------
    # Four polygon vertices (x, y) in pixel coords; order can be CW or CCW.
    "POLY_POINTS": [(50,150),(50,190),(90,150),(90,190)], #channel [(150, 371), (150, 389), (190, 371), (190, 390)],
    "MX": 3,                    # modes along x (half-wavelength count across polygon bbox)
    "MY": 5,                    # modes along y

    # -------- Discretization / physics --------
    "DX": 1.0,                  # pixel spacing in x (if your solver used Δx=Δy=1, leave as 1.0)
    "DY": 1.0,                  # pixel spacing in y
    "HBAR2_OVER_2M": 1.0,       # set to your solver’s value; default dimensionless = 1

    # -------- Optional eigen-overlaps --------
    "DO_OVERLAPS": True,        # set True to compute overlaps with eigenbasis
}

import os, csv, json
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy import ndimage as ndi
import time

# ---------------- I/O ----------------
def load_npz(npz_path):
    g = np.load(npz_path)
    phi = g["phi"].astype(np.float32)
    allowed = g["allowed"].astype(bool) if "allowed" in g else (phi > 0)
    return phi, allowed

# -------------- Geometry helpers --------------
def make_coords(shape):
    Ny, Nx = shape
    x = np.arange(Nx, dtype=np.float64)
    y = np.arange(Ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y)  # X[row=y, col=x]
    return X, Y

def polygon_mask(shape, points):
    """Boolean mask of pixels whose centers lie inside the polygon defined by 'points'."""
    X, Y = make_coords(shape)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    path = MplPath(points)
    inside = path.contains_points(pts, radius=0.0)
    return inside.reshape(shape)

def compute_outline(mask, width=1):
    width = max(1, int(width))
    eroded = ndi.binary_erosion(mask, iterations=width, border_value=0)
    outline = mask & ~eroded
    return outline

# -------------- Standing wave builder --------------
def standing_wave_in_polygon(shape, points, mx, my):
    """
    Build separable standing wave sin(kx*(x-xmin)) * sin(ky*(y-ymin))
    inside polygon; bbox extent sets kx, ky.
    """
    Ny, Nx = shape
    pts = np.asarray(points, dtype=np.float64)
    px_min, py_min = np.min(pts, axis=0)
    px_max, py_max = np.max(pts, axis=0)
    Lx = max(1.0, px_max - px_min)
    Ly = max(1.0, py_max - py_min)

    X, Y = make_coords(shape)
    kx = mx * np.pi / Lx
    ky = my * np.pi / Ly
    base = np.sin(kx * (X - px_min)) * np.sin(ky * (Y - py_min))

    poly = polygon_mask(shape, pts)
    psi = np.where(poly, base, 0.0).astype(np.complex128)
    return psi, poly, (kx, ky)

# -------------- Laplacian & <H> --------------
def laplacian_dirichlet(psi, allowed, dx=1.0, dy=1.0):
    """
    5-point Laplacian with Dirichlet zero outside the grid and outside 'allowed'.
    """
    psi_in = np.where(allowed, psi, 0.0)

    # shift helpers (zero padding)
    def shift(arr, axis, direction):
        out = np.zeros_like(arr, dtype=arr.dtype)
        if axis == 0:
            if direction > 0:
                out[1:, :] = arr[:-1, :]
            else:
                out[:-1, :] = arr[1:, :]
        else:
            if direction > 0:
                out[:, 1:] = arr[:, :-1]
            else:
                out[:, :-1] = arr[:, 1:]
        return out

    psi_up    = shift(psi_in, axis=0, direction=+1)
    psi_down  = shift(psi_in, axis=0, direction=-1)
    psi_left  = shift(psi_in, axis=1, direction=+1)
    psi_right = shift(psi_in, axis=1, direction=-1)

    lap = (psi_left + psi_right - 2.0*psi_in) / (dx*dx) + (psi_up + psi_down - 2.0*psi_in) / (dy*dy)

    # Outside allowed, enforce zero
    lap = np.where(allowed, lap, 0.0)
    return lap

def grad_energy(psi, allowed, dx=1.0, dy=1.0):
    """
    Equivalent kinetic energy from |∇ψ|^2: ∫ (|∂xψ|^2 + |∂yψ|^2) dxdy
    using centered differences (Dirichlet outside allowed).
    """
    psi_in = np.where(allowed, psi, 0.0).astype(np.complex128)

    def shift(arr, axis, direction):
        out = np.zeros_like(arr, dtype=arr.dtype)
        if axis == 0:
            if direction > 0:
                out[1:, :] = arr[:-1, :]
            else:
                out[:-1, :] = arr[1:, :]
        else:
            if direction > 0:
                out[:, 1:] = arr[:, :-1]
            else:
                out[:, :-1] = arr[:, 1:]
        return out

    psi_xp = shift(psi_in, 1, +1)
    psi_xm = shift(psi_in, 1, -1)
    psi_yp = shift(psi_in, 0, +1)
    psi_ym = shift(psi_in, 0, -1)

    dpsi_dx = (psi_xp - psi_xm) / (2.0*dx)
    dpsi_dy = (psi_yp - psi_ym) / (2.0*dy)

    gx2 = np.abs(dpsi_dx)**2
    gy2 = np.abs(dpsi_dy)**2

    gx2 = np.where(allowed, gx2, 0.0)
    gy2 = np.where(allowed, gy2, 0.0)

    integral = np.sum(gx2 + gy2) * dx * dy
    return integral

def expectation_H(psi, allowed, dx=1.0, dy=1.0, hbar2_over_2m=1.0):
    """
    <H> = <ψ| (-ħ²/2m ∇²) |ψ>  for V=0 in the allowed region (infinite walls).
    """
    lap = laplacian_dirichlet(psi, allowed, dx=dx, dy=dy)
    Hpsi = -hbar2_over_2m * lap
    val = np.sum(np.conjugate(psi) * Hpsi).real * dx * dy
    return val

# -------------- Overlaps (optional) --------------
def overlaps_with_eigenbasis(h5_path, allowed, psi2D, top_k=20):
    psi_vec = psi2D[allowed].astype(np.complex128)
    with h5py.File(h5_path, "r") as f:
        evals = np.array(f["evals"][:], dtype=np.float64)
        evecs = f["evecs"]
        K = evecs.shape[1]
        batch = 4096 if K > 4096 else K
        c = np.empty(K, dtype=np.complex128)
        for i0 in range(0, K, batch):
            i1 = min(K, i0 + batch)
            V = np.array(evecs[:, i0:i1], dtype=np.complex128)
            c[i0:i1] = np.conjugate(V).T @ psi_vec
    w = np.abs(c)**2
    order = np.argsort(w)[::-1]
    return c, w, evals, order[:top_k]

# -------------- Preview --------------
def show_preview(allowed, psi2D, poly_mask, cfg, kxky=None):
    amp = np.abs(psi2D)
    vmax = np.percentile(amp[allowed], 99.5) if np.any(allowed) else (amp.max() if amp.size else 1.0)
    if vmax <= 0: vmax = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 6.8), dpi=140)
    ax.set_facecolor("black")

    # Allowed region (white)
    if cfg["PREVIEW_STYLE"] == "outline":
        outline = compute_outline(allowed, width=cfg.get("OUTLINE_WIDTH", 2))
        ax.imshow(np.where(outline, 1.0, np.nan), origin="lower", cmap="gray", vmin=0, vmax=1, alpha=1.0)
    else:
        ax.imshow(np.where(allowed, 1.0, 0.0), origin="lower", cmap="gray", vmin=0, vmax=1, alpha=cfg.get("ALLOWED_ALPHA", 0.22))
        outline = compute_outline(allowed, width=cfg.get("OUTLINE_WIDTH", 2))
        ax.imshow(np.where(outline, 1.0, np.nan), origin="lower", cmap="gray", vmin=0, vmax=1, alpha=1.0)

    # Polygon outline (cyan)
    poly_edge = compute_outline(poly_mask, width=2)
    ax.imshow(np.where(poly_edge, 1.0, np.nan), origin="lower", cmap="winter", vmin=0, vmax=1, alpha=0.9)

    # |psi| heatmap
    ax.imshow(amp, origin="lower", cmap=cfg["PREVIEW_CMAP"], alpha=0.95, vmin=0.0, vmax=vmax)

    if kxky is not None:
        kx, ky = kxky
        ax.set_title(f"|psi0|  (standing wave in polygon, kx={kx:.3f}, ky={ky:.3f})")
    else:
        ax.set_title("|psi0|  (standing wave in polygon)")

    ax.axis("off")
    fig.tight_layout()
    plt.show(block=True)

# -------------- Main --------------
def main():
    cfg = CONFIG
    os.makedirs(cfg["OUT_DIR"], exist_ok=True)
    phi, allowed = load_npz(cfg["NPZ_PATH"])

    # Standing wave in polygon (before applying 'allowed')
    psi2D, poly_mask, kxky = standing_wave_in_polygon(
        phi.shape, cfg["POLY_POINTS"], int(cfg["MX"]), int(cfg["MY"])
    )

    # Enforce infinite walls and normalize
    psi2D = np.where(allowed, psi2D, 0.0)
    norm = np.sqrt(np.sum(np.abs(psi2D)**2) + 1e-30)
    psi2D /= norm

    # Preview
    if cfg["SHOW_PREVIEW"]:
        show_preview(allowed, psi2D, poly_mask, cfg, kxky=kxky)

    # <H> from Laplacian
    DX, DY = float(cfg["DX"]), float(cfg["DY"])
    H2O2M = float(cfg["HBAR2_OVER_2M"])
    H_exp = expectation_H(psi2D, allowed, dx=DX, dy=DY, hbar2_over_2m=H2O2M)

    # Cross-check via gradient form (should be close numerically)
    T_grad = H2O2M * grad_energy(psi2D, allowed, dx=DX, dy=DY)

    print("\n=== Energy diagnostics ===")
    print(f"<H> via Laplacian      : {H_exp:.10f}")
    print(f"∫ (|∂ψ/∂x|^2+|∂ψ/∂y|^2) : {T_grad:.10f}  (× ħ²/2m already applied)")
    print(f"abs(diff))             : {abs(H_exp - T_grad):.3e}")

    # Optional overlaps with eigenbasis
    if cfg.get("DO_OVERLAPS", False):
        c, w, evals, top_idx = overlaps_with_eigenbasis(cfg["H5_PATH"], allowed, psi2D, top_k=cfg["TOP_K"])
        S = float(np.sum(w))
        expE_from_overlaps = float(np.sum(w * evals) / (S + 1e-30)) if evals.size == w.size else None

        print(f"\nSum |c_n|^2 = {S:.8f}  (should be ~1.0)")
        if expE_from_overlaps is not None:
            print(f"<E> from overlaps   = {expE_from_overlaps:.10f}")

        print(f"\nTop {len(top_idx)} contributors:")
        for rank, n in enumerate(top_idx, 1):
            print(f"{rank:2d}. n={n:6d}  |c_n|^2={w[n]:.6e}   E_n={evals[n]:.10f}")

        if cfg["SAVE_CSV"]:
            csv_path = os.path.join(cfg["OUT_DIR"], "polygon_overlaps.csv")
            with open(csv_path, "w", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(["index_n", "abs2_c_n", "E_n"])
                for n in range(w.size):
                    writer.writerow([n, w[n], evals[n] if n < len(evals) else ""])
            print(f"[Saved] overlaps CSV -> {csv_path}")

    # Save JSON summary
    summary = {
        "poly_points": cfg["POLY_POINTS"],
        "mx": int(cfg["MX"]), "my": int(cfg["MY"]),
        "kxky": {"kx": float(kxky[0]), "ky": float(kxky[1])},
        "dx": float(cfg["DX"]), "dy": float(cfg["DY"]), "hbar2_over_2m": float(cfg["HBAR2_OVER_2M"]),
        "H_expectation": float(H_exp),
        "T_grad_check": float(T_grad),
    }
    if cfg.get("DO_OVERLAPS", False):
        summary["did_overlaps"] = True
    os.makedirs(cfg["OUT_DIR"], exist_ok=True)
    js_path = os.path.join(cfg["OUT_DIR"], "polygon_energy_summary.json")
    with open(js_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\n[Saved] JSON summary -> {js_path}")

if __name__ == "__main__":
    start=time.time()
    main()
    end=time.time()
    print(f"Finished in {end-start} seconds")
