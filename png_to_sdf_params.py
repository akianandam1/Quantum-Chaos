#!/usr/bin/env python3
# png_to_sdf_params.py
# Make a smooth signed-distance field (phi) from a black/white PNG (white=inside).

# ===================== USER PARAMETERS =====================
IMG_PATH        = "potentials/potential5.png"     # white = allowed, black = wall
OUT_NPZ         = "potentials/potential5.npz"
PREVIEW_PNG     = "potentials/potential5sdf.png"

# Target solver grid
GRID_N          = 512                 # final grid Ny = Nx = GRID_N

# High-res rasterization (bigger -> smoother SDF; paid once)
SUPERSAMPLE     = 24                  # 16–24 is very clean
BLUR_SIGMA      = 0.9                 # pre-contour blur (0.4–0.9 helps)
SPLINE_SMOOTH   = 0.005               # B-spline smoothing; raise if outline is noisy
SPLINE_POINTS   = 8000                # samples on the smooth closed curve
CONTOUR_LEVEL   = 0.5                 # isovalue for finding the boundary in [0,1]
# ============================================================

import os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import skimage.measure as skmeasure
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def png_to_sdf(img_path: str,
               out_npz: str,
               preview_png: str,
               N: int,
               supersample: int,
               blur_sigma: float,
               spline_smooth: float,
               spline_pts: int,
               contour_level: float = 0.5):
    """
    Build a smooth signed distance field phi from a black/white PNG.
    Returns (phi, occ) where:
      - phi: float32 (N,N), signed distance in *final-grid cell units* (>0 inside)
      - occ: float32 (N,N), fractional occupancy (anti-aliased, optional)
    Also writes NPZ and preview PNG.
    """
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)

    # 1) Load & upsample strongly
    Hi = N * supersample
    im = Image.open(img_path).convert("L")
    arr = np.asarray(im.resize((Hi, Hi), Image.LANCZOS), dtype=np.float32) / 255.0
    if blur_sigma > 0:
        arr = ndi.gaussian_filter(arr, blur_sigma)

    # 2) Find the main contour (white region) and fit a periodic cubic spline
    contours = skmeasure.find_contours(arr, level=contour_level)
    if not contours:
        raise RuntimeError("No contours found; check PNG polarity / level.")
    def interior_mean(c):
        yx = np.mean(c, axis=0)
        y = int(np.clip(yx[0], 0, arr.shape[0]-1))
        x = int(np.clip(yx[1], 0, arr.shape[1]-1))
        return arr[y, x]
    contours.sort(key=lambda c: -interior_mean(c))
    c = contours[0]
    y, x = c[:, 0], c[:, 1]
    if np.hypot(x[0]-x[-1], y[0]-y[-1]) > 1e-6:
        x = np.r_[x, x[0]]; y = np.r_[y, y[0]]

    tck, _ = splprep([x, y], s=spline_smooth*len(x), per=True, k=3)
    u = np.linspace(0, 1, spline_pts, endpoint=False)
    xs, ys = splev(u, tck)

    # 3) Rasterize the smooth polygon at very high resolution
    poly = [(float(xx), float(yy)) for xx, yy in zip(xs, ys)]
    canvas = Image.new("L", (Hi, Hi), 0)
    ImageDraw.Draw(canvas, "L").polygon(poly, outline=255, fill=255)
    filled_hi = np.asarray(canvas, dtype=np.float32) / 255.0

    # Keep largest component (safety)
    lab, n = ndi.label(filled_hi > 0.5)
    if n >= 1:
        sizes = ndi.sum(filled_hi > 0.5, lab, index=np.arange(1, n+1))
        keep = 1 + int(np.argmax(sizes))
        filled_hi = (lab == keep).astype(np.float32)

    # 4) Compute SDF at high-res and downsample to the final grid
    inside_hi = filled_hi > 0.5
    d_in  = ndi.distance_transform_edt(inside_hi).astype(np.float32)
    d_out = ndi.distance_transform_edt(~inside_hi).astype(np.float32)
    phi_hi = d_in - d_out  # >0 inside, <0 outside (units: hi-res pixels)

    # Block-average down to N×N; convert pixels -> final-grid cells
    Hc = Hi // supersample * supersample
    phi_hi = phi_hi[:Hc, :Hc]
    filled_hi = filled_hi[:Hc, :Hc]
    block = supersample

    def block_mean(a):
        return a.reshape(Hc//block, block, Hc//block, block).mean(axis=(1,3))

    occ = block_mean(filled_hi).astype(np.float32)
    phi = (block_mean(phi_hi) / supersample).astype(np.float32)  # signed distance in cell units

    # 5) Save NPZ and an accurate preview of the solver boundary
    np.savez_compressed(out_npz, phi=phi, occ=occ,
                        meta=np.array([N, supersample, blur_sigma, spline_smooth, spline_pts], dtype=np.float32))
    print(f"[SDF] wrote {out_npz}  (phi shape={phi.shape})")

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), dpi=150)

    # (A) fractional occupancy (anti-aliased area average)
    ax[0].imshow(occ, origin='lower', cmap='gray', vmin=0, vmax=1)
    ax[0].set_title("fractional occupancy")
    ax[0].axis('off')

    # (B) signed distance field
    im1 = ax[1].imshow(phi, origin='lower', cmap='coolwarm')
    ax[1].set_title("signed distance ϕ (cells)")
    ax[1].axis('off')
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    # (C) what the solver uses: smooth fill from φ with φ=0 overlay
    # filled regions from φ<0 (black) and φ>0 (white); no pixel stair-steps
    levels = [phi.min() - 1e-9, 0.0, phi.max() + 1e-9]
    ax[2].contourf(phi, levels=levels, colors=['black', 'white'], antialiased=True)
    # overlay the zero level-set curve

    for c in skmeasure.find_contours(phi, level=0.0):
        ax[2].plot(c[:, 1], c[:, 0], 'r-', lw=1.2, alpha=0.9)
    ax[2].set_title("smooth Dirichlet boundary (φ=0)")
    ax[2].axis('off')

    plt.tight_layout()
    plt.savefig(preview_png, bbox_inches="tight")
    plt.close()
    print(f"[SDF] wrote preview {preview_png}")

    return phi, occ

def main():
    png_to_sdf(IMG_PATH, OUT_NPZ, PREVIEW_PNG,
               N=GRID_N,
               supersample=SUPERSAMPLE,
               blur_sigma=BLUR_SIGMA,
               spline_smooth=SPLINE_SMOOTH,
               spline_pts=SPLINE_POINTS,
               contour_level=CONTOUR_LEVEL)

if __name__ == "__main__":
    main()