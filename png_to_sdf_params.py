# Build a smooth signed-distance field (phi) from a black/white PNG (white=inside).

# ===================== USER PARAMETERS =====================
IMG_PATH        = "potentials/isolated_large.png"   # your PNG (white = allowed, black = wall)
OUT_NPZ         = "potentials/isolated_large.npz"
PREVIEW_PNG     = "potentials/isolated_large_sdf.png"

GRID_N          = 1024        # final solver grid (Ny = Nx)
SUPERSAMPLE     = 24         # hi-res rasterization for smooth curves (16–24 is very clean)
BLUR_SIGMA      = 0.9        # slight blur before contouring
SPLINE_SMOOTH   = 0.005      # B-spline smoothing factor (increase if outline is jaggy)
SPLINE_POINTS   = 8000       # samples along smoothed closed curve
CONTOUR_LEVEL   = 0.5        # isovalue in [0,1] for finding boundary

# --- robust PNG→mask options ---
POLARITY        = "white_inside"   # "white_inside" | "black_inside" | "auto"
USE_ALPHA       = True             # prefer alpha if present *and informative*
GAUSS_EDGE_BLUR = 0.8              # small pre-threshold blur (px); 0 disables
ALPHA_MIN_VAR   = 1e-6             # if alpha variance < this, ignore alpha channel
# ============================================================

import os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import skimage.measure as skmeasure
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import imageio.v3 as iio


def _as01(a):
    a = np.asarray(a, dtype=np.float32)
    return a/255.0 if a.max() > 1.5 else a


def otsu_threshold(g01):
    g = np.asarray(g01, dtype=np.float32)
    g = np.clip(g, 0, 1)
    hist, edges = np.histogram(g, bins=256, range=(0, 1))
    w1 = np.cumsum(hist)
    w2 = hist.sum() - w1
    m1 = np.cumsum(hist * edges[:-1])
    m2 = m1[-1] - m1
    num = (m1[-1] * w1 - m1) ** 2
    den = w1 * w2 + 1e-20
    k = int(np.argmax(num / den))
    return float(edges[k])


def load_mask_from_png(path, polarity="auto", use_alpha=True,
                       blur_sigma=0.8, alpha_min_var=1e-6):
    """Return boolean 'inside' mask where True = allowed (white). Robust to bad alpha."""
    im = iio.imread(path)  # (H,W), (H,W,3) or (H,W,4)
    H, W = im.shape[:2]

    # Choose a scalar field to threshold
    if im.ndim == 2:
        g = _as01(im)
    elif im.shape[2] == 4 and use_alpha:
        a = _as01(im[..., 3])
        # If alpha is nearly constant, fall back to luminance
        if float(a.var()) > alpha_min_var:
            g = a
        else:
            rgb = im[..., :3].astype(np.float32)
            g = _as01(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])
    else:
        rgb = im[..., :3].astype(np.float32)
        g = _as01(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])

    # Optional slight blur to anti-alias edges
    if blur_sigma and blur_sigma > 0:
        g = ndi.gaussian_filter(g, blur_sigma)

    # Threshold (Otsu)
    thr = otsu_threshold(g)
    bw = (g >= thr)  # True ~ “white-ish”

    # Decide polarity
    if polarity == "white_inside":
        inside = bw
    elif polarity == "black_inside":
        inside = ~bw
    else:
        touches_white = bw[0, :].any() or bw[-1, :].any() or bw[:, 0].any() or bw[:, -1].any()
        touches_black = (~bw)[0, :].any() or (~bw)[-1, :].any() or (~bw)[:, 0].any() or (~bw)[:, -1].any()
        if touches_white and not touches_black:
            inside = ~bw
        elif touches_black and not touches_white:
            inside = bw
        else:
            # tie-breaker: prefer the smaller area as “inside”
            inside = bw if bw.sum() < (~bw).sum() else ~bw

    # Final sanity: ensure some interior area but not full frame
    area = int(inside.sum())
    if area == 0 or area == H * W:
        raise ValueError("PNG→mask failed: interior is empty or full (after polarity/alpha checks).")

    return inside


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
    Build a smooth SDF phi from a black/white PNG.
      - phi: float32 (N,N), signed distance in final-grid *cells* (>0 inside)
      - allowed: uint8 (N,N), 1=inside mask
    Also saves a 3-panel preview (occupancy, phi heatmap, φ=0 boundary).
    """
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)

    # 1) Load → resize strongly for smooth contouring
    Hi = N * supersample
    im = Image.open(img_path).convert("L")
    arr = np.asarray(im.resize((Hi, Hi), Image.LANCZOS), dtype=np.float32) / 255.0
    if blur_sigma > 0:
        arr = ndi.gaussian_filter(arr, blur_sigma)

    # 2) Extract main contour and fit a periodic cubic spline
    contours = skmeasure.find_contours(arr, level=contour_level)
    if not contours:
        raise RuntimeError("No contours found; check PNG polarity/level or increase BLUR_SIGMA.")
    def interior_mean(c):
        yx = np.mean(c, axis=0)
        y = int(np.clip(yx[0], 0, arr.shape[0] - 1))
        x = int(np.clip(yx[1], 0, arr.shape[1] - 1))
        return arr[y, x]
    contours.sort(key=lambda c: -interior_mean(c))
    c = contours[0]
    y, x = c[:, 0], c[:, 1]
    # ensure closed
    if np.hypot(x[0] - x[-1], y[0] - y[-1]) > 1e-6:
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]

    tck, _ = splprep([x, y], s=spline_smooth * len(x), per=True, k=3)
    u = np.linspace(0, 1, spline_pts, endpoint=False)
    xs, ys = splev(u, tck)

    # 3) Rasterize the smooth polygon at very high resolution
    poly = [(float(xx), float(yy)) for xx, yy in zip(xs, ys)]
    canvas = Image.new("L", (Hi, Hi), 0)
    ImageDraw.Draw(canvas, "L").polygon(poly, outline=255, fill=255)
    filled_hi = np.asarray(canvas, dtype=np.float32) / 255.0

    # Keep largest connected component (safety)
    lab, n = ndi.label(filled_hi > 0.5)
    if n >= 1:
        sizes = ndi.sum(filled_hi > 0.5, lab, index=np.arange(1, n + 1))
        keep = 1 + int(np.argmax(sizes))
        filled_hi = (lab == keep).astype(np.float32)

    # 4) Build inside mask robustly (handles alpha edge-cases)
    inside_full = load_mask_from_png(img_path, polarity=POLARITY,
                                     use_alpha=USE_ALPHA,
                                     blur_sigma=GAUSS_EDGE_BLUR,
                                     alpha_min_var=ALPHA_MIN_VAR)

    # Downsample to final grid (area-average). For binary mask, simple resize is fine:
    inside_img = Image.fromarray((inside_full.astype(np.uint8) * 255))
    inside_ds = inside_img.resize((N, N), Image.NEAREST)
    inside = (np.asarray(inside_ds, dtype=np.uint8) > 127)

    # Signed Euclidean distance (φ>0 inside)
    from scipy.ndimage import distance_transform_edt as edt
    d_in = edt(inside)
    d_out = edt(~inside)
    phi = (d_in - d_out).astype(np.float32)

    # Save
    np.savez_compressed(out_npz, phi=phi, allowed=inside.astype(np.uint8))
    print(f"[SDF] wrote {out_npz}  (phi shape={phi.shape})")

    # Preview
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), dpi=150)
    ax[0].imshow(inside.astype(np.float32), origin="lower", cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("fractional/occupancy (downsampled mask)"); ax[0].axis("off")

    imh = ax[1].imshow(phi, origin="lower", cmap="coolwarm")
    ax[1].set_title("signed distance φ (cells)"); ax[1].axis("off")
    fig.colorbar(imh, ax=ax[1], fraction=0.046, pad=0.04)

    lo, hi = float(np.nanmin(phi)), float(np.nanmax(phi))
    eps = 1e-9
    if lo < 0.0 < hi:
        levels = [min(lo, 0.0) - eps, 0.0, max(hi, 0.0) + eps]
        ax[2].contourf(phi, levels=levels, colors=["black", "white"], antialiased=True)
        ax[2].contour(phi, levels=[0.0], colors=["red"], linewidths=1.5)
        ax[2].set_title("smooth Dirichlet boundary (φ=0)")
    else:
        ax[2].imshow((phi > 0).astype(float), origin="lower", cmap="gray", vmin=0, vmax=1)
        ax[2].set_title("mask from φ (fallback)")
    ax[2].axis("off")

    fig.tight_layout()
    if preview_png:
        fig.savefig(preview_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[SDF] wrote preview {preview_png}")

    return phi, inside


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