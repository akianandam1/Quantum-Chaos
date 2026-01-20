#!/usr/bin/env python3
"""
render_one_mode.py

Config-at-the-top script to:
  - load a single eigenstate (by index) from an eigenpairs HDF5 database
  - print its precise energy (real or complex)
  - render the eigenstate to a PNG, overlayed on the geometry background

It tries to use `geometry_used.npz` located next to the HDF5 file (same folder).
If that is missing, it falls back to an SDF NPZ (phi/allowed).
"""

import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# ========= USER CONFIG (edit these) =========
# Path to eigenpairs database (.h5)
H5_PATH = r"eigensdf/doublewell/final/eigenpairs.h5"

# Index of the eigenstate in the database (0-based)
MODE_INDEX = 923

# If True: interpret MODE_INDEX after sorting by energy key (Re(E) or |E|)
USE_SORTED_INDEX = True
SORT_KEY = "real"  # "real" or "abs"

# Output image (None -> auto next to H5_PATH)
OUT_PNG = None

# Geometry fallback (used only if geometry_used.npz is missing)
SDF_FALLBACK_PATH = r"potentials/double_potential_final.npz"

# What to render
RENDER_REAL = False
RENDER_PROB = True

# Look / overlay
OUTPUT_PIXELS = 1024      # set to 1024 to match a 1024x1024 grid nicely
FIG_DPI = 256             # together with OUTPUT_PIXELS this yields an exact pixel size
FIGSIZE = None            # None -> computed from OUTPUT_PIXELS (or from grid if OUTPUT_PIXELS is None)
SAVE_TIGHT = False        # False preserves exact pixel size; True may change output size
SHOW_TITLE = False        # titles change layout/pixels; keep False for exact-size renders
ALPHA_OVERLAY = 0.35   # blend eigenimage over background (0..1)
SHOW_BACKGROUND = True
INTERPOLATION = "bilinear"  # "bilinear" helps avoid jagged pixel edges in the visualization

# Background colors (outside stays black; interior can be styled to “pop” more than white)
BG_OUT_RGB = (0, 0, 0)          # disallowed region stays black
BG_IN_RGB = (70, 80, 95)        # mid-slate interior (lighter but not white). Try (55,65,80) or (90,100,115)
BG_EDGE_SOFTEN = True           # use SDF smoothing if phi is available

# Smooth wall rendering (recommended)
DRAW_BOUNDARY_CONTOUR = True
BOUNDARY_COLOR = "#111111"
BOUNDARY_LW = 1.0

# Probability contrast (simple, “hotspots pop”)
# Some good high-contrast choices to try (keep outside region black):
#   - "turbo"   (very vivid, great separation)
#   - "inferno" (high contrast, classic)
#   - "magma"   (slightly softer than inferno)
#   - "plasma"  (vivid, warmer)
#   - "viridis" (clean, more subdued)
# How to color the density overlay:
#   - "cmap": use PROB_CMAP for coloring
#   - "mono": use a single RGB color, with density controlling transparency (very “pop”)
PROB_STYLE = "mono"  # "mono" or "cmap"

PROB_CMAP = "plasma"
PROB_PCT_HI = 99.5
PROB_GAMMA = 0.60

# Make low density "uncolored" (transparent) and only color high density.
# This threshold is applied after the percentile+gamma normalization step (so it is in [0,1]).
PROB_SHOW_MIN = 0.10          # higher -> only color peaks (more contrast)
PROB_ALPHA_GAMMA = 2.2        # >1 makes only the very top densities opaque (more pop)

# Used only for PROB_STYLE="mono"
PROB_MONO_RGB = (0, 170, 255)  # electric blue/cyan (pops well on black/white)

# Optional absorber/potential overlay tint (if geometry_used.npz contains absorber/imag_profile)
SHOW_ABSORBER_OVERLAY = False
ABSORBER_ALPHA = 0.45

# Output
SAVE_PNG = True
# ===========================================


def _load_geometry_near_h5(h5_path: str):
    """
    Returns: (Ny, Nx, allowed, occ, absorber_mask, imag_profile, phi)
    where absorber_mask/imag_profile/phi may be None.
    """
    out_dir = os.path.dirname(os.path.abspath(h5_path))
    geom_used = os.path.join(out_dir, "geometry_used.npz")

    if os.path.exists(geom_used):
        g = np.load(geom_used)
        Ny, Nx = g["shape"]
        allowed = g["allowed"].astype(bool)
        occ = g["occ"].astype(np.float32) if "occ" in g.files else allowed.astype(np.float32)
        absorber = g["absorber"].astype(bool) if "absorber" in g.files else None
        imag_profile = g["imag_profile"].astype(np.float64) if "imag_profile" in g.files else None
        phi = g["phi"].astype(np.float32) if "phi" in g.files else None
        return int(Ny), int(Nx), allowed, occ, absorber, imag_profile, phi

    # fallback to SDF pack (phi>0 defines interior)
    g = np.load(SDF_FALLBACK_PATH)
    phi = g["phi"].astype(np.float32) if "phi" in g.files else None
    if "allowed" in g:
        allowed = g["allowed"].astype(bool)
    else:
        if phi is None:
            raise ValueError("Fallback SDF must contain 'phi' if 'allowed' is not present.")
        allowed = phi > 0.0
    Ny, Nx = allowed.shape
    occ = g["occ"].astype(np.float32) if "occ" in g.files else allowed.astype(np.float32)
    return int(Ny), int(Nx), allowed, occ, None, None, phi


def _unpack(vec: np.ndarray, allowed: np.ndarray, shape):
    Ny, Nx = shape
    out = np.zeros((Ny, Nx), dtype=np.complex128)
    out[allowed] = vec.reshape(-1)
    return out


def _format_energy(E):
    # print precise complex/real value
    if np.iscomplexobj(E):
        return f"{E.real:.17g}{E.imag:+.17g}j"
    return f"{float(E):.17g}"


def _make_background_rgb(occ, allowed, phi=None, absorber=None, imag_profile=None):
    """
    Base background is a clean white-interior / black-exterior field.

    If `phi` is available (signed distance), we use it to anti-alias the wall:
      occ_smooth = clip(0.5 + phi/(2*w), 0, 1)
    so the black wall edge looks smooth instead of jagged.

    If absorber/imag_profile exist and SHOW_ABSORBER_OVERLAY is True, tint absorber region red.
    """
    if not SHOW_BACKGROUND:
        return None

    if (phi is not None) and BG_EDGE_SOFTEN:
        # Smooth transition across wall over ~1 pixel
        w = 1.0
        occ_smooth = np.clip(0.5 + (phi / (2.0 * w)), 0.0, 1.0).astype(np.float32)
    else:
        # fallback: hard mask
        occ_smooth = occ.astype(np.float32)

    in_rgb = np.array(BG_IN_RGB, dtype=np.float32).reshape(1, 1, 3)
    out_rgb = np.array(BG_OUT_RGB, dtype=np.float32).reshape(1, 1, 3)
    a = np.clip(occ_smooth, 0.0, 1.0)[..., None]  # inside weight
    bg = (1.0 - a) * out_rgb + a * in_rgb

    if SHOW_ABSORBER_OVERLAY and (absorber is not None or imag_profile is not None):
        if imag_profile is not None:
            W = np.array(imag_profile, dtype=np.float64, copy=False)
            W = np.where(np.isfinite(W), W, 0.0)
            wmax = float(W.max()) if W.size else 0.0
            a = (W / wmax) if wmax > 0 else np.zeros_like(W)
        else:
            a = absorber.astype(np.float32)

        a = np.clip(a, 0.0, 1.0) * float(ABSORBER_ALPHA)
        a = a * allowed.astype(np.float32)

        bg[..., 0] = (1 - a) * bg[..., 0] + a * 255.0
        bg[..., 1] = (1 - a) * bg[..., 1] + a * 60.0
        bg[..., 2] = (1 - a) * bg[..., 2] + a * 60.0

    return np.clip(bg, 0, 255).astype(np.uint8)


def main():
    # --- load eigenpair ---
    with h5py.File(H5_PATH, "r") as h5:
        evals = np.array(h5["evals"][:])
        K = int(evals.shape[0])
        if K == 0:
            raise RuntimeError("No eigenvalues found in this database.")

        if USE_SORTED_INDEX:
            key = np.abs(evals) if SORT_KEY == "abs" else np.real(evals)
            order = np.argsort(key)
            if MODE_INDEX < 0 or MODE_INDEX >= K:
                raise IndexError(f"MODE_INDEX={MODE_INDEX} out of range [0, {K-1}]")
            idx = int(order[MODE_INDEX])
            sorted_rank = int(MODE_INDEX)
        else:
            if MODE_INDEX < 0 or MODE_INDEX >= K:
                raise IndexError(f"MODE_INDEX={MODE_INDEX} out of range [0, {K-1}]")
            idx = int(MODE_INDEX)
            sorted_rank = None

        E = evals[idx]
        vec = np.array(h5["evecs"][:, idx], dtype=np.complex128)

    # --- load geometry ---
    Ny, Nx, allowed, occ, absorber, imag_profile, phi = _load_geometry_near_h5(H5_PATH)
    n = int(allowed.sum())
    if vec.shape[0] != n:
        raise ValueError(f"Vector length {vec.shape[0]} does not match allowed DOFs {n}.")

    # --- print energy ---
    tag = f"index={idx}"
    if sorted_rank is not None:
        tag += f" (sorted_rank={sorted_rank}, key={SORT_KEY})"
    print(f"[mode] {tag}")
    print(f"[energy] E = {_format_energy(E)}")

    # --- render ---
    psi = _unpack(vec, allowed, (Ny, Nx))
    bg = _make_background_rgb(occ, allowed, phi=phi, absorber=absorber, imag_profile=imag_profile)

    # colormap
    try:
        cmap_prob = cm.get_cmap(PROB_CMAP)
    except Exception:
        cmap_prob = cm.inferno

    # Choose figure size to get a predictable pixel output.
    if FIGSIZE is not None:
        figsize = FIGSIZE
    else:
        if OUTPUT_PIXELS is None:
            figsize = (Nx / float(FIG_DPI), Ny / float(FIG_DPI))
        else:
            figsize = (float(OUTPUT_PIXELS) / float(FIG_DPI), float(OUTPUT_PIXELS) / float(FIG_DPI))

    fig, ax = plt.subplots(figsize=figsize, dpi=FIG_DPI)

    if RENDER_PROB:
        p = np.abs(psi) ** 2
        p /= (p[allowed].sum() + 1e-20)
        hi = np.percentile(p[allowed], PROB_PCT_HI) if allowed.any() else np.percentile(p, PROB_PCT_HI)
        scale = hi if hi > 1e-20 else p.max() + 1e-20
        p = np.clip(p / scale, 0.0, 1.0)
        if PROB_GAMMA != 1.0:
            p = p ** PROB_GAMMA
        if PROB_STYLE == "mono":
            r, g, b = PROB_MONO_RGB
            img = np.zeros((Ny, Nx, 3), dtype=np.uint8)
            img[..., 0] = np.uint8(np.clip(r, 0, 255))
            img[..., 1] = np.uint8(np.clip(g, 0, 255))
            img[..., 2] = np.uint8(np.clip(b, 0, 255))
        else:
            img = (cmap_prob(p)[:, :, :3] * 255).astype(np.uint8)
    elif RENDER_REAL:
        re = psi.real
        s = np.max(np.abs(re)) + 1e-12
        x = 0.5 * (re / s + 1.0)
        img = (cm.RdBu_r(x)[:, :, :3] * 255).astype(np.uint8)
    else:
        raise ValueError("Enable either RENDER_PROB or RENDER_REAL.")

    if bg is not None:
        if RENDER_PROB:
            # Per-pixel transparency: low density shows NO coloring; high density is colored.
            t0 = float(np.clip(PROB_SHOW_MIN, 0.0, 0.999999))
            a = np.clip((p - t0) / (1.0 - t0), 0.0, 1.0)
            if PROB_ALPHA_GAMMA != 1.0:
                a = a ** float(PROB_ALPHA_GAMMA)
            a = (float(ALPHA_OVERLAY) * a).astype(np.float32)
            blend = ((1.0 - a)[..., None] * bg.astype(np.float32) + a[..., None] * img.astype(np.float32)).astype(np.uint8)
        else:
            blend = (ALPHA_OVERLAY * img + (1 - ALPHA_OVERLAY) * bg).astype(np.uint8)
    else:
        blend = img

    ax.imshow(blend, origin="lower", interpolation=INTERPOLATION)

    # Draw a smooth wall boundary from phi=0 (if available)
    if DRAW_BOUNDARY_CONTOUR and (phi is not None):
        ax.contour(phi, levels=[0.0], colors=[BOUNDARY_COLOR], linewidths=BOUNDARY_LW, origin="lower")
    ax.axis("off")

    if SHOW_TITLE:
        title = f"#{idx}  E={_format_energy(E)}"
        if sorted_rank is not None:
            title = f"(sorted #{sorted_rank}) " + title
        ax.set_title(title, fontsize=10, pad=10)

    if OUT_PNG is None:
        out_dir = os.path.dirname(os.path.abspath(H5_PATH))
        base = os.path.splitext(os.path.basename(H5_PATH))[0]
        out_path = os.path.join(out_dir, f"{base}_mode_{idx:06d}.png")
    else:
        out_path = OUT_PNG

    if SAVE_TIGHT:
        fig.tight_layout()
    if SAVE_PNG:
        fig.savefig(out_path, bbox_inches="tight" if SAVE_TIGHT else None)
        print(f"[ok] wrote: {out_path}")
    else:
        print("[warn] SAVE_PNG is False; nothing was saved.")
    plt.close(fig)


if __name__ == "__main__":
    main()


