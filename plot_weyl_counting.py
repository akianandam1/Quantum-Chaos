#!/usr/bin/env python3
"""
plot_weyl_counting.py

Config-at-the-top script to illustrate spectral completeness via the counting function:

  - Empirical: N(E) = number of eigenvalues <= E
  - Weyl (area term, consistent with lowband_complete_solver.py):
        N_weyl(E) = (A / (4*pi)) * E

It plots N(E) vs N_weyl(E) and also plots the relative error.
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt


# ========= USER CONFIG (edit these) =========
H5_PATH = r"eigensdf/doublewell/final2/eigenpairs.h5"   # eigenpairs.h5 or eigenpairs_complex.h5

# Geometry source for area A:
# If None, will look for geometry_used.npz next to H5_PATH, else fall back to SDF_FALLBACK_PATH.
GEOM_NPZ_PATH = None
SDF_FALLBACK_PATH = r"potentials/final_potential.npz"

# Energy window to show (applied to Re(E) if complex)
E_MIN = 0.0
E_MAX = 3.0    # set None to use max available

# Plotting options
USE_REAL_PART = True   # for complex evals: use Re(E)
STEP_STYLE = True      # True -> step plot for N(E), False -> line

# Output
OUT_DIR = None         # None -> directory of H5_PATH
OUT_PREFIX = "weyl_counting"
SAVE_PNG = True
SAVE_PDF = True
SHOW = True
# ===========================================


def _load_geometry_area(h5_path: str, geom_npz_path: str | None, sdf_fallback_path: str) -> tuple[float, str]:
    """
    Returns (A, source_path) where A is the 'area' used in Weyl estimate.
    Here we follow existing repo convention: A := number of allowed pixels.
    """
    if geom_npz_path is None:
        cand = os.path.join(os.path.dirname(os.path.abspath(h5_path)), "geometry_used.npz")
        geom_npz_path = cand if os.path.exists(cand) else None

    if geom_npz_path is not None and os.path.exists(geom_npz_path):
        g = np.load(geom_npz_path)
        if "allowed" in g.files:
            allowed = g["allowed"].astype(bool)
        else:
            phi = g["phi"].astype(np.float32)
            allowed = phi > 0.0
        return float(allowed.sum()), geom_npz_path

    g = np.load(sdf_fallback_path)
    if "allowed" in g.files:
        allowed = g["allowed"].astype(bool)
    else:
        phi = g["phi"].astype(np.float32)
        allowed = phi > 0.0
    return float(allowed.sum()), sdf_fallback_path


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(H5_PATH)) if OUT_DIR is None else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # --- load evals ---
    with h5py.File(H5_PATH, "r") as h5:
        evals = np.array(h5["evals"][:])
    if evals.size == 0:
        raise RuntimeError("No eigenvalues found in H5_PATH.")

    E = np.real(evals) if (USE_REAL_PART and np.iscomplexobj(evals)) else np.asarray(evals, dtype=np.float64)
    E = E[np.isfinite(E)]
    E = np.sort(E)
    if E.size == 0:
        raise RuntimeError("No finite eigenvalues found.")

    emin = float(E_MIN)
    emax = float(np.max(E)) if (E_MAX is None) else float(E_MAX)
    if emax <= emin:
        raise ValueError("E_MAX must be > E_MIN (or None).")

    # restrict to window
    mask = (E >= emin) & (E <= emax)
    Ew = E[mask]
    if Ew.size == 0:
        raise RuntimeError(f"No eigenvalues in [{emin}, {emax}].")

    N = np.arange(1, Ew.size + 1, dtype=np.float64)

    # --- Weyl area estimate ---
    A, geom_src = _load_geometry_area(H5_PATH, GEOM_NPZ_PATH, SDF_FALLBACK_PATH)
    N_weyl = (A / (4.0 * np.pi)) * Ew

    # relative error (avoid division by tiny values near 0)
    denom = np.maximum(N_weyl, 1e-12)
    rel_err = (N - N_weyl) / denom

    # summary metrics at the top of the window
    top_rel = float(rel_err[-1])
    top_abs = float(N[-1] - N_weyl[-1])

    print(f"[store] {H5_PATH}")
    print(f"[geom] area A={A:.0f} (allowed pixels) from: {geom_src}")
    print(f"[window] E in [{emin}, {emax}] -> {int(N[-1])} states")
    print(f"[end] N(E_max)={N[-1]:.0f}, N_weyl(E_max)={N_weyl[-1]:.2f}, abs diff={top_abs:.2f}, rel err={top_rel*100:.2f}%")

    # --- plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.0), dpi=160, sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})

    # N(E) vs Weyl
    if STEP_STYLE:
        ax1.step(Ew, N, where="post", color="#1f77b4", lw=1.8, label="Computed $N(E)$")
    else:
        ax1.plot(Ew, N, color="#1f77b4", lw=1.8, label="Computed $N(E)$")
    ax1.plot(Ew, N_weyl, color="#111111", lw=2.0, ls="--", label=r"Weyl $A E / (4\pi)$")
    ax1.set_ylabel(r"$N(E)$")
    ax1.grid(True, alpha=0.18)
    ax1.legend(frameon=True)
    ax1.set_title("Counting function vs Weyl estimate")

    # error
    ax2.plot(Ew, 100.0 * rel_err, color="#d62728", lw=1.4)
    ax2.axhline(0.0, color="#111111", lw=0.8, alpha=0.7)
    ax2.set_xlabel(r"Energy $E$" + (" (Re(E))" if (USE_REAL_PART and np.iscomplexobj(evals)) else ""))
    ax2.set_ylabel("Rel. error (%)")
    ax2.grid(True, alpha=0.18)

    fig.tight_layout()

    base = f"{OUT_PREFIX}_E[{emin:g},{emax:g}]"
    if SAVE_PNG:
        out_png = os.path.join(out_dir, base + ".png")
        fig.savefig(out_png, bbox_inches="tight")
        print(f"[save] {out_png}")
    if SAVE_PDF:
        out_pdf = os.path.join(out_dir, base + ".pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"[save] {out_pdf}")
    if SHOW:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()











