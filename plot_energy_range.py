#!/usr/bin/env python3
"""
plot_energy_range.py

Config-at-the-top script to plot eigenvalues in the complex plane for a specified range.

Works with:
  - lowband_complete_solver.py outputs: eigenpairs.h5 (evals are real)
  - complex_absorbing_solver.py outputs: eigenpairs_complex.h5 (evals are complex)

By default, "energy" means Re(E) for complex eigenvalues.
"""

import os

import h5py
import numpy as np
import matplotlib.pyplot as plt


# ========= USER CONFIG (edit these) =========
# Provide one or two datasets. If H5_PATH_2 is None, only the first is plotted.
H5_PATH_1 = r"eigensdf/singleresonances/trial1/eigenpairs_complex.h5"
H5_PATH_2 = r"eigensdf/singleresonances/trial2/eigenpairs_complex.h5"  # e.g. r"eigensdf/singleresonances/trial2/eigenpairs_complex.h5"
LABEL_1 = "dataset 1"
LABEL_2 = "dataset 2"
E_MIN = 0.047
E_MAX = 0.049

# Plot options
SORT_BY_ENERGY = True   # sort by Re(E) (only affects draw order)
LOG_X = False           # log scale on x-axis (Re(E)) (use with care near 0)
LOG_Y = False           # log scale on y-axis (|Im(E)|); if enabled, uses |Im(E)|
POINT_SIZE = 18         # marker size
ALPHA_1 = 0.85
ALPHA_2 = 0.85
COLOR_1 = "tab:blue"
COLOR_2 = "tab:orange"

# Labels next to each point
ANNOTATE_POINTS = True
ANNOTATE_FONT_SIZE = 7
ANNOTATE_OFFSET = (3, 2)     # pixels (x,y)
ANNOTATE_MAX = None           # safety cap; set to None to annotate everything

# If None, auto-generate next to H5_PATH
OUT_PNG = None
# ===========================================


def load_evals(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if "evals" not in f:
            raise KeyError(f"Dataset 'evals' not found in {h5_path}")
        return np.array(f["evals"][:])


def select_complex_plane_points(evals: np.ndarray, emin: float, emax: float):
    E_re = np.real(evals).astype(np.float64, copy=False)
    E_im = np.imag(evals).astype(np.float64, copy=False)
    sel = (E_re >= emin) & (E_re <= emax) & np.isfinite(E_re) & np.isfinite(E_im)
    idx = np.where(sel)[0].astype(np.int64, copy=False)
    E_re_sel = E_re[idx]
    E_im_sel = E_im[idx]
    if SORT_BY_ENERGY and E_re_sel.size > 1:
        order = np.argsort(E_re_sel)
        idx = idx[order]
        E_re_sel = E_re_sel[order]
        E_im_sel = E_im_sel[order]
    return idx, E_re_sel, E_im_sel


def main() -> None:
    emin = float(E_MIN)
    emax = float(E_MAX)
    if emax <= emin:
        raise ValueError("E_MAX must be > E_MIN")

    if not H5_PATH_1:
        raise ValueError("H5_PATH_1 must be set.")

    evals1 = load_evals(H5_PATH_1)
    idx1, x1, y1 = select_complex_plane_points(evals1, emin, emax)
    if x1.size == 0:
        raise RuntimeError(f"[{LABEL_1}] No eigenvalues found with Re(E) in [{emin}, {emax}].")

    has_second = H5_PATH_2 is not None and str(H5_PATH_2).strip() != ""
    if has_second:
        evals2 = load_evals(H5_PATH_2)
        idx2, x2, y2 = select_complex_plane_points(evals2, emin, emax)
        if x2.size == 0:
            raise RuntimeError(f"[{LABEL_2}] No eigenvalues found with Re(E) in [{emin}, {emax}].")
    else:
        evals2 = None
        idx2 = np.array([], dtype=np.int64)
        x2, y2 = np.array([], dtype=float), np.array([], dtype=float)

    if OUT_PNG is None:
        base1 = os.path.splitext(os.path.basename(H5_PATH_1))[0]
        out_dir = os.path.dirname(os.path.abspath(H5_PATH_1))
        if has_second:
            base2 = os.path.splitext(os.path.basename(str(H5_PATH_2)))[0]
            out_name = f"{base1}_vs_{base2}_complex_plane_Re[{emin:g},{emax:g}]"
        else:
            out_name = f"{base1}_complex_plane_Re[{emin:g},{emax:g}]"
        if LOG_X:
            out_name += "_logx"
        if LOG_Y:
            out_name += "_logyAbsIm"
        out_path = os.path.join(out_dir, out_name + ".png")
    else:
        out_path = OUT_PNG

    plt.figure(figsize=(6, 5), dpi=140)

    y1_plot = np.abs(y1) if LOG_Y else y1
    plt.scatter(x1, y1_plot, s=POINT_SIZE, alpha=ALPHA_1, color=COLOR_1, label=f"{LABEL_1} (n={x1.size})")

    if has_second:
        y2_plot = np.abs(y2) if LOG_Y else y2
        plt.scatter(x2, y2_plot, s=POINT_SIZE, alpha=ALPHA_2, color=COLOR_2, label=f"{LABEL_2} (n={x2.size})")

    if ANNOTATE_POINTS:
        max1 = x1.size if ANNOTATE_MAX is None else min(int(ANNOTATE_MAX), x1.size)
        for i in range(max1):
            plt.annotate(
                str(int(idx1[i])),
                (x1[i], y1_plot[i]),
                textcoords="offset points",
                xytext=ANNOTATE_OFFSET,
                fontsize=ANNOTATE_FONT_SIZE,
                color=COLOR_1,
            )
        if has_second:
            max2 = x2.size if ANNOTATE_MAX is None else min(int(ANNOTATE_MAX), x2.size)
            for i in range(max2):
                plt.annotate(
                    str(int(idx2[i])),
                    (x2[i], y2_plot[i]),
                    textcoords="offset points",
                    xytext=ANNOTATE_OFFSET,
                    fontsize=ANNOTATE_FONT_SIZE,
                    color=COLOR_2,
                )

    plt.axhline(0.0, color="k", lw=0.6, alpha=0.6)
    if LOG_X:
        plt.xscale("log")
    if LOG_Y:
        plt.yscale("log")

    plt.xlabel("Re(E)")
    plt.ylabel("|Im(E)|" if LOG_Y else "Im(E)")
    plt.title(f"Complex spectrum slice: Re(E) in [{emin:g}, {emax:g}]")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"[ok] loaded {evals1.size} eigenvalues from: {H5_PATH_1}")
    print(f"[ok] selected {x1.size} with Re(E) in [{emin}, {emax}] for {LABEL_1}")
    if has_second and evals2 is not None:
        print(f"[ok] loaded {evals2.size} eigenvalues from: {H5_PATH_2}")
        print(f"[ok] selected {x2.size} with Re(E) in [{emin}, {emax}] for {LABEL_2}")
    print(f"[ok] wrote plot: {out_path}")


if __name__ == "__main__":
    main()


