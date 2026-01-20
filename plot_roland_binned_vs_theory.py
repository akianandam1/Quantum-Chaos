#!/usr/bin/env python3
"""
plot_roland_binned_vs_theory.py

Load the NPZ produced by roland_graph_datapoints.py and create a publication-style plot of:
  - binned (mean) datapoints in increasing energy order
  - theory curve: x/(1+x)
  - variance shading (±1 std) per bin

The NPZ is expected to contain per-eigenstate arrays:
  - E, w12, x, indices
and optionally window_stats.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# ========= USER CONFIG (edit these) =========
DATA_NPZ_PATH = r"eigensdf/doublewell/final2/roland/partial_barrier_data.npz"

# Binning in increasing energy order
BATCH_SIZE = 100
START_RANK = 0          # 0-based index in the energy-sorted list
END_RANK = None         # None -> use all available

# Shading / error definition
SHADE_MODE = "std"      # "std" or "sem" (standard error of mean)
SHADE_SIGMA = 1.0       # 1.0 -> ±1σ, 2.0 -> ±2σ, etc.

# Plotting
OUT_DIR = None          # None -> same directory as DATA_NPZ_PATH
OUT_NAME = "roland_binned_vs_theory.png"
SAVE_PDF = True
SHOW = True
Y_MAX = 1.5             # y-axis upper limit

# Appearance
POINT_COLOR = "#2C7FB8"
BAND_COLOR = "#2C7FB8"
THEORY_COLOR = "#111111"
# ===========================================


def main() -> None:
    mpl.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.linewidth": 1.1,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    d = np.load(DATA_NPZ_PATH)
    if not all(k in d.files for k in ["E", "w12", "x"]):
        raise KeyError(f"{DATA_NPZ_PATH} must contain arrays: E, w12, x")

    E = np.asarray(d["E"], dtype=float)
    w12 = np.asarray(d["w12"], dtype=float)
    x = np.asarray(d["x"], dtype=float)

    if not (E.shape == w12.shape == x.shape):
        raise ValueError("E, w12, x must have the same shape.")

    # sort by increasing energy
    good = np.isfinite(E) & np.isfinite(w12) & np.isfinite(x)
    E = E[good]
    w12 = w12[good]
    x = x[good]
    order = np.argsort(E)
    E = E[order]
    w12 = w12[order]
    x = x[order]

    start = int(START_RANK)
    end = int(E.size) if END_RANK is None else int(END_RANK)
    start = max(0, min(start, E.size))
    end = max(start, min(end, E.size))

    E = E[start:end]
    w12 = w12[start:end]
    x = x[start:end]

    if E.size < BATCH_SIZE:
        raise ValueError(f"Not enough points ({E.size}) for BATCH_SIZE={BATCH_SIZE}.")

    # bin
    n_bins = E.size // int(BATCH_SIZE)
    E = E[: n_bins * int(BATCH_SIZE)]
    w12 = w12[: n_bins * int(BATCH_SIZE)]
    x = x[: n_bins * int(BATCH_SIZE)]

    E_bin = E.reshape(n_bins, int(BATCH_SIZE))
    w12_bin = w12.reshape(n_bins, int(BATCH_SIZE))
    x_bin = x.reshape(n_bins, int(BATCH_SIZE))

    E_mean = E_bin.mean(axis=1)
    x_mean = x_bin.mean(axis=1)
    w_mean = w12_bin.mean(axis=1)

    w_std = w12_bin.std(axis=1, ddof=1) if BATCH_SIZE > 1 else np.zeros(n_bins)
    if SHADE_MODE == "sem":
        w_err = w_std / np.sqrt(float(BATCH_SIZE))
    else:
        w_err = w_std
    w_err = float(SHADE_SIGMA) * w_err

    # theory curve (span x-range)
    x_lo = float(np.min(x_mean))
    x_hi = float(np.max(x_mean))
    xs = np.linspace(max(0.0, x_lo * 0.95), x_hi * 1.05, 600)
    theory = xs / (1.0 + xs)

    # output paths
    out_dir = os.path.dirname(os.path.abspath(DATA_NPZ_PATH)) if OUT_DIR is None else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, OUT_NAME)
    out_pdf = os.path.splitext(out_png)[0] + ".pdf"

    # plot
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)

    ax.plot(xs, theory, color=THEORY_COLOR, lw=2.2, label=r"Theory: $x/(1+x)$", zorder=1)

    # Unconnected points with +/- variance ticks
    err_label = (r"$\pm$1 SEM" if SHADE_MODE == "sem" else r"$\pm$1 std") if float(SHADE_SIGMA) == 1.0 else (r"$\pm$" + f"{float(SHADE_SIGMA):g} " + ("SEM" if SHADE_MODE == "sem" else "std"))
    ax.errorbar(
        x_mean,
        w_mean,
        yerr=w_err,
        fmt="o",
        linestyle="none",
        ms=4.8,
        mfc=POINT_COLOR,
        mec="white",
        mew=0.5,
        ecolor=POINT_COLOR,
        elinewidth=1.0,
        capsize=3.0,
        capthick=1.0,
        alpha=0.95,
        label=f"Binned mean (batch={BATCH_SIZE}), {err_label}",
        zorder=3,
    )

    ax.set_xlabel(r"$x(E)$")
    ax.set_ylabel(r"$\langle w_{12}\rangle$")
    ax.set_title("Binned Roland datapoints vs theory")
    ax.grid(True, alpha=0.18)
    ax.set_ylim(0.0, float(Y_MAX))

    # annotate info
    ax.text(
        0.01,
        0.02,
        f"Sorted by energy; bins={n_bins}, points={n_bins*BATCH_SIZE}\n"
        f"Energy range: [{E_mean.min():.3g}, {E_mean.max():.3g}]",
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        va="bottom",
    )

    ax.legend(frameon=True, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_png)
    print(f"[save] {out_png}")
    if SAVE_PDF:
        fig.savefig(out_pdf)
        print(f"[save] {out_pdf}")
    if SHOW:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()


