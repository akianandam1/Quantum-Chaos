#!/usr/bin/env python3
"""
plot_roland_binned_vs_theory.py

Load the NPZ produced by roland_graph_datapoints.py and create a publication-style plot of:
  - binned (mean) datapoints in increasing energy order
  - theory curve: x/(1+x)
  - uncertainty visualization per bin:
      * std/sem: error bars around the mean
      * quartiles: min–max band plus 25–75% (IQR) band, with mean points

The NPZ is expected to contain per-eigenstate arrays:
  - E, w12, x, indices
and optionally window_stats.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# ========= USER CONFIG (edit these) =========
DATA_NPZ_PATH = r"eigensdf/doublewell/channel2/roland/partial_barrier_data.npz"

# Binning in increasing energy order
BATCH_SIZE =25
START_RANK =25          # 0-based index in the energy-sorted list
END_RANK = None         # None -> use all available

# Shading / error definition
SHADE_MODE = "quartiles"      # "std", "sem" (standard error of mean), or "quartiles"
SHADE_SIGMA = 1       # 1.0 -> ±1σ, 2.0 -> ±2σ, etc.
QTL_OUTER_ALPHA = 0.18  # used only for SHADE_MODE="quartiles" (min–max band)
QTL_INNER_ALPHA = 0.35  # used only for SHADE_MODE="quartiles" (25–75% band)
QTL_SHOW_EXTREMES = False  # if False, show only the 25–75% band (hide min–max)

# Plotting
OUT_DIR = None          # None -> same directory as DATA_NPZ_PATH
OUT_NAME = "roland_binned_vs_theory.png"
SAVE_PDF = True
SHOW = True
Y_MAX = 1.2            # y-axis upper limit (used only if Y_LIM is None)
X_LOG = False           # if True, use log scale on x-axis (requires x>0)
X_LIM = (0,30)            # None or (xmin, xmax). If X_LOG=True, xmin must be > 0.
Y_LIM = (0,1.2)            # None or (ymin, ymax). If None, defaults to (0, Y_MAX).
X_FLOOR_TO_INT = True    # if True, group by floor(x) so there is one point per integer n_E (see X_FLOOR_GROUP_AVG)
X_FLOOR_GROUP_AVG = True  # if True and X_FLOOR_TO_INT=True, average over all batches with the same floor(x)
SHOW_VARIANCE = False      # if False, hide uncertainty visualization (error bars / shaded bands) and plot means only
PRINT_MAX_W12 = True      # print the max w12 in the data used for this plot
NORMALIZE_BY_MAX_W12 = False  # divide all w12 by that max so max becomes 1
X_SCALE = 1             # multiply all x values by this factor before binning/plotting (e.g. 0.5, 2.0)

# Appearance
POINT_COLOR = "#2C7FB8"
BAND_COLOR = "#F54927"#"#2C7FB8"
THEORY_COLOR = "#111111"
# ===========================================

def _group_bins_by_floor_x(
    x_mean: np.ndarray,
    w_mean: np.ndarray,
    shade_mode: str,
    *,
    shade_sigma: float,
    qtl_show_extremes: bool,
):
    """
    Collapse binned points by integer floor(x). Each *bin* counts as one sample.

    Returns a dict with keys:
      x, w_mean,
      w_err (for std/sem),
      w_min/w_max/w_q25/w_q75 (for quartiles)
    """
    if x_mean.size == 0:
        return {"x": x_mean, "w_mean": w_mean}

    x_floor = np.floor(x_mean).astype(int)
    # Sort by floor(x) for stable output
    order = np.argsort(x_floor, kind="stable")
    x_floor = x_floor[order]
    w_mean = w_mean[order]

    uniq, start_idx, counts = np.unique(x_floor, return_index=True, return_counts=True)
    x_out = uniq.astype(float)

    w_out = np.empty_like(x_out, dtype=float)

    if shade_mode in ("std", "sem"):
        w_err = np.zeros_like(x_out, dtype=float)
    elif shade_mode == "quartiles":
        w_min = np.empty_like(x_out, dtype=float)
        w_max = np.empty_like(x_out, dtype=float)
        w_q25 = np.empty_like(x_out, dtype=float)
        w_q75 = np.empty_like(x_out, dtype=float)
    else:
        raise ValueError(f"Unknown SHADE_MODE={shade_mode!r}. Use 'std', 'sem', or 'quartiles'.")

    for i, (s, c) in enumerate(zip(start_idx, counts)):
        vals = w_mean[s : s + c]
        w_out[i] = float(np.mean(vals))
        if shade_mode in ("std", "sem"):
            if c <= 1:
                std = 0.0
            else:
                std = float(np.std(vals, ddof=1))
            err = std / np.sqrt(float(c)) if shade_mode == "sem" else std
            w_err[i] = float(shade_sigma) * err
        else:  # quartiles
            w_q25[i], w_q75[i] = np.percentile(vals, [25, 75])
            if bool(qtl_show_extremes):
                w_min[i] = float(np.min(vals))
                w_max[i] = float(np.max(vals))

    out = {"x": x_out, "w_mean": w_out}
    if shade_mode in ("std", "sem"):
        out["w_err"] = w_err
    else:
        out["w_q25"] = w_q25
        out["w_q75"] = w_q75
        if bool(qtl_show_extremes):
            out["w_min"] = w_min
            out["w_max"] = w_max
    return out


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

    # Optional: scale x-values (applies everywhere downstream: binning, theory range, plot)
    if float(X_SCALE) != 1.0:
        x = x * float(X_SCALE)

    # Optional: report/normalize against the maximum w12 in the plotted dataset slice
    w12_max = float(np.max(w12)) if w12.size else float("nan")
    if PRINT_MAX_W12:
        print(f"[w12] max over selected data (after START_RANK/END_RANK filters): {w12_max:.17g}")
    if NORMALIZE_BY_MAX_W12:
        if not np.isfinite(w12_max) or w12_max <= 0.0:
            raise ValueError(f"Cannot normalize by max w12={w12_max!r} (must be finite and > 0).")
        w12 = w12 / w12_max

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

    # quartiles (if requested)
    if SHADE_MODE == "quartiles":
        w_min = np.min(w12_bin, axis=1)
        w_max = np.max(w12_bin, axis=1)
        w_q25, w_q75 = np.percentile(w12_bin, [25, 75], axis=1)

    # Optional: group by floor(x) so there is one point per integer n_E.
    if X_FLOOR_TO_INT:
        if bool(X_FLOOR_GROUP_AVG):
            grouped = _group_bins_by_floor_x(
                x_mean,
                w_mean,
                str(SHADE_MODE).strip().lower(),
                shade_sigma=float(SHADE_SIGMA),
                qtl_show_extremes=bool(QTL_SHOW_EXTREMES),
            )
            x_mean = grouped["x"]
            w_mean = grouped["w_mean"]
            # Keep E_mean only for potential annotation; approximate as bin index order after grouping.
            # (Not used for plotting besides optional annotation.)
            E_mean = np.arange(x_mean.size, dtype=float)
            if SHADE_MODE in ("std", "sem"):
                w_err = grouped["w_err"]
            elif SHADE_MODE == "quartiles":
                w_q25 = grouped["w_q25"]
                w_q75 = grouped["w_q75"]
                if bool(QTL_SHOW_EXTREMES):
                    w_min = grouped["w_min"]
                    w_max = grouped["w_max"]
        else:
            # Legacy behavior: just floor x-values but keep all bins.
            x_mean = np.floor(x_mean).astype(float)

    # theory curve (span x-range)
    x_lo = float(np.min(x_mean))
    x_hi = float(np.max(x_mean))
    if X_LOG:
        # log scale can't include x<=0; use the smallest positive binned x as lower bound
        pos = x_mean[np.isfinite(x_mean) & (x_mean > 0)]
        x_min_pos = float(pos.min()) if pos.size else 1e-6
        lo = max(x_min_pos * 0.95, 1e-12)
        hi = max(x_hi * 1.05, lo * 1.01)
        xs = np.logspace(np.log10(lo), np.log10(hi), 600)
    else:
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

    # If log x, drop any nonpositive binned points to avoid matplotlib warnings/errors.
    if X_LOG:
        keep = np.isfinite(x_mean) & (x_mean > 0) & np.isfinite(w_mean)
        x_mean = x_mean[keep]
        w_mean = w_mean[keep]
        w_err = w_err[keep]
        E_mean = E_mean[keep]
        if SHADE_MODE == "quartiles":
            w_min = w_min[keep]
            w_max = w_max[keep]
            w_q25 = w_q25[keep]
            w_q75 = w_q75[keep]

    if SHADE_MODE in ("std", "sem"):
        if bool(SHOW_VARIANCE):
            # Unconnected points with +/- variance ticks
            err_label = (
                (r"$\pm$1 SEM" if SHADE_MODE == "sem" else r"$\pm$1 std")
                if float(SHADE_SIGMA) == 1.0
                else (r"$\pm$" + f"{float(SHADE_SIGMA):g} " + ("SEM" if SHADE_MODE == "sem" else "std"))
            )
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
                ecolor=BAND_COLOR,
                elinewidth=1.0,
                capsize=3.0,
                capthick=1.0,
                alpha=0.95,
                label=f"Mean (batch={BATCH_SIZE}), {err_label}",
                zorder=3,
            )
        else:
            ax.plot(
                x_mean,
                w_mean,
                "o",
                ms=4.8,
                mfc=POINT_COLOR,
                mec="white",
                mew=0.5,
                alpha=0.95,
                label=f"Mean (batch={BATCH_SIZE})",
                zorder=3,
            )
    elif SHADE_MODE == "quartiles":
        # For fill_between we sort by x to avoid self-crossing polygons if x_mean isn't monotone.
        sx = np.argsort(x_mean)
        xs_mean = x_mean[sx]
        ws_mean = w_mean[sx]
        ws_q25 = w_q25[sx]
        ws_q75 = w_q75[sx]

        if bool(SHOW_VARIANCE):
            if bool(QTL_SHOW_EXTREMES):
                ws_min = w_min[sx]
                ws_max = w_max[sx]
                ax.fill_between(
                    xs_mean,
                    ws_min,
                    ws_max,
                    color=BAND_COLOR,
                    alpha=float(QTL_OUTER_ALPHA),
                    linewidth=0.0,
                    label="Min–Max (per bin)",
                    zorder=2,
                )
            ax.fill_between(
                xs_mean,
                ws_q25,
                ws_q75,
                color=BAND_COLOR,
                alpha=float(QTL_INNER_ALPHA),
                linewidth=0.0,
                label="25–75% (IQR)",
                zorder=2.2,
            )
        ax.plot(
            xs_mean,
            ws_mean,
            "o",
            ms=4.8,
            mfc=POINT_COLOR,
            mec="white",
            mew=0.5,
            alpha=0.95,
            label=f"Mean (batch={BATCH_SIZE})",
            zorder=3,
        )
    else:
        raise ValueError(f"Unknown SHADE_MODE={SHADE_MODE!r}. Use 'std', 'sem', or 'quartiles'.")

    ax.set_xlabel(r"$n_E$")
    ax.set_ylabel(r"$\langle w_{12}\rangle$")
    ax.set_title("Values Computer from Eigenstates Vs. Theoretical Model")
    ax.grid(True, alpha=0.18)
    if X_LOG:
        ax.set_xscale("log")

    # Axis ranges (optional)
    if X_LIM is not None:
        ax.set_xlim(float(X_LIM[0]), float(X_LIM[1]))
    if Y_LIM is not None:
        ax.set_ylim(float(Y_LIM[0]), float(Y_LIM[1]))
    else:
        ax.set_ylim(0.0, float(Y_MAX))

    # annotate info
    # ax.text(
    #     0.01,
    #     0.02,
    #     # f"Sorted by energy; bins={n_bins}, points={n_bins*BATCH_SIZE}\n"
    #     f"Energy range: [{E_mean.min():.3g}, {E_mean.max():.3g}]",
    #     transform=ax.transAxes,
    #     fontsize=9,
    #     color="#333333",
    #     va="bottom",
    # )

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


