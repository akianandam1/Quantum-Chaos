#!/usr/bin/env python3
"""
deviation_histogram.py

Publication-style histogram / bar chart:
  - x-axis: percent above ergodic baseline (e.g. 10, 20, ... 60)
  - y-axis: percent of states with that deviation

Edit the CONFIG section at the top, then run:
  python deviation_histogram.py
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ========= USER CONFIG (edit these) =========
TITLE = "Eigenstate Deviation Above Classical Baseline"  # e.g. "Deviation above ergodic baseline"
X_LABEL = "Percent above classical baseline"
Y_LABEL = "Percent of states"

# Bin centers (or bin labels) for the x-axis, in percent.
# Example: 10, 20, ..., 60 means “~10% above baseline”, etc.
X_BINS_PERCENT = [10, 20, 30, 40, 50, 60]

# Provide EITHER:
#   (A) Y_PERCENT (already in percent, same length as X_BINS_PERCENT), OR
#   (B) COUNTS (raw counts in each bin) + TOTAL_STATES (to convert to percent)
Y_PERCENT = None
# Y_PERCENT = [12.5, 18.0, 22.0, 20.5, 17.0, 10.0]

#Original 
COUNTS = [30542, 17339, 9937, 6027, 3933, 2548]

#COUNTS = [9076, 7801, 6235, 4625, 3235, 2550]

#Original 
TOTAL_STATES = 50893  # e.g. 1000

#TOTAL_STATES = 10240  # e.g. 1000

# Appearance / output
COLOR = "#2C7FB8"        # blue
EDGE_COLOR = "#1b1b1b"
BAR_ALPHA = 0.92
FIGSIZE = (7.0, 4.2)
DPI = 300
OUTPUT_PATH = "eigensdf/doublewell/final2/deviation_histogram.png"  # also consider .pdf for publication

SHOW_VALUE_LABELS = True
VALUE_LABEL_FMT = "{:.1f}%"   # shown above each bar
VALUE_LABEL_FONTSIZE = 9

Y_LIM = None  # e.g. (0, 30)
GRID_ALPHA = 0.18
# ===========================================


def _to_percent(x_bins_percent, y_percent, counts, total_states):
    x = np.asarray(x_bins_percent, dtype=float)
    if y_percent is not None:
        y = np.asarray(y_percent, dtype=float)
        if y.shape != x.shape:
            raise ValueError("Y_PERCENT must have the same length as X_BINS_PERCENT.")
        return x, y

    c = np.asarray(counts, dtype=float)
    if c.shape != x.shape:
        raise ValueError("COUNTS must have the same length as X_BINS_PERCENT.")
    if total_states is None:
        raise ValueError("If Y_PERCENT is None, you must set TOTAL_STATES.")
    total_states = float(total_states)
    if total_states <= 0:
        raise ValueError("TOTAL_STATES must be > 0.")
    y = 100.0 * c / total_states
    return x, y


def main() -> None:
    # Nice, clean default styling without external deps
    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
        }
    )

    x, y = _to_percent(X_BINS_PERCENT, Y_PERCENT, COUNTS, TOTAL_STATES)

    if np.any(~np.isfinite(y)) or np.any(y < 0):
        raise ValueError("Y values must be finite and non-negative.")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Bar width: infer from spacing if possible
    if x.size >= 2:
        dx = float(np.median(np.diff(np.sort(x))))
        width = 0.75 * dx
    else:
        width = 0.8

    bars = ax.bar(
        x,
        y,
        width=width,
        color=COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
        alpha=BAR_ALPHA,
    )

    # Labels / title
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    if TITLE:
        ax.set_title(TITLE, pad=10)

    # Ticks: show exact bin centers
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}%" if float(v).is_integer() else f"{v:g}%" for v in x])

    # Grid + spines
    ax.grid(True, axis="y", alpha=GRID_ALPHA, linewidth=0.8)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if Y_LIM is not None:
        ax.set_ylim(*Y_LIM)

    # Value labels
    if SHOW_VALUE_LABELS:
        y_max = float(np.max(y)) if y.size else 0.0
        pad = 0.01 * max(y_max, 1.0)
        for rect, yi in zip(bars, y):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + pad,
                VALUE_LABEL_FMT.format(float(yi)),
                ha="center",
                va="bottom",
                fontsize=VALUE_LABEL_FONTSIZE,
                color="#222222",
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"[ok] wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()












