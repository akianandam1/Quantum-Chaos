#!/usr/bin/env python3
"""
extreme_state_census.py

Scan an existing HDF5 file of eigenstates and find "extreme" states in the sense
you described:

- extremely localized in one well (most weight in small or large)
- extremely channel-heavy (surprisingly big weight in the channel despite it being narrow)
- optionally: extreme vs uniform baseline

This is meant to mimic the "a few states couple very strongly, many stay trapped"
logic you see in resonance width bifurcation, but for your closed double-well+channel
geometry.

You only need to edit CONFIG below.
"""

import os
import json
import csv
import numpy as np
import h5py
from scipy import ndimage as ndi


CONFIG = {
    # ----------------- Input files -----------------
    "NPZ_PATH": r"potentials/potential7.npz",
    "H5_PATH": r"eigensdf/doublewell/trial1/eigenpairs.h5",

    # ----------------- Region detection -----------------
    # "manual_label_png" = use your manually painted PNG with 1/2/3 labels
    # "component_split"  = try to auto-split the allowed region into small/channel/large
    "REGION_MODE": "manual_label_png",
    "LABEL_PNG": r"path\\to\\labels.png",  # if manual mode: 1=small, 2=channel, 3=large

    # if component_split:
    "CHANNEL_RADIUS": 9.0,

    # ----------------- Extremeness criteria -----------------
    # 1) localized in a well if prob in that well >= this
    "WELL_LOCAL_FRAC": 0.85,      # e.g. 0.85 = 85% of probability in small or large

    # 2) channel-heavy if channel prob >= this
    "CHANNEL_HEAVY_FRAC": 0.15,   # adjust depending on how tiny your channel is

    # 3) optional: extreme relative to uniform baseline
    #    state is "uniform-extreme" if ratio_vs_uniform_* >= this
    #    (this needs region areas to compute uniform baseline)
    "USE_UNIFORM_EXTREME": True,
    "UNIFORM_EXTREME_RATIO": 2.0,  # e.g. 2.0 = 200% of uniform density

    # ----------------- Run knobs -----------------
    "OUT_DIR": r"analysis_out",
    "STATE_BATCH": 256,       # how many eigenstates (columns) per pass
    "ROW_CHUNK": 65536,       # how many rows of HDF5 to stream at a time
    "PROGRESS_EVERY": 500,    # print every N states (0 = off)
    "PREVIEW_MASKS": False,   # set True if you want to save region_masks.png
}


# =============== helper: load npz ===============
def load_npz(npz_path):
    g = np.load(npz_path)
    phi = g["phi"].astype(np.float32)
    allowed = g["allowed"].astype(bool) if "allowed" in g else (phi > 0)
    return phi, allowed


# =============== helper: manual label png ===============
def partition_manual_from_png(label_png, phi_shape, preview_path=None):
    # we support either indexed labels or RGB (red=1, green=2, blue=3)
    try:
        from PIL import Image
        arr = np.array(Image.open(label_png))
    except Exception:
        import matplotlib.image as mpimg
        arr = mpimg.imread(label_png)

    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3]
        if rgb.dtype != np.uint8:
            rgb = (255 * rgb).astype(np.uint8)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        lbl = np.zeros(r.shape, dtype=np.uint8)
        # red -> small
        lbl[(r > 200) & (g < 50) & (b < 50)] = 1
        # green -> channel
        lbl[(g > 200) & (r < 50) & (b < 50)] = 2
        # blue -> large
        lbl[(b > 200) & (r < 50) & (g < 50)] = 3
    else:
        lbl = arr.astype(np.uint8)

    if lbl.shape != phi_shape:
        raise ValueError(f"Label PNG shape {lbl.shape} does not match phi shape {phi_shape}.")

    small_mask = (lbl == 1)
    channel_mask = (lbl == 2)
    large_mask = (lbl == 3)

    if preview_path:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), dpi=130)
        ax[0].imshow(small_mask, origin="lower", cmap="gray"); ax[0].set_title("small"); ax[0].axis("off")
        ax[1].imshow(channel_mask, origin="lower", cmap="gray"); ax[1].set_title("channel"); ax[1].axis("off")
        ax[2].imshow(large_mask, origin="lower", cmap="gray"); ax[2].set_title("large"); ax[2].axis("off")
        fig.tight_layout(); fig.savefig(preview_path, bbox_inches="tight"); plt.close(fig)

    return small_mask, channel_mask, large_mask


# =============== helper: component split ===============
def partition_component_split(phi, allowed, channel_radius=9.0, preview_path=None):
    # this is the same logic you've been using: carve a band, find two bodies, grow, then channel = band touching both
    d = np.where(allowed, np.maximum(phi, 0.0), 0.0)
    band = allowed & (d <= float(channel_radius))
    bodies = allowed & ~band
    lab, nlab = ndi.label(bodies)
    if nlab < 2:
        # try to loosen radius a bit
        for rr in [channel_radius * 1.2, channel_radius * 1.5, channel_radius * 1.8]:
            band = allowed & (d <= float(rr))
            bodies = allowed & ~band
            lab, nlab = ndi.label(bodies)
            if nlab >= 2:
                break
    if nlab < 2:
        raise RuntimeError("Could not split allowed region into two bodies; adjust CHANNEL_RADIUS")

    sizes = ndi.sum(np.ones_like(lab), labels=lab, index=np.arange(1, nlab + 1))
    order = np.argsort(sizes)[::-1]
    seedA = (lab == int(order[0] + 1))
    seedB = (lab == int(order[1] + 1))
    growth_mask = allowed & ~band
    growA = ndi.binary_propagation(seedA, mask=growth_mask)
    growB = ndi.binary_propagation(seedB, mask=growth_mask)
    areaA, areaB = int(growA.sum()), int(growB.sum())
    if areaA >= areaB:
        large_mask, small_mask = growA, growB
    else:
        large_mask, small_mask = growB, growA

    # find channel bits that touch both
    st = ndi.generate_binary_structure(2, 1)
    smd = ndi.binary_dilation(small_mask, structure=st, iterations=1)
    lgd = ndi.binary_dilation(large_mask, structure=st, iterations=1)
    labc, nlabc = ndi.label(band, structure=st)
    keep = np.zeros_like(band, dtype=bool)
    for k in range(1, nlabc + 1):
        comp = (labc == k)
        if (comp & smd).any() and (comp & lgd).any():
            keep |= comp
    channel_mask = keep

    if preview_path:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), dpi=130)
        ax[0].imshow(small_mask, origin="lower", cmap="gray"); ax[0].set_title("small"); ax[0].axis("off")
        ax[1].imshow(channel_mask, origin="lower", cmap="gray"); ax[1].set_title("channel"); ax[1].axis("off")
        ax[2].imshow(large_mask, origin="lower", cmap="gray"); ax[2].set_title("large"); ax[2].axis("off")
        fig.tight_layout(); fig.savefig(preview_path, bbox_inches="tight"); plt.close(fig)

    return small_mask, channel_mask, large_mask


# =============== main scan ===============
def main():
    cfg = CONFIG
    os.makedirs(cfg["OUT_DIR"], exist_ok=True)

    phi, allowed = load_npz(cfg["NPZ_PATH"])
    preview_path = os.path.join(cfg["OUT_DIR"], "extreme_region_masks.png") if cfg["PREVIEW_MASKS"] else None

    # get region masks
    if cfg["REGION_MODE"] == "manual_label_png":
        small_mask, channel_mask, large_mask = partition_manual_from_png(
            cfg["LABEL_PNG"], phi.shape, preview_path
        )
    else:
        small_mask, channel_mask, large_mask = partition_component_split(
            phi, allowed, cfg["CHANNEL_RADIUS"], preview_path
        )

    # areas
    A_small = int(small_mask.sum())
    A_chan = int(channel_mask.sum())
    A_large = int(large_mask.sum())
    A_tot = int(allowed.sum())
    baseline = 1.0 / float(A_tot) if A_tot > 0 else 0.0

    print(f"[areas] small={A_small}, channel={A_chan}, large={A_large}, total_allowed={A_tot}")
    print(f"[uniform baseline] 1/Area_total = {baseline:.6e}")

    # flatten masks to match eigenvector ordering: h5 has evecs of shape (N_allowed, K)
    # we assume evecs are already on allowed points, so we need to map mask -> indices
    idx_map = -np.ones_like(allowed, dtype=np.int64)
    idx_map[allowed] = np.arange(A_tot, dtype=np.int64)

    idx_small = idx_map[small_mask]
    idx_small = idx_small[idx_small >= 0]
    idx_chan = idx_map[channel_mask]
    idx_chan = idx_chan[idx_chan >= 0]
    idx_large = idx_map[large_mask]
    idx_large = idx_large[idx_large >= 0]

    # thresholds
    WELL_LOCAL_FRAC = float(cfg["WELL_LOCAL_FRAC"])
    CHANNEL_HEAVY_FRAC = float(cfg["CHANNEL_HEAVY_FRAC"])
    USE_UNIFORM_EXTREME = bool(cfg["USE_UNIFORM_EXTREME"])
    UNIFORM_EXTREME_RATIO = float(cfg["UNIFORM_EXTREME_RATIO"])

    # tallies
    total_states = 0
    n_well_local = 0
    n_channel_heavy = 0
    n_uniform_extreme = 0

    extreme_records = []

    STATE_BATCH = int(cfg["STATE_BATCH"])
    ROW_CHUNK = int(cfg["ROW_CHUNK"])
    PROGRESS_EVERY = int(cfg["PROGRESS_EVERY"])

    with h5py.File(cfg["H5_PATH"], "r") as f:
        evals = np.array(f["evals"][:], dtype=np.float64)
        evecs = f["evecs"]
        N_pts, K = evecs.shape
        # sanity
        if N_pts != A_tot:
            print(f"[warn] H5 has {N_pts} points but allowed has {A_tot} — check consistency")

        for i0 in range(0, K, STATE_BATCH):
            i1 = min(K, i0 + STATE_BATCH)
            B = i1 - i0

            # we’ll accumulate per-state sums
            sum_tot = np.zeros(B, dtype=np.float64)
            sum_small = np.zeros(B, dtype=np.float64)
            sum_chan = np.zeros(B, dtype=np.float64)
            sum_large = np.zeros(B, dtype=np.float64)

            # stream rows
            for r0 in range(0, N_pts, ROW_CHUNK):
                r1 = min(N_pts, r0 + ROW_CHUNK)
                V = np.asarray(evecs[r0:r1, i0:i1])
                if np.iscomplexobj(V):
                    mag2 = (V.real * V.real + V.imag * V.imag).astype(np.float64, copy=False)
                else:
                    mag2 = (V * V).astype(np.float64, copy=False)

                sum_tot += mag2.sum(axis=0)

                # small
                if idx_small.size:
                    loc = idx_small[(idx_small >= r0) & (idx_small < r1)] - r0
                    if loc.size:
                        sum_small += mag2[loc, :].sum(axis=0)
                # channel
                if idx_chan.size:
                    loc = idx_chan[(idx_chan >= r0) & (idx_chan < r1)] - r0
                    if loc.size:
                        sum_chan += mag2[loc, :].sum(axis=0)
                # large
                if idx_large.size:
                    loc = idx_large[(idx_large >= r0) & (idx_large < r1)] - r0
                    if loc.size:
                        sum_large += mag2[loc, :].sum(axis=0)

            # classify the B states
            for j in range(B):
                k = i0 + j
                total_states += 1
                tot = sum_tot[j] + 1e-20
                frac_small = sum_small[j] / tot
                frac_chan = sum_chan[j] / tot
                frac_large = sum_large[j] / tot

                is_well_local = (frac_small >= WELL_LOCAL_FRAC) or (frac_large >= WELL_LOCAL_FRAC)
                is_channel_heavy = (frac_chan >= CHANNEL_HEAVY_FRAC)

                is_uniform_extreme = False
                if USE_UNIFORM_EXTREME:
                    # compare average densities against uniform baseline
                    # avg density = frac / area
                    ds = frac_small / max(A_small, 1)
                    dc = frac_chan / max(A_chan, 1)
                    dl = frac_large / max(A_large, 1)
                    # ratio vs uniform
                    rs = ds / baseline if baseline > 0 else 0.0
                    rc = dc / baseline if baseline > 0 else 0.0
                    rl = dl / baseline if baseline > 0 else 0.0
                    if (rs >= UNIFORM_EXTREME_RATIO) or (rc >= UNIFORM_EXTREME_RATIO) or (rl >= UNIFORM_EXTREME_RATIO):
                        is_uniform_extreme = True
                else:
                    rs = rc = rl = 0.0  # just for record

                if is_well_local:
                    n_well_local += 1
                if is_channel_heavy:
                    n_channel_heavy += 1
                if is_uniform_extreme:
                    n_uniform_extreme += 1

                if is_well_local or is_channel_heavy or is_uniform_extreme:
                    extreme_records.append({
                        "index": k,
                        "energy": float(evals[k]) if k < len(evals) else None,
                        "frac_small": float(frac_small),
                        "frac_channel": float(frac_chan),
                        "frac_large": float(frac_large),
                        "well_local": bool(is_well_local),
                        "channel_heavy": bool(is_channel_heavy),
                        "uniform_extreme": bool(is_uniform_extreme),
                        "ratio_small": float(rs),
                        "ratio_channel": float(rc),
                        "ratio_large": float(rl),
                    })

            if PROGRESS_EVERY and (total_states % PROGRESS_EVERY == 0):
                print(f"[progress] processed {total_states}/{K} states...", flush=True)

    # summarize
    pct_well_local = 100.0 * n_well_local / total_states if total_states else 0.0
    pct_channel_heavy = 100.0 * n_channel_heavy / total_states if total_states else 0.0
    pct_uniform_extreme = 100.0 * n_uniform_extreme / total_states if total_states else 0.0

    print("\n=== extreme state census ===")
    print(f"total states: {total_states}")
    print(f"well-localized (>= {WELL_LOCAL_FRAC:.2f} in small/large): {n_well_local} ({pct_well_local:.2f}%)")
    print(f"channel-heavy (>= {CHANNEL_HEAVY_FRAC:.2f} in channel):  {n_channel_heavy} ({pct_channel_heavy:.2f}%)")
    if USE_UNIFORM_EXTREME:
        print(f"uniform-extreme (ratio >= {UNIFORM_EXTREME_RATIO:.2f}): {n_uniform_extreme} ({pct_uniform_extreme:.2f}%)")

    # write CSV
    csv_path = os.path.join(cfg["OUT_DIR"], "extreme_states.csv")
    with open(csv_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=[
            "index", "energy",
            "frac_small", "frac_channel", "frac_large",
            "well_local", "channel_heavy", "uniform_extreme",
            "ratio_small", "ratio_channel", "ratio_large",
        ])
        w.writeheader()
        for rec in extreme_records:
            w.writerow(rec)
    print(f"[saved] {csv_path}")

    # write JSON summary
    summary_path = os.path.join(cfg["OUT_DIR"], "extreme_states_summary.json")
    with open(summary_path, "w") as fp:
        json.dump({
            "npz": cfg["NPZ_PATH"],
            "h5": cfg["H5_PATH"],
            "region_mode": cfg["REGION_MODE"],
            "areas": {
                "small": A_small,
                "channel": A_chan,
                "large": A_large,
                "total_allowed": A_tot,
            },
            "baseline_density": baseline,
            "thresholds": {
                "well_local_frac": WELL_LOCAL_FRAC,
                "channel_heavy_frac": CHANNEL_HEAVY_FRAC,
                "uniform_extreme_used": USE_UNIFORM_EXTREME,
                "uniform_extreme_ratio": UNIFORM_EXTREME_RATIO,
            },
            "counts": {
                "total_states": total_states,
                "well_local": n_well_local,
                "channel_heavy": n_channel_heavy,
                "uniform_extreme": n_uniform_extreme,
            },
            "percentages": {
                "well_local": pct_well_local,
                "channel_heavy": pct_channel_heavy,
                "uniform_extreme": pct_uniform_extreme,
            }
        }, fp, indent=2)
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
