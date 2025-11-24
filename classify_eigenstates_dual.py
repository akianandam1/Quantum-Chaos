#!/usr/bin/env python3
# classify_eigenstates_dual.py
#
# Two classification modes selectable at the top:
#   - CLASSIFY_MODE="threshold": original behavior — each eigenstate is assigned
#       to {small, channel, large, unclassified} using cutoff CUTOFF.
#   - CLASSIFY_MODE="average":   new behavior — for each eigenstate we compute
#       (frac_small, frac_channel, frac_large) and ADD these fractions to a
#       running sum. At the end we divide by the total number of states K and
#       print the average percentages across the spectrum.
#
# Region definition:
#   Use your existing manual label PNG (recommended), or component-split.
#
# >>> Edit the CONFIG block below <<<

CONFIG = {
    # ---- Region mode ----
    "REGION_MODE": "manual_label_png",   # "manual_label_png" or "component_split"
    "NPZ_PATH": "potentials/potential10.npz",
    "LABEL_PNG": "potentials/potential10_labelled.png",# for manual_label_png: 0 bg, 1 small, 2 channel, 3 large
    "CHANNEL_RADIUS": 9.0,               # for component_split only

    # ---- Eigenpairs ----
    "H5_PATH":  r"eigensdf/doublewell/potential10/trial1/eigenpairs.h5",

    # ---- Classification mode ----
    "CLASSIFY_MODE": "threshold",          # "threshold" or "average"
    "CUTOFF": 0.60,                      # used only when CLASSIFY_MODE="threshold"

    # ---- I/O and run knobs ----
    "OUT_DIR": r"eigensdf/doublewell/potential10/trial1",
    "BATCH": 1024,
    "PREVIEW": True,
    "SAVE_CSV": True,                    # save per-state fractions and (if threshold) label
    "PROGRESS_EVERY": 500,               # print rolling summary every N states (0 = off)
}

import os, json, warnings
import numpy as np
import h5py
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=SyntaxWarning)

# -------------------- Utilities --------------------
def load_npz(npz_path):
    g = np.load(npz_path)
    phi = g["phi"].astype(np.float32)
    allowed = g["allowed"].astype(bool) if "allowed" in g else (phi > 0)
    return phi, allowed

def build_index_map(allowed):
    Ny, Nx = allowed.shape
    idx_map = -np.ones((Ny, Nx), dtype=np.int64)
    idx_map[allowed] = np.arange(allowed.sum(), dtype=np.int64)
    return idx_map

def show_preview(small_mask, channel_mask, large_mask, preview_path):
    if not preview_path: return
    fig, ax = plt.subplots(1,3,figsize=(12,4), dpi=130)
    ax[0].imshow(small_mask, origin="lower", cmap="gray"); ax[0].set_title("Small well"); ax[0].axis("off")
    ax[1].imshow(channel_mask, origin="lower", cmap="gray"); ax[1].set_title("Channel"); ax[1].axis("off")
    ax[2].imshow(large_mask, origin="lower", cmap="gray"); ax[2].set_title("Large well"); ax[2].axis("off")
    fig.tight_layout(); fig.savefig(preview_path, bbox_inches="tight"); plt.close(fig)

def region_indices(idx_map, region_mask):
    idxs = idx_map[region_mask]
    return idxs[idxs >= 0].astype(np.int64)

# -------------------- Region: manual label PNG --------------------
def partition_manual_from_png(label_png, phi_shape, preview_path=None):
    if not label_png: raise ValueError("LABEL_PNG is empty.")
    try:
        from PIL import Image
        arr = np.array(Image.open(label_png))
    except Exception:
        import matplotlib.image as mpimg
        arr = mpimg.imread(label_png)

    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3]
        if rgb.dtype != np.uint8: rgb = (255*rgb).astype(np.uint8)
        r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
        lbl = np.zeros(r.shape, dtype=np.uint8)
        lbl[(r>200) & (g<50) & (b<50)] = 1  # red -> small
        lbl[(g>200) & (r<50) & (b<50)] = 2  # green -> channel
        lbl[(b>200) & (r<50) & (g<50)] = 3  # blue -> large
    else:
        lbl = arr.astype(np.uint8)

    if lbl.shape != phi_shape:
        raise ValueError(f"Label PNG shape {lbl.shape} does not match phi shape {phi_shape}.")

    small = (lbl==1); channel = (lbl==2); large = (lbl==3)
    show_preview(small, channel, large, preview_path)
    return small, channel, large

# -------------------- Region: component-split (minimal, robust) --------------------
def partition_component_split(phi, allowed, channel_radius=9.0, preview_path=None):
    d = np.where(allowed, np.maximum(phi, 0.0), 0.0)
    band = allowed & (d <= float(channel_radius))
    bodies = allowed & ~band
    lab, nlab = ndi.label(bodies)
    if nlab < 2:
        for rr in [channel_radius*1.2, channel_radius*1.4, channel_radius*1.6, channel_radius*1.8]:
            band = allowed & (d <= float(rr))
            bodies = allowed & ~band
            lab, nlab = ndi.label(bodies)
            if nlab >= 2:
                channel_radius = rr; break
    if nlab < 2:
        raise RuntimeError("Component split failed; increase CHANNEL_RADIUS.")

    # two largest bodies as seeds, grow back but not through the band
    sizes = ndi.sum(np.ones_like(lab), labels=lab, index=np.arange(1, nlab+1))
    order = np.argsort(sizes)[::-1]
    seedA = (lab == int(order[0]+1)); seedB = (lab == int(order[1]+1))
    growth_mask = allowed & ~band
    growA = ndi.binary_propagation(seedA, mask=growth_mask)
    growB = ndi.binary_propagation(seedB, mask=growth_mask)
    areaA, areaB = int(growA.sum()), int(growB.sum())
    if areaA >= areaB:
        large_mask, small_mask = growA, growB
    else:
        large_mask, small_mask = growB, growA

    # Channel: keep only band components that touch both wells (dilated 1 px)
    st = ndi.generate_binary_structure(2,1)
    smd = ndi.binary_dilation(small_mask, structure=st, iterations=1)
    lgd = ndi.binary_dilation(large_mask, structure=st, iterations=1)
    labc, nlabc = ndi.label(band, structure=st)
    keep = np.zeros_like(band, dtype=bool)
    for k in range(1, nlabc+1):
        comp = (labc == k)
        if (comp & smd).any() and (comp & lgd).any():
            keep |= comp
    channel = keep

    show_preview(small_mask, channel, large_mask, preview_path)
    return small_mask, channel, large_mask

# -------------------- Core classification logic --------------------
def classify(h5_path, idx_small, idx_chan, idx_large, mode="threshold", cutoff=0.80, batch=1024, out_csv=None, progress_every=0):
    with h5py.File(h5_path, "r") as f:
        evals = np.array(f["evals"][:], dtype=np.float64)
        evecs = f["evecs"]
        n, K = evecs.shape

        # storage for CSV
        rows = []
        # counters for threshold mode
        counts = {"small":0, "channel":0, "large":0, "unclassified":0}
        # accumulators for average mode
        sum_small = 0.0; sum_chan = 0.0; sum_large = 0.0

        processed = 0
        for i0 in range(0, K, batch):
            i1 = min(K, i0+batch)
            V = np.array(evecs[:, i0:i1], dtype=np.complex128)
            P = np.abs(V)**2
            s = P.sum(axis=0) + 1e-20
            P = P / s

            frac_small = P[idx_small, :].sum(axis=0) if idx_small.size else np.zeros((i1-i0,))
            frac_chan  = P[idx_chan,  :].sum(axis=0) if idx_chan.size  else np.zeros((i1-i0,))
            frac_large = P[idx_large, :].sum(axis=0) if idx_large.size else np.zeros((i1-i0,))

            for j in range(i1-i0):
                fs = float(frac_small[j]); fc = float(frac_chan[j]); fl = float(frac_large[j])
                # normalize tiny numerical drift so fs+fc+fl ~ 1
                tot = fs+fc+fl
                if tot > 0: fs, fc, fl = fs/tot, fc/tot, fl/tot

                if mode == "threshold":
                    label = max(("small","channel","large"), key=lambda k: {"small":fs,"channel":fc,"large":fl}[k])
                    if {"small":fs,"channel":fc,"large":fl}[label] >= cutoff:
                        counts[label] += 1
                    else:
                        counts["unclassified"] += 1
                    if out_csv is not None:
                        rows.append((i0+j, evals[i0+j], fs, fc, fl, label))
                else:  # "average"
                    sum_small += fs; sum_chan += fc; sum_large += fl
                    if out_csv is not None:
                        rows.append((i0+j, evals[i0+j], fs, fc, fl, ""))

                processed += 1
                if progress_every and (processed % progress_every == 0):
                    if mode == "threshold":
                        total = sum(counts.values())
                        pct = {k: 100.0*counts[k]/total if total>0 else 0.0 for k in counts}
                        print(f"[Progress {processed}/{K}] small={pct['small']:.2f}%  channel={pct['channel']:.2f}%  large={pct['large']:.2f}%  unclassified={pct['unclassified']:.2f}%", flush=True)
                    else:
                        # rolling mean so far
                        mean_s = 100.0*sum_small/processed
                        mean_c = 100.0*sum_chan /processed
                        mean_l = 100.0*sum_large/processed
                        print(f"[Progress {processed}/{K}] mean_small={mean_s:.2f}%  mean_channel={mean_c:.2f}%  mean_large={mean_l:.2f}%", flush=True)

        # finalize
        result = {}
        if mode == "threshold":
            total = sum(counts.values())
            pct = {k: 100.0*counts[k]/total if total>0 else 0.0 for k in counts}
            result["counts"] = counts
            result["percentages"] = pct
        else:
            K_total = evecs.shape[1]
            mean_small = sum_small / K_total
            mean_chan  = sum_chan  / K_total
            mean_large = sum_large / K_total
            result["average_fractions"] = {"small": mean_small, "channel": mean_chan, "large": mean_large}
            result["average_percentages"] = {k: 100.0*v for k,v in result["average_fractions"].items()}

    # CSV
    if out_csv is not None:
        import csv
        with open(out_csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["index","energy","frac_small","frac_channel","frac_large","label_or_blank"])
            for r in rows: w.writerow(r)

    return result

# -------------------- Main --------------------
def main():
    REGION_MODE   = CONFIG["REGION_MODE"]
    NPZ_PATH      = CONFIG["NPZ_PATH"]
    H5_PATH       = CONFIG["H5_PATH"]
    LABEL_PNG     = CONFIG["LABEL_PNG"]
    CHANNEL_RADIUS= float(CONFIG["CHANNEL_RADIUS"])
    CLASSIFY_MODE = CONFIG["CLASSIFY_MODE"]
    CUTOFF        = float(CONFIG["CUTOFF"])
    OUT_DIR       = CONFIG["OUT_DIR"]
    BATCH         = int(CONFIG["BATCH"])
    PREVIEW       = bool(CONFIG["PREVIEW"])
    SAVE_CSV      = bool(CONFIG["SAVE_CSV"])
    PROGRESS_EVERY= int(CONFIG["PROGRESS_EVERY"])

    os.makedirs(OUT_DIR, exist_ok=True)
    phi, allowed = load_npz(NPZ_PATH)
    idx_map = build_index_map(allowed)
    preview_path = os.path.join(OUT_DIR, "region_masks.png") if PREVIEW else None

    if REGION_MODE == "manual_label_png":
        small_mask, channel_mask, large_mask = partition_manual_from_png(LABEL_PNG, phi.shape, preview_path)
    elif REGION_MODE == "component_split":
        small_mask, channel_mask, large_mask = partition_component_split(phi, allowed, CHANNEL_RADIUS, preview_path)
    else:
        raise ValueError("REGION_MODE must be 'manual_label_png' or 'component_split'.")

    # Area fractions (sanity)
    A_tot = int(allowed.sum())
    f_small = 100.0*small_mask.sum()/A_tot
    f_chan  = 100.0*channel_mask.sum()/A_tot
    f_large = 100.0*large_mask.sum()/A_tot
    print(f"[Area Fractions] small={f_small:.3f}%  channel={f_chan:.3f}%  large={f_large:.3f}%  (total allowed={A_tot})")

    idx_small = region_indices(idx_map, small_mask)
    idx_chan  = region_indices(idx_map, channel_mask)
    idx_large = region_indices(idx_map, large_mask)

    csv_path = os.path.join(OUT_DIR, "eigenstate_fractions.csv") if SAVE_CSV else None
    result = classify(H5_PATH, idx_small, idx_chan, idx_large,
                      mode=CLASSIFY_MODE, cutoff=CUTOFF, batch=BATCH,
                      out_csv=csv_path, progress_every=PROGRESS_EVERY)

    if CLASSIFY_MODE == "threshold":
        counts = result["counts"]; pct = result["percentages"]
        print(f"[Classification Cutoff={CUTOFF:.2f}]")
        print("  small       : {0:6d}  ({1:6.2f}%)".format(counts["small"], pct["small"]))
        print("  channel     : {0:6d}  ({1:6.2f}%)".format(counts["channel"], pct["channel"]))
        print("  large       : {0:6d}  ({1:6.2f}%)".format(counts["large"], pct["large"]))
        print("  unclassified: {0:6d}  ({1:6.2f}%)".format(counts["unclassified"], pct["unclassified"]))
    else:
        meanp = result["average_percentages"]
        print("[Average mode] Mean percentage across eigenstates:")
        print("  small   : {0:6.2f}%".format(meanp["small"]))
        print("  channel : {0:6.2f}%".format(meanp["channel"]))
        print("  large   : {0:6.2f}%".format(meanp["large"]))

    # Save JSON summary
    summary = {
        "region_mode": REGION_MODE,
        "classify_mode": CLASSIFY_MODE,
        "npz": NPZ_PATH,
        "h5": H5_PATH,
        "label_png": LABEL_PNG if REGION_MODE=="manual_label_png" else None,
        "channel_radius": CHANNEL_RADIUS if REGION_MODE=="component_split" else None,
        "cutoff": CUTOFF,
        "area_fractions_percent": {"small": f_small, "channel": f_chan, "large": f_large},
    }
    summary.update(result)
    js_path = os.path.join(OUT_DIR, "summary_dual.json")
    with open(js_path, "w") as fp: json.dump(summary, fp, indent=2)
    print(f"[Saved] JSON -> {js_path}")
    if SAVE_CSV and csv_path:
        print(f"[Saved] CSV  -> {csv_path}")
    if PREVIEW:
        print(f"[Saved] preview -> {preview_path}")

if __name__ == "__main__":
    main()
