#!/usr/bin/env python3
# region_density_by_state.py
#
# For every eigenstate, compute the *average probability density* in each
# region (small well, channel, large well). We define
#
#   avg_density_region(n) = (sum_{i in region} |psi_n(i)|^2) / (#pixels in region)
#                         = frac_region(n) / area_region
#
# Baseline uniform density across the total allowed set:
#   baseline = 1 / area_total_allowed
#
# Outputs (in OUT_DIR):
#   - region_avg_densities_by_state.csv  (Excel-friendly)
#   - region_avg_density_summary.json    (areas + baseline)
#   - region_masks.png                   (preview of masks, if PREVIEW=True)
#   - (optional) Parquet if pandas+pyarrow available
#
# --------------------- CONFIG: edit here ---------------------


# baseline 1/Total Area = 3.033318e-06

CONFIG = {
    "REGION_MODE": "manual_label_png",   # "manual_label_png" or "component_split"
    "NPZ_PATH": r"potentials/potential11.npz",
    "LABEL_PNG": "potentials/potential11_labelled.png",
    "CHANNEL_RADIUS": 9.0,               # for component_split only

    "H5_PATH":  r"eigensdf/doublewell/potential11/trial1/eigenpairs.h5",

    "OUT_DIR": r"eigensdf/doublewell/potential11/trial1",
    "BATCH": 100,
    "PROGRESS_EVERY": 100,
    "PREVIEW": True,
    "SAVE_CSV": True,
    "SAVE_PARQUET": False,
}

import os, json, warnings, csv
import numpy as np
import h5py
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import time


warnings.filterwarnings("ignore", category=SyntaxWarning)

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

def show_preview_masks(small_mask, channel_mask, large_mask, preview_path):
    if not preview_path: return
    fig, ax = plt.subplots(1,3,figsize=(12,4), dpi=130)
    ax[0].imshow(small_mask, origin="lower", cmap="gray"); ax[0].set_title("Small well"); ax[0].axis("off")
    ax[1].imshow(channel_mask, origin="lower", cmap="gray"); ax[1].set_title("Channel"); ax[1].axis("off")
    ax[2].imshow(large_mask, origin="lower", cmap="gray"); ax[2].set_title("Large well"); ax[2].axis("off")
    fig.tight_layout(); fig.savefig(preview_path, bbox_inches="tight"); plt.close(fig)

def region_indices(idx_map, region_mask):
    idxs = idx_map[region_mask]
    return idxs[idxs >= 0].astype(np.int64)

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
    show_preview_masks(small, channel, large, preview_path)
    return small, channel, large

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

    show_preview_masks(small_mask, channel, large_mask, preview_path)
    return small_mask, channel, large_mask

def compute_region_densities(h5_path, idx_small, idx_chan, idx_large, areas, batch=1024, progress_every=0, out_csv=None, out_parquet=None):
    A_small, A_chan, A_large, A_tot = areas
    baseline = 1.0 / float(A_tot)

    rows = []
    with h5py.File(h5_path, "r") as f:
        evals = np.array(f["evals"][:], dtype=np.float64)
        evecs = f["evecs"]
        n_pts, K = evecs.shape

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

            avg_small = frac_small / max(A_small, 1)
            avg_chan  = frac_chan  / max(A_chan,  1)
            avg_large = frac_large / max(A_large, 1)

            ratio_small = avg_small / baseline
            ratio_chan  = avg_chan  / baseline
            ratio_large = avg_large / baseline

            for j in range(i1-i0):
                idx = i0 + j
                e   = float(evals[idx]) if idx < len(evals) else None
                rows.append((
                    idx, e,
                    float(avg_small[j]), float(avg_chan[j]), float(avg_large[j]),
                    float(frac_small[j]), float(frac_chan[j]), float(frac_large[j]),
                    float(ratio_small[j]), float(ratio_chan[j]), float(ratio_large[j]),
                    float(baseline)
                ))

            processed += (i1 - i0)
            if progress_every and (processed % progress_every == 0):
                print(f"[Progress] processed {processed}/{K} eigenstates...", flush=True)

    if out_csv is not None:
        with open(out_csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow([
                "index", "energy",
                "avg_density_small", "avg_density_channel", "avg_density_large",
                "frac_small", "frac_channel", "frac_large",
                "ratio_vs_uniform_small", "ratio_vs_uniform_channel", "ratio_vs_uniform_large",
                "uniform_baseline_1_over_area_total"
            ])
            for r in rows:
                w.writerow(r)

    if out_parquet is not None:
        try:
            import pandas as pd
            df = pd.DataFrame(rows, columns=[
                "index","energy",
                "avg_density_small","avg_density_channel","avg_density_large",
                "frac_small","frac_channel","frac_large",
                "ratio_vs_uniform_small","ratio_vs_uniform_channel","ratio_vs_uniform_large",
                "uniform_baseline_1_over_area_total"
            ])
            df.to_parquet(out_parquet, index=False)
            print(f"[Saved] Parquet -> {out_parquet}")
        except Exception as e:
            print(f"[Parquet skipped] {e}")

    return rows

def main():
    REGION_MODE   = CONFIG["REGION_MODE"]
    NPZ_PATH      = CONFIG["NPZ_PATH"]
    H5_PATH       = CONFIG["H5_PATH"]
    LABEL_PNG     = CONFIG["LABEL_PNG"]
    CHANNEL_RADIUS= float(CONFIG["CHANNEL_RADIUS"])
    OUT_DIR       = CONFIG["OUT_DIR"]
    BATCH         = int(CONFIG["BATCH"])
    PROGRESS_EVERY= int(CONFIG["PROGRESS_EVERY"])
    PREVIEW       = bool(CONFIG["PREVIEW"])
    SAVE_CSV      = bool(CONFIG["SAVE_CSV"])
    SAVE_PARQUET  = bool(CONFIG["SAVE_PARQUET"])

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

    A_small = int(small_mask.sum())
    A_chan  = int(channel_mask.sum())
    A_large = int(large_mask.sum())
    A_tot   = int(allowed.sum())

    baseline = 1.0 / float(A_tot) if A_tot > 0 else 0.0
    print(f"[Areas] small={A_small}  channel={A_chan}  large={A_large}  total_allowed={A_tot}")
    print(f"[Uniform baseline density] 1/Area_total = {baseline:.6e}")

    idx_small = region_indices(idx_map, small_mask)
    idx_chan  = region_indices(idx_map, channel_mask)
    idx_large = region_indices(idx_map, large_mask)

    out_csv = os.path.join(OUT_DIR, "region_avg_densities_by_state.csv") if SAVE_CSV else None
    out_parquet = os.path.join(OUT_DIR, "region_avg_densities_by_state.parquet") if SAVE_PARQUET else None

    _ = compute_region_densities(
        H5_PATH, idx_small, idx_chan, idx_large,
        areas=(A_small, A_chan, A_large, A_tot),
        batch=BATCH, progress_every=PROGRESS_EVERY,
        out_csv=out_csv, out_parquet=out_parquet
    )

    summary = {
        "npz": NPZ_PATH, "h5": H5_PATH, "region_mode": REGION_MODE,
        "label_png": LABEL_PNG if REGION_MODE=="manual_label_png" else None,
        "channel_radius": CHANNEL_RADIUS if REGION_MODE=="component_split" else None,
        "areas": {"small": A_small, "channel": A_chan, "large": A_large, "total_allowed": A_tot},
        "uniform_baseline": baseline
    }
    js_path = os.path.join(OUT_DIR, "region_avg_density_summary.json")
    with open(js_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[Saved] JSON summary -> {js_path}")
    if out_csv: print(f"[Saved] CSV -> {out_csv}")
    if out_parquet: print(f"[Saved] Parquet -> {out_parquet}")

if __name__ == "__main__":
    start=time.time()
    main()
    end=time.time()
    print(f"Finished in {end-start} seconds")
