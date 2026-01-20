#!/usr/bin/env python3
# area_percentages_quick.py
#
# Drop-in helper: print the **area percentages** of (small, channel, large)
# using the same region-recognition you’ve been using.
#
# Edit CONFIG below and just run this file. It prints a one-liner summary,
# plus a slightly more detailed block, and writes a tiny JSON/CSV.
#
CONFIG = {
    "REGION_MODE": "manual_label_png",   # "manual_label_png" or "component_split"
    "NPZ_PATH": r"potentials/final_potential.npz",
    "LABEL_PNG": r"potentials/labelled_final_potential.png",# if manual_label_png: 0 bg, 1 small, 2 channel, 3 large (or pure RGB)
    "CHANNEL_RADIUS": 9.0,               # for component_split only

    "OUT_DIR": r"eigensdf/doublewell/final",
    "PREVIEW": True,                     # save region_masks.png
}

import os, json, warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

warnings.filterwarnings("ignore", category=SyntaxWarning)

def load_npz(npz_path):
    g = np.load(npz_path)
    phi = g["phi"].astype(np.float32)
    allowed = g["allowed"].astype(bool) if "allowed" in g else (phi > 0)
    return phi, allowed

def show_preview_masks(small_mask, channel_mask, large_mask, preview_path):
    if not preview_path: return
    fig, ax = plt.subplots(1,3,figsize=(11,3.5), dpi=130)
    ax[0].imshow(small_mask, origin="lower", cmap="gray"); ax[0].set_title("Small"); ax[0].axis("off")
    ax[1].imshow(channel_mask, origin="lower", cmap="gray"); ax[1].set_title("Channel"); ax[1].axis("off")
    ax[2].imshow(large_mask, origin="lower", cmap="gray"); ax[2].set_title("Large"); ax[2].axis("off")
    fig.tight_layout(); fig.savefig(preview_path, bbox_inches="tight"); plt.close(fig)

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

def main():
    REGION_MODE   = CONFIG["REGION_MODE"]
    NPZ_PATH      = CONFIG["NPZ_PATH"]
    LABEL_PNG     = CONFIG["LABEL_PNG"]
    CHANNEL_RADIUS= float(CONFIG["CHANNEL_RADIUS"])
    OUT_DIR       = CONFIG["OUT_DIR"]
    PREVIEW       = bool(CONFIG["PREVIEW"])

    os.makedirs(OUT_DIR, exist_ok=True)
    phi, allowed = load_npz(NPZ_PATH)
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

    p_small = 100.0 * A_small / A_tot if A_tot else 0.0
    p_chan  = 100.0 * A_chan  / A_tot if A_tot else 0.0
    p_large = 100.0 * A_large / A_tot if A_tot else 0.0

    print(f"[Areas %] small={p_small:.3f}% | channel={p_chan:.3f}% | large={p_large:.3f}%  (total allowed px={A_tot})")

    print("--- Area breakdown ---")
    print(f"small : {A_small:8d} px  ({p_small:6.3f}%)")
    print(f"channel: {A_chan:8d} px  ({p_chan:6.3f}%)")
    print(f"large : {A_large:8d} px  ({p_large:6.3f}%)")
    print(f"TOTAL : {A_tot:8d} px")

    data = {
        "areas": {"small": A_small, "channel": A_chan, "large": A_large, "total_allowed": A_tot},
        "percentages": {"small": p_small, "channel": p_chan, "large": p_large},
        "npz": NPZ_PATH, "region_mode": REGION_MODE,
        "label_png": LABEL_PNG if REGION_MODE=="manual_label_png" else None,
        "channel_radius": CHANNEL_RADIUS if REGION_MODE=="component_split" else None
    }
    js_path = os.path.join(OUT_DIR, "area_percentages.json")
    with open(js_path, "w") as f:
        json.dump(data, f, indent=2)

    csv_path = os.path.join(OUT_DIR, "area_percentages.csv")
    with open(csv_path, "w") as f:
        f.write("region,area_pixels,percent")
        f.write(f"small,{A_small},{p_small}")
        f.write(f"channel,{A_chan},{p_chan}")
        f.write(f"large,{A_large},{p_large}")
        f.write(f"total_allowed,{A_tot},100.0")

    print(f"[Saved] JSON -> {js_path}")
    print(f"[Saved] CSV  -> {csv_path}")
    if PREVIEW:
        print(f"[Saved] preview -> {preview_path}")

if __name__ == "__main__":
    main()
