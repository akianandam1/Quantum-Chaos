#!/usr/bin/env python3
# eigenstate_probe.py
#
# Look up any eigenstate by index and report:
#   - its energy
#   - % probability in (small well, channel, large well)
#
# This version assumes you are using the **manual label PNG** method for regions:
#   label PNG (same size as phi) with:
#     0 = background
#     1 = small well
#     2 = channel
#     3 = large well
# OR pure colors: red(255,0,0)=1, green(0,255,0)=2, blue(0,0,255)=3
#
# >>> Edit the CONFIG block <<<

CONFIG = {
    "NPZ_PATH": r"potentials/potential7.npz",     # path to SDF npz (contains phi and allowed)
    "H5_PATH":  r"eigensdf/doublewell/trial1/eigenpairs.h5",  # path to eigenpairs.h5
    "LABEL_PNG": "potentials/labeled_potential.png",         # manual label image (see above)
    "INDICES": ['38716', '3085', '38678', '3086', '40840', '40858', '3249', '373', '38579', '262', '3214', '38603', '3097', '32076', '31935', '3206', '38719', '3282', '3118', '38801', '38689', '258', '3217', '3263', '3234', '3246', '38797', '38565', '353', '3252', '40846', '3141', '38755', '425', '3283', '40806', '38800', '31848', '38578', '38581', '38577', '32054', '38641', '38680', '3222', '31924', '31890', '5', '3219', '31908', '38728', '40807', '36405', '321', '430', '3095', '368', '3077', '429', '38752', '38799', '40860', '31844', '38807', '31954', '504', '3176', '38676', '38805', '40841', '31819', '38562', '3115', '38711', '38592', '3294', '3076', '38609', '15', '40782', '277', '38600', '259', '36576', '38683', '3131', '31928', '19', '3215', '503', '31915', '38718', '403'],# energy 2.3 channel[15234,5756,15239,33957,15237,15238], # energy 1 channel[6518, 33201, 932, 933, 36424, 931, 33197, 36425,32201,969],                               # eigenstate index or list of indices
    "SAVE_JSON": True,                            # save results to probe_results.json
    "OUT_DIR": "eigensdf/doublewell/trial1/",                   # where to save json (if enabled)
}



import os, json, sys
import numpy as np
import h5py

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

def load_labels_png(label_png, shape):
    if not label_png:
        raise ValueError("LABEL_PNG is empty; please set CONFIG['LABEL_PNG'].")
    try:
        from PIL import Image
        arr = np.array(Image.open(label_png))
    except Exception:
        import matplotlib.image as mpimg
        arr = mpimg.imread(label_png)
    # If RGB(A), map pure colors to labels 1/2/3. Otherwise treat as integer labels.
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3]
        if rgb.dtype != np.uint8:
            rgb = (255*rgb).astype(np.uint8)
        r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
        lbl = np.zeros(r.shape, dtype=np.uint8)
        lbl[(r>200) & (g<50) & (b<50)] = 1  # red -> small
        lbl[(g>200) & (r<50) & (b<50)] = 2  # green -> channel
        lbl[(b>200) & (r<50) & (g<50)] = 3  # blue -> large
    else:
        lbl = arr.astype(np.uint8)
    if lbl.shape != shape:
        raise ValueError(f"Label PNG shape {lbl.shape} does not match phi shape {shape}.")
    small = (lbl == 1)
    channel = (lbl == 2)
    large = (lbl == 3)
    return small, channel, large

def region_indices(idx_map, region_mask):
    idxs = idx_map[region_mask]
    return idxs[idxs >= 0].astype(np.int64)

def probe_indices(h5_path, idx_small, idx_chan, idx_large, indices):
    out = []
    with h5py.File(h5_path, "r") as f:
        evals = np.array(f["evals"][:], dtype=np.float64)
        evecs = f["evecs"]
        n, K = evecs.shape
        # Normalize indices to list
        if isinstance(indices, (int, np.integer)):
            indices = [int(indices)]
        else:
            indices = [int(i) for i in indices]
        for k in indices:
            if k < 0 or k >= K:
                out.append({"index": k, "error": f"Index {k} out of range [0, {K-1}]"})
                continue
            v = np.array(evecs[:, k], dtype=np.complex128)
            p = np.abs(v)**2
            s = float(p.sum()) + 1e-20
            p /= s
            fs = float(p[idx_small].sum()) if idx_small.size else 0.0
            fc = float(p[idx_chan].sum()) if idx_chan.size else 0.0
            fl = float(p[idx_large].sum()) if idx_large.size else 0.0
            # Numerical cleanup: ensure fractions sum to ~1.0
            tot = fs + fc + fl
            if tot > 0:
                fs, fc, fl = fs/tot, fc/tot, fl/tot
            out.append({
                "index": k,
                "energy": float(evals[k]) if k < len(evals) else None,
                "frac_small": fs,
                "frac_channel": fc,
                "frac_large": fl,
                "percent_small": 100.0*fs,
                "percent_channel": 100.0*fc,
                "percent_large": 100.0*fl,
            })
    return out

def main():
    NPZ = CONFIG["NPZ_PATH"]
    H5  = CONFIG["H5_PATH"]
    LAB = CONFIG["LABEL_PNG"]
    OUT_DIR = CONFIG["OUT_DIR"]
    SAVE_JSON = bool(CONFIG.get("SAVE_JSON", True))
    indices = CONFIG["INDICES"]

    phi, allowed = load_npz(NPZ)
    idx_map = build_index_map(allowed)
    small_mask, channel_mask, large_mask = load_labels_png(LAB, phi.shape)

    idx_small = region_indices(idx_map, small_mask)
    idx_chan  = region_indices(idx_map, channel_mask)
    idx_large = region_indices(idx_map, large_mask)

    results = probe_indices(H5, idx_small, idx_chan, idx_large, indices)

    # Pretty print
    for r in results:
        if "error" in r:
            print(f"[{r['index']}] ERROR: {r['error']}")
        else:
            print(f"[{r['index']}] E = {r['energy']:.6f}  |  small={r['percent_small']:.2f}%  channel={r['percent_channel']:.2f}%  large={r['percent_large']:.2f}%")

    if SAVE_JSON:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, "probe_results.json")
        with open(out_path, "w") as fp:
            json.dump(results, fp, indent=2)
        print(f"[Saved] {out_path}")

if __name__ == "__main__":
    main()
