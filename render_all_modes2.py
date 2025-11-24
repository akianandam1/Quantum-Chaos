#!/usr/bin/env python3
# render_all_modes.py
# Re-render eigenfunction montages from what's already saved on disk.
# Works with HDF5 (eigenpairs.h5) or NPZ shards (manifest.json).

# ========= USER PARAMS =========
OUT_DIR         = "eigensdf/doublewell/roland_narrow1"                      # where your store lives
OUTPUT_DIR = "eigensdf/doublewell/roland_narrow1/sortedrenders"                    # where we output images
SDF_PATH        = "potentials/potential8sdf.npz"     # fallback if geometry_used.npz not present
CONTOURS_PATH   = None  # e.g., "eigs_out/geometry_sdf_contours.npz" or None

# What to render
RENDER_REAL     = False    # set True if you also want Re(psi) montages
RENDER_PROB     = True     # probability |psi|^2 montages with high contrast

# Montage layout & look
MONTAGE_COUNT   = 60       # modes per montage image
MONTAGE_COLS    = 10
MONTAGE_DPI     = 150
ALPHA_OVERLAY   = 0.35     # blend eigenimage over occupancy background (0..1). Try 0.25–0.45.
SHOW_BACKGROUND = True     # False -> pure field (no white/gray bg)

# Probability contrast controls
PROB_PCT_HI     = 99.5     # clip upper percentile (e.g., 99.0–99.9)
PROB_GAMMA      = 0.6      # gamma < 1 boosts mid-tones (0.5–0.8 good)
PROB_CMAP       = "inferno"  # sequential: "inferno", "magma", "plasma"
# ===============================

import os, json, numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- load geometry (preferred: what you solved with) ----------
geom_used = os.path.join(OUT_DIR, "geometry_used.npz")
if os.path.exists(geom_used):
    g = np.load(geom_used)
    Ny, Nx = g["shape"]
    allowed = g["allowed"].astype(bool)
    occ = g["occ"].astype(np.float32) if "occ" in g.files else allowed.astype(np.float32)
else:
    # fallback to SDF pack (phi>0 defines interior)
    g = np.load(SDF_PATH)
    phi = g["phi"].astype(np.float32)
    occ = g["occ"].astype(np.float32) if "occ" in g.files else (phi > 0).astype(np.float32)
    allowed = (phi > 0.0)
    Ny, Nx = phi.shape

n = int(allowed.sum())
print(f"[geom] grid {Ny} x {Nx}, DOFs n={n}")

# Optional: load smooth boundary contours (if you saved them)
contours = None
if CONTOURS_PATH and os.path.exists(CONTOURS_PATH):
    cpack = np.load(CONTOURS_PATH, allow_pickle=True)
    contours = list(cpack["contours"])
    print(f"[geom] loaded {len(contours)} φ=0 contour polylines")

def unpack(vec):
    out = np.zeros((Ny, Nx), dtype=np.complex128)
    out[allowed] = vec.reshape(-1)
    return out

# ---------- colormaps ----------
REAL_CMAP = cm.RdBu_r
try:
    PROB_CMAP_FN = cm.get_cmap(PROB_CMAP)
except Exception:
    PROB_CMAP_FN = cm.inferno

# ---------- renderer ----------
def save_modes_montage(evals, evecs, global_start, which):
    total = evecs.shape[1]
    if total == 0: return

    if SHOW_BACKGROUND:
        bg = (255*np.dstack([occ,occ,occ])).astype(np.uint8)
    else:
        bg = None

    for offset in range(0, total, MONTAGE_COUNT):
        stop = min(offset + MONTAGE_COUNT, total)
        rows = int(np.ceil((stop-offset)/MONTAGE_COLS))
        fig, axes = plt.subplots(rows, MONTAGE_COLS,
                                 figsize=(1.6*MONTAGE_COLS, 1.6*rows),
                                 dpi=MONTAGE_DPI)
        axes = np.array(axes).ravel()

        last_i = -1
        for i, k in enumerate(range(offset, stop)):
            ax = axes[i]
            psi_img = unpack(evecs[:, k])

            if which == "real":
                re = psi_img.real
                s = np.max(np.abs(re)) + 1e-12
                x = 0.5*(re/s + 1.0)
                img = (REAL_CMAP(x)[:, :, :3]*255).astype(np.uint8)
            else:
                # --- High-contrast probability renderer ---
                p = np.abs(psi_img)**2
                p /= (p.sum() + 1e-20)                        # normalize prob
                hi = np.percentile(p, PROB_PCT_HI)            # robust upper bound
                scale = hi if hi > 1e-20 else p.max() + 1e-20
                p = np.clip(p / scale, 0.0, 1.0)
                if PROB_GAMMA != 1.0:
                    p = p**PROB_GAMMA
                img = (PROB_CMAP_FN(p)[:, :, :3]*255).astype(np.uint8)

            if bg is not None:
                blend = (ALPHA_OVERLAY*img + (1-ALPHA_OVERLAY)*bg).astype(np.uint8)
            else:
                blend = img

            gidx = global_start + k
            ax.imshow(blend, origin='lower')
            ax.set_title(f"#{gidx}  E≈{evals[k]:.6g}", fontsize=7)
            ax.axis('off')

            # Optional overlay of φ=0 curve
            if contours:
                for c in contours:
                    ax.plot(c[:,1], c[:,0], 'k-', lw=0.8, alpha=0.7)

            last_i = i

        # turn off unused axes
        for j in range(last_i+1, axes.size):
            axes[j].axis('off')

        plt.tight_layout()
        tag = f"{which}_{global_start+offset:06d}_{global_start+stop-1:06d}"
        out = os.path.join(OUTPUT_DIR, f"modes_{tag}.png")
        plt.savefig(out, bbox_inches="tight"); plt.close(fig)
        print("wrote", out)

# ---------- detect backend and render ----------
h5_path = os.path.join(OUT_DIR, "eigenpairs_sorted.h5")
man_path = os.path.join(OUT_DIR, "manifest.json")

if os.path.exists(h5_path):
    try:
        import h5py
    except ImportError:
        raise SystemExit("HDF5 store found but h5py not installed. pip install h5py")

    with h5py.File(h5_path, "r") as h5:
        K = int(h5["evals"].shape[0])
        print(f"[store:h5] total modes: {K}")
        CHUNK = 240  # render in manageable blocks
        for start in range(0, K, CHUNK):
            end = min(start+CHUNK, K)
            E = np.array(h5["evals"][start:end])
            V = np.array(h5["evecs"][:, start:end])
            if RENDER_REAL:
                save_modes_montage(E, V, global_start=start, which="real")
            if RENDER_PROB:
                save_modes_montage(E, V, global_start=start, which="prob")

elif os.path.exists(man_path):
    with open(man_path, "r") as f:
        man = json.load(f)
    shards = man.get("shards", [])
    print(f"[store:npz] shards: {len(shards)}")
    for sh in shards:
        start = int(sh["start"]); stop = int(sh["stop"])
        npz = os.path.join(OUT_DIR, sh["path"])
        data = np.load(npz)
        E = np.asarray(data["evals"])
        V = np.asarray(data["evecs"])
        if RENDER_REAL:
            save_modes_montage(E, V, global_start=start, which="real")
        if RENDER_PROB:
            save_modes_montage(E, V, global_start=start, which="prob")
else:
    raise SystemExit("No eigenpair store found in OUT_DIR (missing eigenpairs.h5 and manifest.json).")

print("All done.")