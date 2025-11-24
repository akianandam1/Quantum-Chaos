#!/usr/bin/env python3
"""
partial_barrier_analysis_regions.py

End-to-end script to test the <w_12> vs x = phi/h curve
for the bent-well geometry.

- Reads SDF NPZ (phi, allowed).
- Reads an RGB-labelled PNG (red=small, green=channel, blue=large),
  and builds small/channel/large masks exactly like region_density_by_state.py.
- Writes a quick preview PNG of the three regions.
- Combines (small + channel) into a single "small+channel" region for analysis.
- Loads eigenpairs from eigenpairs.h5.
- Walks over windows of eigenstates, computes <w_12> for each window.
- Estimates a crude classical flux phi by launching trajectories in the large well.
- Sets h = 1 / N_ch (N_ch = number of states in the window after optional filtering).
- Plots <w_12> vs x = phi/h and overlays x/(1+x).

All user parameters live in CONFIG below.
"""

import os
import json
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt

# ===================== USER CONFIG =====================

CONFIG = {
    # Geometry / potential / labelling
    "NPZ_PATH": r"potentials/potential8.npz",   # SDF npz (phi, allowed)
    "LABEL_PNG": r"potentials/labeled_potential8.png",  # RGB-labelled regions
    "GEOM_PREVIEW_PNG": r"eigensdf/doublewell/trial2/roland/region_masks_preview.png",

    # Eigenpairs
    "H5_PATH": r"eigensdf/doublewell/trial2/eigenpairs.h5",
    "OUT_DIR": r"eigensdf/doublewell/trial2/roland",

    # Eigenstate window parameters
    "START_INDEX": 200,   # 0-based starting eigenstate index
    "WINDOW_SIZE": 25,     # number of states per window
    "NUM_WINDOWS": 5,     # how many windows to try

    # Optional crude chaoticity filter:
    # require p_i >= MIN_REL_FRAC * mu_i_cl in BOTH regions to keep state
    "MIN_REL_FRAC": 0.0,   # set >0.0 to discard strongly localized states

    # Classical flux parameters
    "N_TRAJ": 5000,        # number of trajectories
    "MAX_STEPS": 800,      # steps per trajectory
    "STEP_LEN": 0.5,       # step length in pixels
    "RANDOM_SEED": 12345,

    # Output files
    "PLOT_PATH": r"eigensdf/doublewell/trial2/roland/partial_barrier_curve.png",
    "DATA_OUT_NPZ": r"eigensdf/doublewell/trial2/roland/partial_barrier_data.npz",
}

# =======================================================


# -------------------- GEOMETRY & MASKS -------------------- #

def load_npz(npz_path):
    g = np.load(npz_path)
    phi = g["phi"].astype(np.float32)
    if "allowed" in g:
        allowed = g["allowed"].astype(bool)
    else:
        allowed = (phi > 0)
    return phi, allowed

def show_preview_masks(small_mask, channel_mask, large_mask, preview_path=None):
    if preview_path is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=130)
    ax[0].imshow(small_mask, origin="lower", cmap="gray")
    ax[0].set_title("Small well")
    ax[0].axis("off")
    ax[1].imshow(channel_mask, origin="lower", cmap="gray")
    ax[1].set_title("Channel")
    ax[1].axis("off")
    ax[2].imshow(large_mask, origin="lower", cmap="gray")
    ax[2].set_title("Large well")
    ax[2].axis("off")
    fig.tight_layout()
    fig.savefig(preview_path, bbox_inches="tight")
    plt.close(fig)

def partition_manual_from_png(label_png, phi_shape, preview_path=None):
    """
    Reproduces the logic from region_density_by_state.partition_manual_from_png:
      red   -> small well
      green -> channel
      blue  -> large well
    """
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
        # thresholds copied from your existing code
        lbl[(r > 200) & (g < 50) & (b < 50)] = 1  # red -> small
        lbl[(g > 200) & (r < 50) & (b < 50)] = 2  # green -> channel
        lbl[(b > 200) & (r < 50) & (g < 50)] = 3  # blue -> large
    else:
        lbl = arr.astype(np.uint8)

    if lbl.shape != phi_shape:
        raise ValueError(f"Label PNG shape {lbl.shape} does not match phi shape {phi_shape}.")

    small_mask = (lbl == 1)
    channel_mask = (lbl == 2)
    large_mask = (lbl == 3)

    show_preview_masks(small_mask, channel_mask, large_mask, preview_path)
    return small_mask, channel_mask, large_mask

def load_geometry_and_region_masks(config):
    npz_path = config["NPZ_PATH"]
    label_png = config["LABEL_PNG"]
    preview_png = config["GEOM_PREVIEW_PNG"]

    phi, allowed = load_npz(npz_path)
    Ny, Nx = phi.shape

    small_mask, channel_mask, large_mask = partition_manual_from_png(label_png, phi.shape, preview_png)

    # restrict to allowed region
    small_mask &= allowed
    channel_mask &= allowed
    large_mask &= allowed

    if not small_mask.any():
        raise RuntimeError("small_mask is empty after intersecting with allowed.")
    if not large_mask.any():
        raise RuntimeError("large_mask is empty after intersecting with allowed.")

    # combine small + channel into one region for our purposes
    small_combined = (small_mask | channel_mask) & allowed

    # compute classical area fractions over (small_combined + large)
    union_mask = (small_combined | large_mask) & allowed
    total_area = union_mask.sum()
    mu1_cl = small_combined.sum() / total_area   # region 1: small+channel
    mu2_cl = large_mask.sum() / total_area       # region 2: large

    print(f"[geom] phi shape={phi.shape}, allowed cells={allowed.sum()}")
    print(f"[geom] small area={small_combined.sum()}, large area={large_mask.sum()}, total={total_area}")
    print(f"[geom] mu_small_cl={mu1_cl:.4f}, mu_large_cl={mu2_cl:.4f}")

    return phi, allowed, small_combined, large_mask, (Ny, Nx), (mu1_cl, mu2_cl)


# -------------------- EIGENSTATES / <w_12> -------------------- #

def load_eigenpairs(h5_path):
    f = h5py.File(h5_path, "r")
    evals = np.array(f["evals"][:], dtype=np.float64)
    return f, evals

def unpack_evec(vec, allowed_mask, shape):
    Ny, Nx = shape
    psi = np.zeros((Ny, Nx), dtype=np.complex128)
    psi[allowed_mask] = vec.reshape(-1)
    return psi

def compute_w12_for_window(h5_file,
                           allowed,
                           mask_small,
                           mask_large,
                           mu_small_cl,
                           mu_large_cl,
                           shape,
                           idx_start,
                           window_size,
                           min_rel_frac=0.0):
    """
    Compute <w_12> in a window [idx_start, idx_start+window_size).

    Region 1: small+channel
    Region 2: large

    w_12^j = (p_small / mu_small_cl) * (p_large / mu_large_cl)
    """
    idx_end = idx_start + window_size
    evecs = np.array(h5_file["evecs"][:, idx_start:idx_end], dtype=np.complex128)
    K = evecs.shape[1]

    Ny, Nx = shape
    w_list = []
    idx_kept = []

    for j in range(K):
        v = evecs[:, j]
        psi = unpack_evec(v, allowed, shape)
        dens = np.abs(psi) ** 2
        tot = dens[allowed].sum()
        if not np.isfinite(tot) or tot <= 0:
            continue
        dens /= tot

        p_small = dens[mask_small].sum()
        p_large = dens[mask_large].sum()

        # optional crude chaoticity filter
        if min_rel_frac > 0.0:
            if (p_small < min_rel_frac * mu_small_cl) or (p_large < min_rel_frac * mu_large_cl):
                continue

        r1 = p_small / mu_small_cl
        r2 = p_large / mu_large_cl
        w12 = r1 * r2
        w_list.append(w12)
        idx_kept.append(idx_start + j)

    if not w_list:
        return np.nan, [], np.array([], dtype=int)

    w_arr = np.array(w_list, dtype=float)
    w_mean = w_arr.mean()
    return w_mean, w_arr, np.array(idx_kept, dtype=int)


# -------------------- CLASSICAL FLUX ESTIMATE -------------------- #

def sample_phi(phi, x, y):
    """
    Bilinear sampling of phi at (x,y) in pixel coordinates.
    Returns negative if clearly outside domain.
    """
    Ny, Nx = phi.shape
    if x < 0 or x >= (Nx - 1) or y < 0 or y >= (Ny - 1):
        return -1.0
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    dx = x - x0
    dy = y - y0
    x1 = min(x0 + 1, Nx - 1)
    y1 = min(y0 + 1, Ny - 1)

    f00 = phi[y0, x0]
    f10 = phi[y0, x1]
    f01 = phi[y1, x0]
    f11 = phi[y1, x1]

    f0 = f00 * (1 - dx) + f10 * dx
    f1 = f01 * (1 - dx) + f11 * dx
    return f0 * (1 - dy) + f1 * dy

def compute_normal(phi, x, y):
    """
    Approximate inward normal from grad(phi) at nearest pixel.
    phi > 0 inside allowed region; grad points outward, so we flip sign.
    """
    Ny, Nx = phi.shape
    xi = int(np.clip(round(x), 1, Nx - 2))
    yi = int(np.clip(round(y), 1, Ny - 2))

    dphidx = 0.5 * (phi[yi, xi + 1] - phi[yi, xi - 1])
    dphidy = 0.5 * (phi[yi + 1, xi] - phi[yi - 1, xi])

    n = np.array([dphidx, dphidy], dtype=float)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return np.array([0.0, 1.0], dtype=float)
    n /= norm
    # grad(phi) points outward (phi>0 inside); we want inward normal for reflection
    return -n

def estimate_flux(phi,
                  allowed,
                  mask_small,
                  mask_large,
                  n_traj=5000,
                  max_steps=800,
                  step_len=0.5,
                  seed=12345):
    """
    Very simple classical flux estimate:

    - start points uniform in the large region
    - random directions
    - billiard-like propagation with SDF-based reflections
    - count fraction of trajectories that ever cross from large to small+channel.

    This is not a rigorous SOS flux, but gives a reasonable φ for building x = φ/h.
    """
    rng = np.random.default_rng(seed)
    Ny, Nx = phi.shape

    # region id: 0 = other, 1 = small+channel, 2 = large
    region_id = np.zeros_like(phi, dtype=np.int8)
    region_id[mask_small] = 1
    region_id[mask_large] = 2

    # list of starting pixels in large region
    large_ys, large_xs = np.where(mask_large)
    n_large_pix = large_xs.size
    assert n_large_pix > 0

    crossed_count = 0

    for k in range(n_traj):
        # random starting position inside large region
        idx = rng.integers(0, n_large_pix)
        x0 = large_xs[idx] + rng.random()
        y0 = large_ys[idx] + rng.random()

        theta = 2 * np.pi * rng.random()
        vx = np.cos(theta)
        vy = np.sin(theta)

        x, y = x0, y0
        jx = int(np.clip(round(x), 0, Nx - 1))
        jy = int(np.clip(round(y), 0, Ny - 1))
        prev_region = region_id[jy, jx]

        crossed = False

        for _ in range(max_steps):
            x_new = x + vx * step_len
            y_new = y + vy * step_len
            phi_new = sample_phi(phi, x_new, y_new)

            # hit wall? reflect
            if phi_new <= 0.0:
                n = compute_normal(phi, x, y)
                v = np.array([vx, vy], dtype=float)
                v_ref = v - 2.0 * np.dot(v, n) * n
                vx, vy = v_ref[0], v_ref[1]
                # move a bit inside after reflection
                x = x + vx * (0.5 * step_len)
                y = y + vy * (0.5 * step_len)
                continue

            # free move accepted
            x, y = x_new, y_new

            jx = int(np.clip(round(x), 0, Nx - 1))
            jy = int(np.clip(round(y), 0, Ny - 1))
            region_now = region_id[jy, jx]

            if (not crossed) and (prev_region == 2) and (region_now == 1):
                crossed = True

            if region_now in (1, 2):
                prev_region = region_now

        if crossed:
            crossed_count += 1

    phi_est = crossed_count / n_traj
    print(f"[flux] crossed={crossed_count}/{n_traj} -> phi ≈ {phi_est:.4f}")
    return phi_est


# --------------------------- MAIN --------------------------- #

def main():
    cfg = CONFIG
    os.makedirs(cfg["OUT_DIR"], exist_ok=True)

    # geometry + region masks from labelled PNG
    phi, allowed, mask_small, mask_large, shape, (mu_small_cl, mu_large_cl) = \
        load_geometry_and_region_masks(cfg)

    # quantum data
    h5_path = cfg["H5_PATH"]
    hf, evals = load_eigenpairs(h5_path)
    total_states = evals.size
    print(f"[eig] total eigenvalues: {total_states}")

    start_index = int(cfg["START_INDEX"])
    window_size = int(cfg["WINDOW_SIZE"])
    num_windows = int(cfg["NUM_WINDOWS"])
    min_rel_frac = float(cfg["MIN_REL_FRAC"])

    max_idx = start_index + window_size * num_windows
    if max_idx > total_states:
        max_windows = (total_states - start_index) // window_size
        print(f"[warn] requested up to idx {max_idx}, but only {total_states} states.")
        print(f"[info] reducing NUM_WINDOWS -> {max_windows}")
        num_windows = max_windows

    # classical flux (single value for this geometry)
    phi_flux = estimate_flux(
        phi,
        allowed,
        mask_small,
        mask_large,
        n_traj=int(cfg["N_TRAJ"]),
        max_steps=int(cfg["MAX_STEPS"]),
        step_len=float(cfg["STEP_LEN"]),
        seed=int(cfg["RANDOM_SEED"]),
    )

    window_ranges = []
    w_means = []
    h_vals = []
    n_ch_vals = []
    x_vals = []

    Ny, Nx = shape

    for w in range(num_windows):
        idx_start = start_index + w * window_size
        print(f"\n[window] {w}: indices [{idx_start}, {idx_start + window_size})")

        w_mean, w_arr, kept_idxs = compute_w12_for_window(
            hf,
            allowed,
            mask_small,
            mask_large,
            mu_small_cl,
            mu_large_cl,
            shape,
            idx_start,
            window_size,
            min_rel_frac=min_rel_frac,
        )

        if np.isnan(w_mean):
            print("  [warn] no usable states in this window (after filtering).")
            continue

        N_ch = len(kept_idxs)
        if N_ch == 0:
            print("  [warn] N_ch = 0; skipping.")
            continue

        h_eff = 1.0 / N_ch  # area normalized to 1
        x_val = phi_flux / h_eff  # x = phi / h = phi * N_ch

        print(f"  kept {N_ch}/{window_size} states; <w_12> = {w_mean:.4f}, h = {h_eff:.4e}, x = {x_val:.4f}")

        window_ranges.append((idx_start, idx_start + window_size))
        w_means.append(w_mean)
        h_vals.append(h_eff)
        n_ch_vals.append(N_ch)
        x_vals.append(x_val)

    hf.close()

    if not w_means:
        print("[error] no data points produced. Check masks / thresholds.")
        return

    w_means = np.array(w_means, dtype=float)
    x_vals = np.array(x_vals, dtype=float)
    h_vals = np.array(h_vals, dtype=float)
    n_ch_vals = np.array(n_ch_vals, dtype=int)
    window_ranges = np.array(window_ranges, dtype=int)

    # Save data
    np.savez_compressed(
        cfg["DATA_OUT_NPZ"],
        x=x_vals,
        w12=w_means,
        h=h_vals,
        N_ch=n_ch_vals,
        windows=window_ranges,
        phi_flux=phi_flux,
    )
    print(f"[save] wrote data to {cfg['DATA_OUT_NPZ']}")

    # Plot <w_12> vs x with theory curve x/(1+x)
    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    xs = np.linspace(0.0, max(1.1 * x_max, 1e-3), 400)
    theory = xs / (1.0 + xs)

    plt.figure(figsize=(6, 4), dpi=140)
    plt.plot(xs, theory, "k--", label=r"$x/(1+x)$")
    plt.scatter(x_vals, w_means, s=40, alpha=0.8, label="data")
    plt.xlabel(r"$x = \phi / h$")
    plt.ylabel(r"$\langle w_{12} \rangle$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg["PLOT_PATH"], bbox_inches="tight")
    plt.close()
    print(f"[plot] saved {cfg['PLOT_PATH']}")

if __name__ == "__main__":
    start=time.time()
    main()
    end=time.time()
    print(f"Finished in {end-start} seconds")
