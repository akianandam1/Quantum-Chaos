#!/usr/bin/env python3
"""
channel_modes_energy_from_E.py

For each eigenstate:
  - Compute w_12 (small+channel vs large, normalized by classical area fractions).
  - Compute x_j from energy only:
        x_j = N_perp_max(E_j) = floor(W_eff * sqrt(E_j) / pi),
    where W_eff is a user-specified effective channel width in grid units.

Processes states in windows and reports:
  - average x per window
  - average <w_12> per window

Also prints every PRINT_EVERY-th eigenstate index, x_j, and w_12(j) for sanity.

Relies on:
  - SDF npz: with phi, allowed
  - RGB-labelled PNG: red=small, green=channel, blue=large
  - eigenpairs.h5: evals[K], evecs[n,K] from lowband_complete_solver.py
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

# ===================== USER CONFIG =====================

CONFIG = {
    # SDF and labels
    "NPZ_PATH": r"potentials/potential12.npz",  # SDF npz (phi, allowed)
    "LABEL_PNG": r"potentials/potential12_labelled.png",  # RGB-labelled regions
    "GEOM_PREVIEW_PNG": r"eigensdf/doublewell/roland_narrow1/region_masks_preview.png",

    # Eigenpairs
    "H5_PATH": r"eigensdf/doublewell/roland_narrow1/eigenpairs_sorted.h5",
    "OUT_DIR": r"eigensdf/doublewell/roland_narrow1",

    # Eigenstate processing
    "START_INDEX": 3500,    # first eigenstate index (0-based) to consider
    "WINDOW_SIZE": 100,      # states per window
    "NUM_WINDOWS": 1,      # number of windows
    "MIN_REL_FRAC": 0.0,    # optional crude chaoticity filter (0 to disable)

    # Effective channel width (in grid units / pixels)
    # Measure this by hand (e.g. count allowed pixels across the entrance in your PNG). #potential 8 was 67 pixels, potential 12 was 16 pixels
    "W_EFF": 16.0,

    # Printing
    "PRINT_EVERY": 10,     # print x,w12 for every N-th state (global index)

    # Output
    "PLOT_PATH": r"eigensdf/doublewell/roland_narrow1/partial_barrier_curve.png",
    "DATA_OUT_NPZ": r"eigensdf/doublewell/roland_narrow1/partial_barrier_data.npz",
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
    os.makedirs(os.path.dirname(preview_path), exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=130)
    ax[0].imshow(small_mask, origin="lower", cmap="gray")
    ax[0].set_title("Small")
    ax[0].axis("off")
    ax[1].imshow(channel_mask, origin="lower", cmap="gray")
    ax[1].set_title("Channel")
    ax[1].axis("off")
    ax[2].imshow(large_mask, origin="lower", cmap="gray")
    ax[2].set_title("Large")
    ax[2].axis("off")
    fig.tight_layout()
    fig.savefig(preview_path, bbox_inches="tight")
    plt.close(fig)

def partition_manual_from_png(label_png, phi_shape, preview_path=None):
    """Use RGB-labelled PNG: red->small, green->channel, blue->large."""
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

def load_geometry_and_region_masks(cfg):
    phi, allowed = load_npz(cfg["NPZ_PATH"])
    Ny, Nx = phi.shape

    small_mask, channel_mask, large_mask = partition_manual_from_png(
        cfg["LABEL_PNG"], phi.shape, cfg["GEOM_PREVIEW_PNG"]
    )

    small_mask &= allowed
    channel_mask &= allowed
    large_mask &= allowed

    if not (small_mask | channel_mask).any():
        raise RuntimeError("small+channel region is empty; check your labelled PNG.")
    if not large_mask.any():
        raise RuntimeError("large_mask is empty; check your labelled PNG.")

    # Combine small+channel into region 1
    small_combined = (small_mask | channel_mask) & allowed

    # Classical area fractions over (small+channel + large)
    union_mask = (small_combined | large_mask) & allowed
    total_area = union_mask.sum()
    mu_small_cl = small_combined.sum() / total_area
    mu_large_cl = large_mask.sum() / total_area

    print(f"[geom] phi shape={phi.shape}, allowed cells={allowed.sum()}")
    print(f"[geom] small+channel area={small_combined.sum()}, large area={large_mask.sum()}, total={total_area}")
    print(f"[geom] mu_small_cl={mu_small_cl:.4f}, mu_large_cl={mu_large_cl:.4f}")

    return phi, allowed, small_combined, large_mask, (Ny, Nx), (mu_small_cl, mu_large_cl)


# -------------------- EIGENSTATES / w12 + x(E) -------------------- #

def load_eigenpairs(h5_path):
    f = h5py.File(h5_path, "r")
    evals = np.array(f["evals"][:], dtype=np.float64)
    return f, evals

def unpack_evec(vec, allowed_mask, shape):
    Ny, Nx = shape
    psi = np.zeros((Ny, Nx), dtype=np.complex128)
    psi[allowed_mask] = vec.reshape(-1)
    return psi

def process_window_energy_x_w12(hf,
                                evals,
                                allowed,
                                mask_small_combined,
                                mask_large,
                                mu_small_cl,
                                mu_large_cl,
                                shape,
                                idx_start,
                                window_size,
                                W_eff,
                                min_rel_frac=0.0,
                                print_every=200,
                                global_offset=0):
    """
    For eigenstates [idx_start, idx_start+window_size), compute:

      - w_12 for each state
      - x_j = N_perp_max(E_j) = floor(W_eff * sqrt(E_j) / pi)

    Returns:
      - arrays of energies, w12, x, and the indices actually kept
    """
    idx_end = min(idx_start + window_size, evals.size)
    evecs = np.array(hf["evecs"][:, idx_start:idx_end], dtype=np.complex128)
    K = evecs.shape[1]

    Ny, Nx = shape
    w_list = []
    x_list = []
    e_list = []
    idx_list = []

    for j in range(K):
        global_idx = idx_start + j
        E_j = evals[global_idx]
        if E_j <= 0:
            # skip pathological or non-physical E
            continue

        v = evecs[:, j]
        psi = unpack_evec(v, allowed, shape)
        dens = np.abs(psi) ** 2
        tot = dens[allowed].sum()
        if not np.isfinite(tot) or tot <= 0:
            continue
        dens /= tot

        p_small = dens[mask_small_combined].sum()
        p_large = dens[mask_large].sum()

        # optional crude chaoticity filter
        if min_rel_frac > 0.0:
            if (p_small < min_rel_frac * mu_small_cl) or (p_large < min_rel_frac * mu_large_cl):
                continue

        r1 = p_small / mu_small_cl
        r2 = p_large / mu_large_cl
        w12 = r1 * r2

        # x_j = N_perp_max(E_j) = floor(W_eff * sqrt(E_j) / pi)
        k_j = np.sqrt(E_j)   # assuming -∇^2 ψ = E ψ
        x_j = W_eff * k_j / np.pi
        if x_j < 0:
            x_j = 0

        if print_every > 0 and ((global_idx - global_offset) % print_every == 0):
            print(f"  state {global_idx}: x(E)={x_j:.3f}, w12={w12:.3f}, E={E_j:.3f}")

        w_list.append(w12)
        x_list.append(x_j)
        e_list.append(E_j)
        idx_list.append(global_idx)

    if not w_list:
        return (np.array([]), np.array([]), np.array([]), np.array([], dtype=int))

    return (np.array(e_list, dtype=float),
            np.array(w_list, dtype=float),
            np.array(x_list, dtype=float),
            np.array(idx_list, dtype=int))


# --------------------------- MAIN --------------------------- #

def main():
    cfg = CONFIG
    os.makedirs(cfg["OUT_DIR"], exist_ok=True)

    # geometry and masks
    (phi,
     allowed,
     mask_small_combined,
     mask_large,
     shape,
     (mu_small_cl, mu_large_cl)) = load_geometry_and_region_masks(cfg)

    # effective width set by user
    W_eff = float(cfg["W_EFF"])
    print(f"[width] Using user-specified W_eff = {W_eff} (grid units)")

    # eigenpairs
    hf, evals = load_eigenpairs(cfg["H5_PATH"])
    total_states = evals.size
    print(f"[eig] total eigenvalues: {total_states}")

    start_index = int(cfg["START_INDEX"])
    window_size = int(cfg["WINDOW_SIZE"])
    num_windows = int(cfg["NUM_WINDOWS"])
    min_rel_frac = float(cfg["MIN_REL_FRAC"])
    print_every = int(cfg["PRINT_EVERY"])

    max_idx = start_index + window_size * num_windows
    if max_idx > total_states:
        max_windows = (total_states - start_index) // window_size
        print(f"[warn] requested up to idx {max_idx}, but only {total_states} states.")
        print(f"[info] reducing NUM_WINDOWS -> {max_windows}")
        num_windows = max_windows

    all_E = []
    all_w12 = []
    all_x = []
    all_indices = []
    window_stats = []

    global_offset = start_index

    for w in range(num_windows):
        idx_start = start_index + w * window_size
        print(f"\n[window] {w}: indices [{idx_start}, {idx_start + window_size})")

        E_arr, w12_arr, x_arr, idx_arr = process_window_energy_x_w12(
            hf,
            evals,
            allowed,
            mask_small_combined,
            mask_large,
            mu_small_cl,
            mu_large_cl,
            shape,
            idx_start,
            window_size,
            W_eff,
            min_rel_frac=min_rel_frac,
            print_every=print_every,
            global_offset=global_offset,
        )

        if E_arr.size == 0:
            print("  [warn] no usable states in this window.")
            continue

        mean_w = float(w12_arr.mean())
        mean_x = float(x_arr.mean())
        print(f"  window mean: x(E)={mean_x:.3f}, <w_12>={mean_w:.3f}")

        window_stats.append((idx_start, idx_start + window_size, mean_x, mean_w))

        all_E.append(E_arr)
        all_w12.append(w12_arr)
        all_x.append(x_arr)
        all_indices.append(idx_arr)

    hf.close()

    if not all_w12:
        print("[error] no data produced; check masks/indices.")
        return

    all_E = np.concatenate(all_E)
    all_w12 = np.concatenate(all_w12)
    all_x = np.concatenate(all_x)
    all_indices = np.concatenate(all_indices)
    window_stats = np.array(window_stats, dtype=float)

    # Overall averages
    overall_mean_x = float(all_x.mean())
    overall_mean_w = float(all_w12.mean())
    print(f"\n[overall] mean x(E)={overall_mean_x:.3f}, overall <w_12>={overall_mean_w:.3f}")

    # Save data
    np.savez_compressed(
        cfg["DATA_OUT_NPZ"],
        E=all_E,
        w12=all_w12,
        x=all_x,
        indices=all_indices,
        window_stats=window_stats,
        W_eff=W_eff,
    )
    print(f"[save] wrote data to {cfg['DATA_OUT_NPZ']}")

    # Scatter plot: w_12 vs x(E)
    plt.figure(figsize=(6, 4), dpi=140)
    plt.scatter(all_x, all_w12, s=20, alpha=0.6)
    plt.xlabel(r"$x(E) = N_{\perp,\max}(E)$")
    plt.ylabel(r"$w_{12}$")
    plt.tight_layout()
    plt.savefig(cfg["PLOT_PATH"], bbox_inches="tight")
    plt.close()
    print(f"[plot] saved {cfg['PLOT_PATH']}")

if __name__ == "__main__":
    main()
