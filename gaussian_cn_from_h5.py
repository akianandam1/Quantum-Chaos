#!/usr/bin/env python3
"""
Gaussian wavepacket → preview → eigenbasis coverage → CN time-evolution → coverage again
+ region probabilities (small / channel / large) using SDF-based channel mask.

Requires: numpy, scipy, matplotlib, h5py
Assumes OUT_DIR contains: geometry_used.npz  (with 'phi' and 'allowed')
                           eigenpairs.h5     (datasets 'evals' and 'evecs')
"""

import os, math, time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, identity
from scipy.sparse.linalg import splu
from region_density_by_state import partition_manual_from_png  # uses your RGB rules  :contentReference[oaicite:0]{index=0}
import os


# =======================
# USER PARAMETERS
# =======================
OUT_DIR      = "eigensdf/doublewell/trial2"     # folder with eigenpairs.h5, geometry_used.npz
H5_NAME      = "eigenpairs.h5"
NPZ_PATH    = "potentials/potential8.npz"      # same geometry you solved
LABEL_PNG   = "potentials/labeled_potential8.png"  # your color-coded PNG

do_initial_coverage = True
do_final_coverage= True

# Wavepacket (pixel units)
X0, Y0       = 820.0, 515.0            # center of Gaussian (x right, y up in images)
K_MAG        = 10                     # momentum magnitude (in grid^-1)
THETA        = 2.5*3.14/2                      # direction angle (radians, 0→+x)
SIGMA_PIX    = 12.0                     # Gaussian width (std dev, pixels)

# Preview look
PROB_GAMMA   = 0.65                    # gamma boost for |psi|^2

# Overlap / coverage
E_MAX        = None                    # None → use ALL stored modes; or set a cutoff (e.g. 3.0)
CHUNK_COLS   = 512                     # stream size for eigenvector blocks
TOP_SHOW     = 24                      # print top-N |c|^2 components

# CN integrator
DT           = 0.70                    # time step
NSTEPS       = 40000                    # number of steps (total time = DT*NSTEPS)
REPORT_EVERY = 1000                    # progress print frequency
ALPHA_CLIP   = 5e-3                    # sub-cell Dirichlet weight lower bound

# ----- region label map (manual colors) -----
REGION_LABEL_PNG = "potentials/potential9_labelled.png"  # your color-coded PNG/JPG
# Exact RGB you used in the label image (0..255)
COLOR_SMALL   = (  255, 0,   0)   # green
COLOR_CHANNEL = (0,   255,   0)   # red
COLOR_LARGE   = (  0,   0, 255)   # blue
COLOR_TOL     = 8                 # tolerance per channel for matching (anti-aliasing/blur)
REQUIRE_SAME_SIZE = True          # require same WxH as geometry


# Region probabilities
# Channel = points whose (inside) distance to wall φ is below this threshold
CHANNEL_HALF_WIDTH = 6.0              # pixels; tune to match your narrow neck
SEED_SMALL  = (90, 470)                # (x,y)  seed in the small chamber
SEED_LARGE  = (360, 130)               # (x,y)  seed in the large chamber

# ----- quick preview video (sped-up) -----
QUICK_MP4        = True
QUICK_MP4_PATH   = os.path.join(OUT_DIR, "quick_preview.mp4")
QUICK_SAVE_EVERY = 80          # save a frame every N CN steps
QUICK_FPS        = 40          # video playback fps
QUICK_VMAX_AUTO  = True        # auto-scale colormap per-frame
QUICK_VMAX       = 0.02        # (used if QUICK_VMAX_AUTO=False)




# =======================
# Helpers: geometry pack/unpack, Laplacian
# =======================

import imageio.v2 as iio

from datetime import datetime as _dt
_start = _dt.now()
def tprint(*a, **k):
    now = _dt.now()
    delta = (now - _start).total_seconds()
    print(f"{now:%H:%M:%S}.{now.microsecond//1000:03d} (+{delta:8.3f}s)", *a, **k)

tprint("begin")

def _match_rgb(img_rgb, color, tol=0):
    """Return boolean mask where pixels are within tol of the given (R,G,B)."""
    r,g,b = color
    diff = np.abs(img_rgb[...,0].astype(np.int16) - r) \
         + np.abs(img_rgb[...,1].astype(np.int16) - g) \
         + np.abs(img_rgb[...,2].astype(np.int16) - b)
    # Use L1; with tol per-channel, total tol≈3*tol
    return diff <= (3*tol)

def load_region_masks_from_png(path, allowed, color_small, color_channel, color_large, tol=4,
                               require_same_size=True):
    """
    Read an RGB label image and return boolean masks (small, channel, large).
    Any pixel not matching one of the 3 colors becomes 'other' (ignored).
    Masks are ANDed with 'allowed' to stay inside domain.
    """
    img = iio.imread(path)  # (H,W,3) or (H,W,4) or (H,W)
    if img.ndim == 2:
        # grayscale -> replicate to RGB
        img = np.stack([img]*3, axis=-1)
    elif img.shape[2] == 4:
        # discard alpha for label matching
        img = img[..., :3]

    Ny, Nx = allowed.shape
    Ly, Lx = img.shape[:2]
    if require_same_size and (Lx != Nx or Ly != Ny):
        raise ValueError(f"[regions] label image size {Lx}x{Ly} != geometry {Nx}x{Ny}. "
                         f"Set REQUIRE_SAME_SIZE=False and resample beforehand if needed.")

    # If sizes differ but you allow it, you should resample to geometry size yourself
    # (keep pixels crisp). For now we assume sizes match.

    small_raw   = _match_rgb(img, color_small,   tol)
    channel_raw = _match_rgb(img, color_channel, tol)
    large_raw   = _match_rgb(img, color_large,   tol)

    # Force disjointness; if overlaps exist due to anti-aliased edges, resolve by priority
    # (channel > small > large) — change order if you prefer.
    channel = channel_raw.copy()
    small   = small_raw & (~channel)
    large   = large_raw & (~channel) & (~small)

    # Constrain to allowed domain
    small   = small   & allowed
    channel = channel & allowed
    large   = large   & allowed

    # Sanity: cover only allowed, disjoint
    assert np.all((small & channel) == 0)
    assert np.all((small & large)   == 0)
    assert np.all((large & channel) == 0)

    # Optional: warn if big portion of allowed is unlabeled
    unlabeled = allowed & (~small) & (~channel) & (~large)
    frac_unl = unlabeled.sum() / max(1, allowed.sum())
    if frac_unl > 0.02:
        print(f"[regions] warning: {frac_unl:.1%} of allowed is unlabeled in {path}")

    return small, channel, large

def render_prob_frame_rgb(psi_vec, unpack, allowed, vmax=None):
    """Return an RGB uint8 frame of |psi|^2 over the domain (fast)."""
    from matplotlib import cm
    psi2d = unpack(psi_vec)
    prob  = np.abs(psi2d)**2
    if vmax is None:
        vmax = np.percentile(prob[allowed], 99.5) + 1e-20
    x = np.clip(prob / vmax, 0, 1)**0.6  # gamma to boost dim regions
    rgb = (cm.plasma(x)[..., :3] * 255).astype(np.uint8)
    # draw a thin white boundary
    try:
        from scipy.ndimage import binary_erosion
        edge = allowed & (~binary_erosion(allowed))
        rgb[edge] = np.array([255, 255, 255], dtype=np.uint8)
    except Exception:
        pass
    return rgb


def preview_masks_from_labels(allowed, phi, m_small, m_chan, m_large,
                              title="Manual region labels (close to continue)"):
    Ny, Nx = allowed.shape
    base = np.zeros((Ny, Nx, 3), dtype=float)
    base[allowed] = 0.15  # dim gray
    overlay = base.copy()
    overlay[m_small]  = overlay[m_small]  * 0.2 + np.array([0.15, 0.95, 0.15]) * 0.8  # green
    overlay[m_chan]   = overlay[m_chan]   * 0.2 + np.array([0.95, 0.2,  0.2 ]) * 0.8  # red
    overlay[m_large]  = overlay[m_large]  * 0.2 + np.array([0.2,  0.45, 0.95]) * 0.8  # blue

    plt.figure(figsize=(7.8,7.2), dpi=140)
    plt.imshow(overlay, origin="lower")
    if phi is not None:
        edge = np.isclose(phi, 0.0, atol=0.8)
        ys, xs = np.where(edge)
        plt.scatter(xs, ys, s=0.5, c="white", alpha=0.9)
    plt.title(title); plt.axis("off"); plt.tight_layout(); plt.show()

class QuickWriter:
    """Tiny wrapper to write frames with imageio v2, avoiding macro-block warnings."""
    def __init__(self, path, fps):
        self.path = path
        self.fps  = fps
        self._writer = None
    def __enter__(self):
        # Some builds don’t accept macro_block_size; pass it only if supported
        try:
            self._writer = iio.get_writer(
                self.path, fps=self.fps, codec="libx264", macro_block_size=None
            )
        except TypeError:
            # Fallback: omit macro_block_size
            self._writer = iio.get_writer(self.path, fps=self.fps, codec="libx264")
        return self
    def append(self, frame_rgb):
        self._writer.append_data(frame_rgb)
    def __exit__(self, *exc):
        self._writer.close()


def load_geometry(out_dir):
    g = np.load(os.path.join(out_dir, "geometry_used.npz"))
    allowed = g["allowed"].astype(bool)
    phi = g["phi"].astype(np.float32)
    Ny, Nx = allowed.shape
    idx = -np.ones((Ny, Nx), np.int64)
    idx[allowed] = np.arange(allowed.sum())

    def pack(img2d):
        return img2d[allowed].reshape(-1)

    def unpack(vec):
        out = np.zeros((Ny, Nx), dtype=np.complex128)
        out[allowed] = vec.reshape(-1)
        return out

    return allowed, phi, (Ny, Nx), idx, pack, unpack

def build_dirichlet_sw(phi, allowed, alpha_clip=5e-3):
    """5-point Dirichlet Laplacian with sub-cell wall correction from signed distance φ."""
    Ny, Nx = allowed.shape
    idx = -np.ones((Ny, Nx), np.int64)
    idx[allowed] = np.arange(allowed.sum())
    n = int(allowed.sum())

    rows, cols, vals = [], [], []
    diag = np.zeros(n, np.float64)

    def add(a,b,v): rows.append(a); cols.append(b); vals.append(v)

    def link(a, y, x, yn, xn):
        if 0 <= yn < Ny and 0 <= xn < Nx and allowed[yn, xn]:
            b = idx[yn, xn]
            add(a, b, -1.0)
            diag[a] += 1.0
        else:
            # subcell correction using φ across the face
            phi_i = float(phi[y, x])
            phi_o = float(phi[yn, xn]) if (0 <= yn < Ny and 0 <= xn < Nx) else -phi_i
            denom = (phi_i - phi_o)
            alpha = 0.5 if denom <= 0 else max(alpha_clip, min(1.0, phi_i/denom))
            diag[a] += 1.0/alpha

    for y in range(Ny):
        for x in range(Nx):
            a = idx[y, x]
            if a < 0: continue
            link(a,y,x,y,x+1); link(a,y,x,y,x-1)
            link(a,y,x,y+1,x); link(a,y,x,y-1,x)

    rows += list(range(n)); cols += list(range(n)); vals += list(diag)
    H = coo_matrix((vals,(rows,cols)), shape=(n,n)).tocsr()
    return H

def region_probs_from_masks(psi_vec, unpack, allowed, m_small, m_chan, m_large):
    psi2 = np.abs(unpack(psi_vec))**2
    tot  = float(psi2[allowed].sum() + 1e-20)
    return (
        float(psi2[m_small].sum()/tot),
        float(psi2[m_chan].sum()/tot),
        float(psi2[m_large].sum()/tot),
    )

# =======================
# Wavepacket + preview
# =======================

def gaussian_packet(shape, x0, y0, kx, ky, sigma):
    Ny, Nx = shape
    x = np.arange(Nx)[None, :]
    y = np.arange(Ny)[:, None]
    r2 = (x - x0)**2 + (y - y0)**2
    env = np.exp(-0.5 * r2 / (sigma**2))
    phase = np.exp(1j*(kx*(x - x0) + ky*(y - y0)))
    psi = env * phase
    psi /= np.sqrt((np.abs(psi)**2).sum() + 1e-20)
    return psi.astype(np.complex128)

def preview_overlay(psi2d, allowed, title="Initial wavepacket (close to continue)"):
    prob = np.abs(psi2d)**2
    prob_g = prob**PROB_GAMMA
    Ny, Nx = allowed.shape
    plt.figure(figsize=(7.5,6), dpi=140)
    # walls=black, interior dim gray for contrast
    bg = np.zeros_like(allowed, dtype=float); bg[allowed] = 0.12
    plt.imshow(bg, origin="lower", cmap="gray", vmin=0, vmax=1)
    plt.imshow(prob_g, origin="lower", cmap="plasma", alpha=0.95)
    # outline boundary
    from scipy.ndimage import binary_erosion
    edge = allowed & (~binary_erosion(allowed))
    ys, xs = np.where(edge)
    plt.scatter(xs, ys, s=0.5, c="white", alpha=0.9)
    plt.title(title); plt.axis("off")
    plt.tight_layout(); plt.show()

# =======================
# Overlaps / coverage (streaming)
# =======================

def coverage_against_h5(h5_path, psi_vec, e_max=None, chunk_cols=512, top_show=24, label="[proj]"):
    with h5py.File(h5_path, "r") as f:
        E = f["evals"][:]
        if e_max is None:
            sel = np.arange(E.size)
        else:
            sel = np.where(E <= e_max)[0]
        K = sel.size
        c = np.empty(K, dtype=np.complex128)
        acc = 0.0
        t0 = time.time()
        for s in range(0, K, chunk_cols):
            print(s)
            e = min(s+chunk_cols, K)
            cols = sel[s:e]
            Vblk = f["evecs"][:, cols]        # (n, nb)
            Vblk /= (np.sqrt((np.abs(Vblk)**2).sum(axis=0)) + 1e-20)
            cblk = Vblk.conj().T @ psi_vec
            c[s:e] = cblk
            acc += float(np.sum(np.abs(cblk)**2))
        dt = time.time() - t0
    abs2 = np.abs(c)**2
    tprint(f"{label} modes used: {K}, coverage Σ|c|² = {acc:.6f}  (time {dt:.2f}s)")
    # list top components
    if top_show > 0 and K > 0:
        order = np.argsort(-abs2)[:min(top_show, K)]
        tprint("Top components:")
        for k in order:
            tprint(f"  E={E[sel[k]]:.6f}  |c|^2={abs2[k]:.3e}")
    return E[sel], c, abs2, acc

def split_chambers_seedless(allowed, channel):
    """
    Return (mask_small, mask_channel, mask_large) by labeling connected components
    of allowed \ channel and choosing the two largest as chambers.
    """
    from scipy.ndimage import label

    interior = allowed & (~channel)
    if interior.sum() == 0:
        raise ValueError("Channel mask covers the entire allowed region. Reduce CHANNEL_HALF_WIDTH.")

    lab, nlab = label(interior)
    if nlab < 2:
        raise ValueError("Could not find two chambers (only one connected component). "
                         "Reduce CHANNEL_HALF_WIDTH or check SDF polarity.")

    # areas of each component (exclude background label 0)
    areas = [(lab == i).sum() for i in range(1, nlab+1)]
    order = np.argsort(areas)[::-1]  # largest first
    big  = (lab == (order[0]+1))
    small = (lab == (order[1]+1))

    # decide which is "large" vs "small" by area:
    mask_large = big
    mask_small = small
    mask_chan  = channel.copy()

    # sanity: disjoint and cover allowed
    assert np.all((mask_small & mask_large) == 0)
    assert np.all((mask_small | mask_large | mask_chan) == allowed)
    return mask_small, mask_chan, mask_large

# =======================
# CN integrator
# =======================

def cn_factor(H, dt):
    n = H.shape[0]
    I = identity(n, format="csr", dtype=np.complex128)
    A = (I + 0.5j*dt*H).tocsr()
    B = (I - 0.5j*dt*H).tocsr()
    LU = splu(A.tocsc())
    return LU, B

def cn_evolve(H, psi0, dt, nsteps, report_every=1000, on_frame=None, save_every=None):
    """
    Crank–Nicolson; if on_frame is provided, we call on_frame(step, psi_vec)
    every 'save_every' steps (and also at step==1).
    """
    LU, B = cn_factor(H, dt)
    psi = psi0.copy()
    for t in range(1, nsteps+1):
        tprint(t)
        rhs = B @ psi
        psi = LU.solve(rhs)
        psi /= np.sqrt((np.abs(psi)**2).sum() + 1e-20)
        if on_frame and (t == 1 or (save_every and (t % save_every == 0))):
            on_frame(t, psi)
        if report_every and (t % report_every == 0):
            tprint(f"  [CN] step {t}/{nsteps}")
    return psi

# =======================
# Region masks (small / channel / large)
# =======================

def flood_region(allowed, exclude, seed_xy):
    Ny, Nx = allowed.shape
    sx, sy = seed_xy
    sx = np.clip(int(round(sx)), 0, Nx-1)
    sy = np.clip(int(round(sy)), 0, Ny-1)
    target = allowed & (~exclude)
    if not target[sy, sx]:
        raise ValueError(f"Seed {seed_xy} not in allowed∖exclude")
    reg = np.zeros_like(allowed, dtype=bool)
    from collections import deque
    q = deque([(sy, sx)])
    reg[sy, sx] = True
    while q:
        y, x = q.popleft()
        for yn, xn in ((y+1,x),(y-1,x),(y,x+1),(y,x-1)):
            if 0 <= yn < Ny and 0 <= xn < Nx and (not reg[yn, xn]) and target[yn, xn]:
                reg[yn, xn] = True
                q.append((yn, xn))
    return reg

def make_region_masks(allowed, phi, channel_half):
    """Channel = {allowed & (phi <= channel_half)} ; chambers are flood-filled outside channel from seeds."""
    channel = allowed & (phi > 0) & (phi <= channel_half)
    return channel

def region_probabilities(psi2d, allowed, channel, seed_small, seed_large):
    eps = 1e-20
    # flood-fill chambers excluding the channel
    small = flood_region(allowed, exclude=channel, seed_xy=seed_small)
    large = flood_region(allowed, exclude=channel, seed_xy=seed_large)
    # any leftover interior (rare) goes to large
    leftover = allowed & (~channel) & (~small) & (~large)
    large |= leftover
    # sanity: disjoint partitions of allowed
    assert np.all((small & large) == 0)
    assert np.all((small | large | channel) == allowed)

    prob = np.abs(psi2d)**2
    total = float(prob[allowed].sum() + eps)
    p_small   = float(prob[small].sum())   / total
    p_channel = float(prob[channel].sum()) / total
    p_large   = float(prob[large].sum())   / total
    return p_small, p_channel, p_large, small, channel, large

# =======================
# Main
# =======================

def main():
    # geometry
    allowed, phi, (Ny, Nx), idx, pack, unpack = load_geometry(OUT_DIR)
    # --- read manual masks from your color-labeled PNG (exactly like your working script) ---
    # This will raise if the label image size != phi.shape, which is good (avoids misalignment).
    small_mask, channel_mask, large_mask = partition_manual_from_png(
        LABEL_PNG, phi.shape, preview_path=os.path.join(OUT_DIR, "region_masks.png")
    )

    n = int(allowed.sum())
    tprint(f"[geom] grid {Nx}x{Ny}, DOFs n={n}")

    # Hamiltonian (Dirichlet)
    H = build_dirichlet_sw(phi, allowed, alpha_clip=ALPHA_CLIP).astype(np.complex128)

    # geometry
    allowed, phi, (Ny, Nx), idx, pack, unpack = load_geometry(OUT_DIR)
    print(f"[geom] grid {Nx}x{Ny}, DOFs n={allowed.sum()}")

    # # --- manual region masks from color-labeled PNG ---
    # m_small, m_chan, m_large = load_region_masks_from_png(
    #     REGION_LABEL_PNG, allowed,
    #     COLOR_SMALL, COLOR_CHANNEL, COLOR_LARGE, tol=COLOR_TOL,
    #     require_same_size=REQUIRE_SAME_SIZE
    # )
    #
    # # Preview them immediately
    # preview_masks_from_labels(allowed, phi, m_small, m_chan, m_large,
    #                           title=f"Manual labels preview: {os.path.basename(REGION_LABEL_PNG)}")

    # wavepacket
    kx = K_MAG * math.cos(THETA)
    ky = K_MAG * math.sin(THETA)
    psi2d = gaussian_packet((Ny, Nx), X0, Y0, kx, ky, SIGMA_PIX)
    psi2d *= allowed  # zero outside domain
    psi2d /= np.sqrt((np.abs(psi2d)**2).sum() + 1e-20)

    # preview
    preview_overlay(psi2d, allowed, title=f"Gaussian at (x0={X0:.1f},y0={Y0:.1f}), k=({kx:.2f},{ky:.2f}), σ={SIGMA_PIX}")

    # initial overlaps/coverage
    psi_vec = pack(psi2d)
    h5_path = os.path.join(OUT_DIR, H5_NAME)

    s0, c0, l0 = region_probs_from_masks(psi_vec, unpack, allowed,
                                         small_mask, channel_mask, large_mask)
    print(f"[regions @ t=0]   small={s0:.3f}  channel={c0:.3f}  large={l0:.3f}")

    if do_initial_coverage:
        E0, c0, abs20, cov0 = coverage_against_h5(h5_path, psi_vec, e_max=E_MAX, chunk_cols=CHUNK_COLS, top_show=TOP_SHOW, label="[proj@t0]")

    # evolve (with optional quick video)
    tprint(f"[CN] dt={DT}, steps={NSTEPS}, T={DT * NSTEPS:.3f}")

    writer = None
    if QUICK_MP4:
        writer = QuickWriter(QUICK_MP4_PATH, fps=QUICK_FPS).__enter__()
        tprint(f"[quick] writing {QUICK_MP4_PATH} (every {QUICK_SAVE_EVERY} steps, fps={QUICK_FPS})")

    def _on_frame(step, psi_vec_frame):
        if not QUICK_MP4:
            return
        vmax = None if QUICK_VMAX_AUTO else QUICK_VMAX
        frame = render_prob_frame_rgb(psi_vec_frame, unpack, allowed, vmax=vmax)
        writer.append(frame)

    t0 = time.time()
    psi_vec_T = cn_evolve(
        H, psi_vec, DT, NSTEPS,
        report_every=REPORT_EVERY,
        on_frame=_on_frame if QUICK_MP4 else None,
        save_every=QUICK_SAVE_EVERY if QUICK_MP4 else None
    )
    elapsed = time.time() - t0
    if QUICK_MP4:
        writer.__exit__(None, None, None)
        tprint(f"[quick] wrote {QUICK_MP4_PATH}")

    tprint(f"[CN] done in {elapsed:.2f}s")

    # coverage again
    if do_final_coverage:
        E1, c1, abs21, cov1 = coverage_against_h5(h5_path, psi_vec_T, e_max=E_MAX, chunk_cols=CHUNK_COLS, top_show=TOP_SHOW, label="[proj@T ]")

    # region probabilities at t=0 and t=T
    s0, c0, l0 = region_probs_from_masks(psi_vec, unpack, allowed,
                                         small_mask, channel_mask, large_mask)

    sT, cT, lT = region_probs_from_masks(psi_vec_T, unpack, allowed,
                                         small_mask, channel_mask, large_mask)
    # print(f"[regions @ t=T]   small={sT:.3f}  channel={cT:.3f}  large={lT:.3f}")

    # def probs_from_masks(psi2d, allowed, m_small, m_chan, m_large):
    #     prob = np.abs(psi2d) ** 2
    #     tot = float(prob[allowed].sum() + 1e-20)
    #     return (float(prob[m_small].sum()) / tot,
    #             float(prob[m_chan].sum()) / tot,
    #             float(prob[m_large].sum()) / tot)
    #
    # small0, chan0, large0 = probs_from_masks(psi2d, allowed, m_small, m_chan, m_large)
    # psi2d_T = unpack(psi_vec_T)
    # smallT, chanT, largeT = probs_from_masks(psi2d_T, allowed, m_small, m_chan, m_large)
    print("\n[regions @ t=0]   small={:.3f}  channel={:.3f}  large={:.3f}".format(s0, c0, l0))
    print("[regions @ t=T]   small={:.3f}  channel={:.3f}  large={:.3f}".format(sT, cT, lT))

    # channel_mask = make_region_masks(allowed, phi, CHANNEL_HALF_WIDTH)
    # m_small, m_chan, m_large = split_chambers_seedless(allowed, channel_mask)
    #
    # def probs_for(psi2d):
    #     prob = np.abs(psi2d) ** 2
    #     tot = float(prob[allowed].sum() + 1e-20)
    #     return (float(prob[m_small].sum()) / tot,
    #             float(prob[m_chan].sum()) / tot,
    #             float(prob[m_large].sum()) / tot)
    #
    # small0, chan0, large0 = probs_for(psi2d)
    # psi2d_T = unpack(psi_vec_T)
    # smallT, chanT, largeT = probs_for(psi2d_T)


    # psi2d_T = unpack(psi_vec_T)
    # smallT, chanT, largeT, *_ = region_probabilities(psi2d_T, allowed, channel_mask, SEED_SMALL, SEED_LARGE)
    #
    # tprint("\n[regions @ t=0]   small={:.3f}  channel={:.3f}  large={:.3f}".format(small0, chan0, large0))
    # tprint(  "[regions @ t=T]   small={:.3f}  channel={:.3f}  large={:.3f}".format(smallT, chanT, largeT))

    # # quick visualization of masks (one-time preview, not saved)
    # fig, ax = plt.subplots(1,3, figsize=(11,4), dpi=130)
    # ax[0].imshow(m_small, origin="lower", cmap="Greens"); ax[0].set_title("small chamber mask"); ax[0].axis("off")
    # ax[1].imshow(m_chan,  origin="lower", cmap="Reds");   ax[1].set_title("channel mask");       ax[1].axis("off")
    # ax[2].imshow(m_large, origin="lower", cmap="Blues");  ax[2].set_title("large chamber mask"); ax[2].axis("off")
    # fig.tight_layout(); plt.show()

    # expectation <H> (sanity)
    Hpsi0 = H @ psi_vec
    HpsiT = H @ psi_vec_T
    tprint(f"[energy] <H> t=0 ≈ {float(np.vdot(psi_vec,   Hpsi0).real):.6f}")
    tprint(f"[energy] <H> t=T ≈ {float(np.vdot(psi_vec_T, HpsiT).real):.6f}")

if __name__ == "__main__":
    start=time.time()
    main()
    end=time.time()
    tprint(f"Finished in {end-start} seconds")
