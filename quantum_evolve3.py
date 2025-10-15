"""
quantum_nodes_or_packet.py

White = allowed (V=0), black = hard wall (Dirichlet).
Initial state options:
  - "gaussian": Gaussian at (x0,y0) with momentum (kx,ky)
  - "standing": Standing wave with given *node counts* (nx, ny) in a rectangular ROI,
                optionally "launched" by multiplying a plane-wave e^{i(kx(x-xc)+ky(y-yc))}.
    NOTE: node counts = number of internal nodes; mode index = nodes+1.

Outputs:
  - MP4 of |psi|^2 (gamma-enhanced to show faint structure)
  - MP4 of Re(psi) (asinh scaling)
  - CSV of autocorrelation C(t) and |C|^2 at saved frames
  - CSV of FFT[ C(t) ] (energy spectrum; ħ=1 so angular frequency ω ≈ energy)

Crank–Nicolson with a single LU factorization; direct MP4 writing (no PNGs).
"""

import os, csv, math
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.linalg import splu
import imageio.v2 as imageio
from matplotlib import cm
import matplotlib.pyplot as plt
import time

# ==========================
# ===== USER SETTINGS ======
# ==========================

# Potential image (white = allowed, black = wall)
potential_number = "potential5"
trial_number     = "8"
IMG_PATH         = f"potentials/{potential_number}.png"   # White=allowed, Black=wall

# Grid / evolution
NGRID      = 380          # 224–384 reasonable
DT         = 0.09         # increase to cover more simulated time with same steps
NUM_STEPS  = 30000
SAVE_EVERY = 40
FPS        = 40

# Precision & performance
USE_COMPLEX64 = False     # False => complex128 (more precise; recommended for long runs)

# Visualization
GAMMA_PROB    = 0.55      # gamma for |psi|^2 (smaller → brighter tails)
REAL_SCALE    = "asinh"   # "asinh" or "linear"
ALPHA_OVERLAY = 0.65      # blend sim with background mask

# === Initial state selector ===
INIT_MODE = "standing"    # "gaussian" or "standing"

# --- (1) Gaussian parameters ---
GAUSS_X0    = 40       # if None → auto-place near bottom mouth
GAUSS_Y0    = 322
GAUSS_SIGMA = 4.0         # pixel units
GAUSS_KX    = 10.0         # momentum components (pixels^-1)
GAUSS_KY    = -1        # negative KY launches upward on image

#PREVIOUS WAS 420 PIXELS, STAND_X0, STAND_X1 = 45, 75
#STAND_Y0, STAND_Y1 = 315, 330

# --- (2) Standing-wave parameters (nodes, not mode indices) ---
# Define rectangular ROI (x0:x1, y0:y1) in pixels. The standing wave lives inside it.
STAND_X0, STAND_X1 = 40, 60
STAND_Y0, STAND_Y1 = 288, 300
STAND_NX, STAND_NY = 0, 5       # *node counts* (0 = fundamental)
STAND_TAPER_SIGMA  = 4.0        # optional Gaussian taper to smooth edges (pixels); None/0 disables
# Optional launch (multiply by plane-wave phase to give momentum)
STAND_LAUNCH_KX    = 0
STAND_LAUNCH_KY    = 10
STAND_LAUNCH_REFPT = "center"   # where to reference phase: "center" or (x_ref, y_ref)


# --- Mask quality knobs (add in USER SETTINGS) ---
MASK_SUPERSAMPLE = 10     # was 3–4; higher = cleaner edges
MASK_BLUR_SIGMA  = 1   # 0.4–1.0 smooths stairsteps before threshold
MASK_MIN_ISLAND  = 8     # drop tiny white flecks, but keep thin features
MASK_MIN_HOLE    = 8     # fill tiny black pinholes
MASK_KEEP_LARGEST = True # set False if you have multiple disjoint allowed regions

# Output folder
OUT_DIR       = f"videos/{potential_number}/trial{trial_number}"
os.makedirs(OUT_DIR,exist_ok=True)
# FFT window for spectrum
APPLY_HANN_WINDOW = True

with open(f"videos/{potential_number}/trial{trial_number}/parameters.txt","w") as file:
    file.write(f"NGRID={NGRID},STAND_LAUNCH_KX={STAND_LAUNCH_KX},STAND_LAUNCH_KY={STAND_LAUNCH_KY},STAND_X0={STAND_X0},STAND_X1={STAND_X1},STAND_Y0={STAND_Y0},STAND_Y1={STAND_Y1},STAND_NX={STAND_NX},STAND_NY={STAND_NY}")

# ==========================
# ===== HELPER ROUTINES ====
# ==========================

def to_dtype(z):
    return z.astype(np.complex64) if USE_COMPLEX64 else z.astype(np.complex128)

def load_allowed_mask(path, target_n, supersample=MASK_SUPERSAMPLE,
    blur_sigma=MASK_BLUR_SIGMA,
    min_island=MASK_MIN_ISLAND,
    min_hole=MASK_MIN_HOLE,
    keep_largest=MASK_KEEP_LARGEST):
    """Robust B/W mask: supersample + Otsu + clean small components + downsample."""
    im = Image.open(path).convert("L")
    hiN = target_n * max(1, int(supersample))
    arr = np.asarray(im.resize((hiN, hiN), Image.LANCZOS), dtype=np.float32)/255.0
    if blur_sigma and blur_sigma > 0:
        arr = ndi.gaussian_filter(arr, blur_sigma)
    # Otsu threshold
    hist, bins = np.histogram(arr, bins=256, range=(0,1))
    p = hist.astype(np.float64)/max(1, hist.sum())
    w = np.cumsum(p); mu = np.cumsum(p*np.arange(256)); mu_t = mu[-1]
    sigma_b2 = (mu_t*w - mu)**2 / (w*(1-w) + 1e-12)
    thr = bins[np.nanargmax(sigma_b2)]
    allowed_hr = arr >= thr
    frac = allowed_hr.mean()
    if frac < 0.05 or frac > 0.95:  # safety for inverted images
        allowed_hr = ~allowed_hr
    # remove tiny islands / fill tiny holes
    if min_island>0:
        lab, n = ndi.label(allowed_hr)
        sizes = ndi.sum(allowed_hr, lab, index=np.arange(1, n+1))
        kill = np.where(sizes < min_island)[0] + 1
        if kill.size:
            allowed_hr[np.isin(lab, kill)] = False
    if min_hole>0:
        holes = ~allowed_hr
        lab, n = ndi.label(holes)
        sizes = ndi.sum(holes, lab, index=np.arange(1, n+1))
        fill = np.where(sizes < min_hole)[0] + 1
        if fill.size:
            holes[np.isin(lab, fill)] = False
        allowed_hr = ~holes
    # downsample block-mean → majority
    s = supersample
    H = hiN//s*s
    allowed_hr = allowed_hr[:H, :H]
    allowed = (allowed_hr.reshape(H//s, s, H//s, s).mean(axis=(1,3)) >= 0.5)
    if keep_largest:
        lab, n = ndi.label(allowed)
        if n > 1:
            sizes = ndi.sum(allowed, lab, index=np.arange(1, n+1))
            keep = 1 + int(np.argmax(sizes))
            allowed = (lab == keep)
    allowed = ndi.binary_opening(allowed, iterations=1)
    allowed = ndi.binary_closing(allowed, iterations=1)

    # (Optional) visualize ROI corners
    plt.imshow(allowed, origin='lower', cmap='gray')
    for (xx, yy) in [(STAND_X0, STAND_Y0), (STAND_X0, STAND_Y1), (STAND_X1, STAND_Y0), (STAND_X1, STAND_Y1)]:
        plt.scatter([xx], [yy], c='r', s=10)
    plt.show()

    return allowed.astype(bool)

def build_laplacian(allowed):
    """5-point Dirichlet Laplacian on allowed set; H=-∇² on packed nodes."""
    Ny, Nx = allowed.shape
    idx = -np.ones((Ny, Nx), np.int64)
    idx[allowed] = np.arange(allowed.sum())
    n = int(allowed.sum())
    rows, cols = [], []
    for dy, dx in [(0,1),(1,0)]:
        A = allowed & np.roll(allowed, shift=(-dy,-dx), axis=(0,1))
        if dy== 1: A[0 ,:]=False
        if dy==-1: A[-1,:]=False
        if dx== 1: A[:,0 ]=False
        if dx==-1: A[:,-1]=False
        r = idx[A]; c = np.roll(idx, shift=(-dy,-dx), axis=(0,1))[A]
        rows += [r, c]; cols += [c, r]
    rows = np.concatenate(rows) if rows else np.array([], np.int64)
    cols = np.concatenate(cols) if cols else np.array([], np.int64)
    L = coo_matrix((-np.ones_like(rows,float),(rows,cols)),shape=(n,n)).tocsr()
    H = (L + diags([4.0],[0],shape=(n,n))).tocsr()
    return H, idx, n

def auto_place_bottom(allowed):
    """Pick a point near the bottom mouth (median of lowest band)."""
    Ny, Nx = allowed.shape
    ys, xs = np.where(allowed)
    if ys.size == 0: return Nx//2, Ny//2
    ymax = ys.max()
    band = ys >= (ymax - 4)
    if not band.any(): return int(np.median(xs)), int(np.median(ys))
    x0 = int(np.median(xs[band])); y0 = int(np.median(ys[band])) - 2
    return max(2,min(Nx-3,x0)), max(2,min(Ny-3,y0))

def make_gaussian(allowed, x0, y0, sigma, kx, ky):
    Ny, Nx = allowed.shape
    Y, X = np.mgrid[0:Ny, 0:Nx]
    psi = np.exp(-((X-x0)**2 + (Y-y0)**2)/(2.0*sigma**2)).astype(np.complex128)
    psi *= np.exp(1j*(kx*(X-x0) + ky*(Y-y0)))
    psi *= allowed
    return psi

def make_standing_with_nodes(allowed, x0, x1, y0, y1, nodes_x, nodes_y,
                             taper_sigma=None, launch_kx=0.0, launch_ky=0.0,
                             ref_point="center"):
    """
    Standing wave with given *node counts* inside ROI, masked by geometry.
    Optionally multiply by a plane-wave phase to 'launch' it with momentum.
    """
    Ny, Nx = allowed.shape
    Y, X = np.mgrid[0:Ny, 0:Nx]
    x0c, x1c = max(0,x0), min(Nx,x1)
    y0c, y1c = max(0,y0), min(Ny,y1)
    psi = np.zeros((Ny, Nx), dtype=np.complex128)
    if x1c <= x0c+1 or y1c <= y0c+1:
        return psi

    # Convert node counts → mode indices (Dirichlet)
    mx = max(1, int(nodes_x) + 1)
    my = max(1, int(nodes_y) + 1)

    # Local [0,1] coordinates in ROI
    u = (X - x0c) / max(1, (x1c - x0c))
    v = (Y - y0c) / max(1, (y1c - y0c))
    core = np.sin(mx*np.pi*u) * np.sin(my*np.pi*v)

    # Restrict to ROI
    in_roi = (X >= x0c) & (X < x1c) & (Y >= y0c) & (Y < y1c)
    psi[in_roi] = core[in_roi]

    # Optional taper to localize better within ROI
    if taper_sigma and taper_sigma > 0:
        cx = (x0c + x1c) / 2.0
        cy = (y0c + y1c) / 2.0
        taper = np.exp(-(((X-cx)**2 + (Y-cy)**2) / (2.0*taper_sigma**2)))
        psi *= taper

    # Optional "launch": multiply by plane wave around a reference point
    if isinstance(ref_point, str) and ref_point.lower()=="center":
        xr = (x0c + x1c) / 2.0
        yr = (y0c + y1c) / 2.0
    else:
        xr, yr = ref_point
    if (launch_kx != 0.0) or (launch_ky != 0.0):
        psi *= np.exp(1j*(launch_kx*(X-xr) + launch_ky*(Y-yr)))

    psi *= allowed
    return psi

def k_from_energy_and_angle(E, theta_deg):
    """Given energy E = |k|^2 (ħ=1) and angle, return (kx, ky)."""
    kmag = math.sqrt(max(0.0, E))
    th = math.radians(theta_deg)
    # y increases upward; theta=90° -> upward (ky negative to go up in image coords)
    return kmag*math.cos(th), -kmag*math.sin(th)

def embed_vec_to_img(vec, allowed, dtype):
    img = np.zeros(allowed.shape, dtype=dtype)
    img[allowed] = vec
    return img

def colormap_prob(prob):
    """Map probability to the same blue–white–red palette as Re(psi)."""
    prob = np.clip(prob, 0, 1)
    x = prob**GAMMA_PROB              # same gamma boost you had
    return (cm.RdBu_r(x)[:, :, :3]*255).astype(np.uint8)

def colormap_real(re_img):
    if REAL_SCALE.lower() == "asinh":
        s = np.max(np.abs(re_img)) + 1e-12
        x = np.arcsinh(re_img / (0.05*s))
        x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    else:
        s = np.max(np.abs(re_img)) + 1e-12
        x = 0.5*(re_img/s + 1.0)
    return (cm.RdBu_r(x)[:, :, :3]*255).astype(np.uint8)

# ==========================
# ========= RUNNER =========
# ==========================

def evolve_and_save(allowed, H, psi0_img, out_dir, tag="run"):
    Ny, Nx = allowed.shape
    n = int(allowed.sum())

    Hc = to_dtype(H)
    I  = identity(n, dtype=Hc.dtype, format="csc")
    A  = (I + (0.5j*DT)*Hc).tocsc()
    B  = (I - (0.5j*DT)*Hc).tocsr()
    solver = splu(A, permc_spec="COLAMD", diag_pivot_thresh=0.0)
    psolve = solver.solve

    psi_vec = psi0_img[allowed]
    norm0 = np.linalg.norm(psi_vec)
    if norm0 == 0:
        raise RuntimeError("Initial wavefunction is zero. Adjust placement/ROI or parameters.")
    psi_vec = to_dtype(psi_vec / (norm0 + 1e-15))
    psi_initial = psi_vec.copy()

    os.makedirs(out_dir, exist_ok=True)
    mp4_prob = os.path.join(out_dir, f"{tag}_prob.mp4")
    mp4_real = os.path.join(out_dir, f"{tag}_real.mp4")
    csv_corr = os.path.join(out_dir, f"{tag}_autocorr.csv")
    csv_spec = os.path.join(out_dir, f"{tag}_spectrum.csv")

    # background from mask
    bg = np.zeros((Ny, Nx), np.float32); bg[allowed] = 1.0
    bg_rgb = (255*np.repeat(bg[:, :, None], 3, axis=2)).astype(np.uint8)

    writer_prob = imageio.get_writer(mp4_prob, fps=FPS, codec="libx264",
                                     quality=8, macro_block_size=1)
    writer_real = imageio.get_writer(mp4_real, fps=FPS, codec="libx264",
                                     quality=8, macro_block_size=1)

    frame_dt = DT*SAVE_EVERY
    C_list = []  # complex C(t)

    with open(csv_corr, "w", newline="") as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow(["t", "Re<C>", "Im<C>", "|C|", "|C|^2"])

        for frame_idx, t0 in enumerate(range(0, NUM_STEPS+1, SAVE_EVERY)):
            # render
            psi_img = embed_vec_to_img(psi_vec, allowed, psi_vec.dtype)
            prob = (np.abs(psi_img)**2).astype(np.float32)
            prob /= prob.max() + 1e-12

            prob_rgb = colormap_prob(prob)
            real_rgb = colormap_real(psi_img.real.astype(np.float32))
            frame_prob = (ALPHA_OVERLAY*prob_rgb + (1-ALPHA_OVERLAY)*bg_rgb).astype(np.uint8)
            frame_real = (ALPHA_OVERLAY*real_rgb + (1-ALPHA_OVERLAY)*bg_rgb).astype(np.uint8)
            writer_prob.append_data(frame_prob)
            writer_real.append_data(frame_real)

            # autocorrelation at this saved frame (tiny renorm to kill drift)
            psi_vec /= np.sqrt((np.abs(psi_vec)**2).sum() + 1e-15)
            c = np.vdot(psi_initial, psi_vec)
            C_list.append(c)

            t = frame_idx*frame_dt
            wcsv.writerow([f"{t:.9g}", f"{c.real:.9g}", f"{c.imag:.9g}",
                           f"{abs(c):.9g}", f"{(abs(c)**2):.9g}"])

            # advance
            steps = min(SAVE_EVERY, NUM_STEPS - t0)
            for _ in range(steps):
                rhs = B.dot(psi_vec)
                psi_vec = to_dtype(psolve(rhs))

    writer_prob.close(); writer_real.close()

    # --- FIX: spectrum from complex autocorrelation C(t) ---

    # C_list is a list of complex numbers sampled once per saved frame
    C = np.asarray(C_list, dtype=np.complex128)  # ensure proper complex dtype

    # Optional Hann window to reduce leakage
    Cw = C * np.hanning(C.size) if APPLY_HANN_WINDOW else C

    # Use complex FFT (rfft is for real-only input)
    S = np.fft.fft(Cw)
    freqs = np.fft.fftfreq(Cw.size, d=frame_dt)  # cycles per unit time
    omega = 2.0 * np.pi * freqs  # angular frequency; ħ=1 ⇒ ω ≈ energy

    # Keep non-negative frequencies (DC + positive)
    pos = freqs >= 0
    omega_pos = omega[pos]
    S_mag_pos = np.abs(S[pos])

    # Normalize magnitude for plotting/export (optional)
    if S_mag_pos.max() > 0:
        S_mag_pos = S_mag_pos / S_mag_pos.max()

    # Save spectrum CSV
    with open(csv_spec, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["omega", "|FT{C}|_norm"])
        for om, sm in zip(omega_pos, S_mag_pos):
            w.writerow([f"{om:.9g}", f"{sm:.9g}"])

    spec_img = os.path.join(out_dir, f"{tag}_spectrum.png")

    plt.figure(figsize=(6, 3.5), dpi=150)
    plt.plot(omega_pos, S_mag_pos, linewidth=1.5)
    plt.xlabel("ω  (≈ energy, ħ=1)")
    plt.ylabel(r"|FT{$C$}|  (normalized)")
    plt.title("Energy spectrum from autocorrelation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(spec_img, bbox_inches="tight")
    plt.close()

    print(f"Spectrum image saved to: {spec_img}")

    print(f"Saved:\n  {mp4_prob}\n  {mp4_real}\n  {csv_corr}\n  {csv_spec}")
    return mp4_prob, mp4_real, csv_corr, csv_spec


# ==========================
# ========== MAIN ==========
# ==========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Mask and Laplacian
    allowed = load_allowed_mask(IMG_PATH, NGRID)
    H, idx, n = build_laplacian(allowed)
    if USE_COMPLEX64:
        H = H.astype(np.complex64)

    # --- Build initial state ---
    if INIT_MODE.lower() == "gaussian":
        x0 = GAUSS_X0; y0 = GAUSS_Y0
        if x0 is None or y0 is None:
            x0, y0 = auto_place_bottom(allowed)
        psi0 = make_gaussian(allowed, x0, y0, GAUSS_SIGMA, GAUSS_KX, GAUSS_KY)

    elif INIT_MODE.lower() == "standing":
        # Construct standing wave with *node counts* (nx,ny) inside ROI,
        # then optionally launch it by multiplying a plane-wave.
        psi0 = make_standing_with_nodes(
            allowed,
            STAND_X0, STAND_X1, STAND_Y0, STAND_Y1,
            nodes_x=STAND_NX, nodes_y=STAND_NY,
            taper_sigma=STAND_TAPER_SIGMA,
            launch_kx=STAND_LAUNCH_KX, launch_ky=STAND_LAUNCH_KY,
            ref_point=STAND_LAUNCH_REFPT
        )

        # If you want the standing wave to have (roughly) the SAME kinetic energy
        # as a Gaussian with momentum magnitude K0, you could set launch_kx,launch_ky
        # so that |k|^2 = desired energy, e.g.:
        #   kx, ky = k_from_energy_and_angle(E_desired, theta_deg)

    else:
        raise ValueError("INIT_MODE must be 'gaussian' or 'standing'.")

    if np.all(np.abs(psi0) == 0):
        raise RuntimeError("Initial wavefunction is zero; adjust ROI/placement/nodes/taper.")

    # --- Evolve and save outputs ---
    evolve_and_save(allowed, H, psi0, OUT_DIR, tag=INIT_MODE.lower())

    print("Done.")

if __name__ == "__main__":
    start=time.time()
    main()
    end=time.time()
    print(f"Finsihed in {end-start} seconds")