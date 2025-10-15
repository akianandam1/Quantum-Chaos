import os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, csr_matrix, diags, identity
from scipy.sparse.linalg import eigsh, splu
from skimage import measure
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse.linalg import LinearOperator, eigsh, splu
from scipy.sparse import identity

# -----------------------------
# 1) Smooth occupancy from image
# -----------------------------

def load_smooth_occupancy(
    path,
    target_N=384,
    supersample=10,          # 8–12 gives very smooth edges
    blur_sigma=0.6,          # slight blur before contouring (anti-jaggies)
    contour_level=0.5,       # isovalue for find_contours in [0..1]
    spline_smooth=0.002,     # B-spline smoothing (0.. ~0.01)
    spline_pts=4000,         # points along the smooth closed curve
    keep_largest=True,
):
    """
    Returns:
      occ: (Ny,Nx) float32 in [0,1], fractional occupancy (white=1 allowed, black=0 wall)
      mask_preview: simple boolean mask for preview
    """
    # 1) Load & normalize
    im = Image.open(path).convert("L")
    # supersample canvas
    Hi = target_N * supersample
    arr = np.asarray(im.resize((Hi, Hi), Image.LANCZOS), dtype=np.float32) / 255.0
    if blur_sigma > 0:
        arr = ndi.gaussian_filter(arr, blur_sigma)

    # 2) Find outer white region (allowed) contour(s)
    # skimage.find_contours expects high=1 is boundary
    contours = measure.find_contours(arr, level=contour_level)
    if not contours:
        raise RuntimeError("No contours found; check image polarity or level.")
    # Choose the contour whose interior is mostly white
    def interior_mean(c):
        # take centroid and sample pixel
        yx = np.mean(c, axis=0)[0], np.mean(c, axis=0)[1]
        y = int(np.clip(yx[0], 0, arr.shape[0]-1))
        x = int(np.clip(yx[1], 0, arr.shape[1]-1))
        return arr[y, x]
    contours.sort(key=lambda c: -interior_mean(c))
    c = contours[0]   # best candidate

    # 3) Fit a periodic cubic B-spline to the contour and resample densely
    # Contour is array of (row, col) = (y, x). We want periodic parameterization.
    y, x = c[:, 0], c[:, 1]
    # Close explicitly
    if np.hypot(x[0]-x[-1], y[0]-y[-1]) > 1e-6:
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
    # Parameterize & smooth (per=True => periodic)
    tck, _ = splprep([x, y], s=spline_smooth*len(x), per=True, k=3)
    u = np.linspace(0, 1, spline_pts, endpoint=False)
    xs, ys = splev(u, tck)

    # 4) Rasterize smooth polygon on the supersampled canvas
    poly = [(float(xx), float(yy)) for xx, yy in zip(xs, ys)]
    # Pillow wants (x,y) in integer canvas coords; anti-aliased fill via upsampling is enough
    canvas = Image.new("L", (Hi, Hi), 0)
    draw = ImageDraw.Draw(canvas, "L")
    draw.polygon(poly, outline=255, fill=255)
    filled = np.asarray(canvas, dtype=np.float32) / 255.0

    # 5) Optionally keep only largest filled component (in case of extra bits)
    if keep_largest:
        lab, n = ndi.label(filled > 0.5)
        if n >= 1:
            sizes = ndi.sum(filled > 0.5, lab, index=np.arange(1, n+1))
            keep = 1 + int(np.argmax(sizes))
            filled = (lab == keep).astype(np.float32)

    # 6) Downsample by block averaging to get fractional occupancy
    Hc = Hi // supersample * supersample
    filled = filled[:Hc, :Hc]
    occ = filled.reshape(Hc//supersample, supersample, Hc//supersample, supersample).mean(axis=(1,3))
    occ = occ.astype(np.float32)
    mask_preview = occ >= 0.5
    return occ, mask_preview

# --------------------------------------------------
# 2) Weighted soft-wall Hamiltonian from occupancy
# --------------------------------------------------

def build_weighted_hamiltonian(occ, V0=5e3, p=2.0):
    """
    occ in [0,1]: fractional occupancy (1=allowed, 0=wall).
    Build H = -div(w ∇·) + V using face weights w from occ.
    This version assembles the standard (positive) graph Laplacian:
      L_ii = sum_j w_ij,  L_ij = -w_ij  (i≠j)
    so that the quadratic form is  sum_{<i,j>} w_ij |psi_i - psi_j|^2  +  sum_i V_i |psi_i|^2  ≥ 0.
    """
    Ny, Nx = occ.shape
    wx = 0.5*(occ[:, 1:] + occ[:, :-1])   # vertical faces between (i,j)-(i,j+1)
    wy = 0.5*(occ[1:, :] + occ[:-1, :])   # horizontal faces between (i,j)-(i+1,j)

    n = Ny * Nx
    def idx(i, j): return i*Nx + j

    rows, cols, vals = [], [], []
    diag = np.zeros(n, dtype=np.float64)

    # x-neighbors
    for i in range(Ny):
        for j in range(Nx-1):
            w = float(wx[i, j])
            if w <= 1e-12: continue
            a = idx(i, j)
            b = idx(i, j+1)
            # off-diagonal = -w (symmetric)
            rows.append(a); cols.append(b); vals.append(-w)
            rows.append(b); cols.append(a); vals.append(-w)
            # diagonal accumulates +w on both nodes
            diag[a] += w
            diag[b] += w

    # y-neighbors
    for i in range(Ny-1):
        for j in range(Nx):
            w = float(wy[i, j])
            if w <= 1e-12: continue
            a = idx(i, j)
            b = idx(i+1, j)
            rows.append(a); cols.append(b); vals.append(-w)
            rows.append(b); cols.append(a); vals.append(-w)
            diag[a] += w
            diag[b] += w

    # Soft wall potential (keeps symmetry, V ≥ 0)
    V = V0 * (1.0 - occ)**p
    diag += V.ravel()

    # Assemble H (symmetric real)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    H = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    H += diags(diag, 0)
    # pack/unpack helpers
    def pack(img):  return img.reshape(-1)
    def unpack(vec): return vec.reshape(Ny, Nx)
    return H, n, pack, unpack



def build_dirichlet_hamiltonian_from_occ(occ, thresh=0.5):
    """
    Construct standard Dirichlet Laplacian on allowed cells (occ>=thresh).
    Returns H (CSR, real symmetric positive-definite), n, pack, unpack, allowed mask.
    """
    allowed = (occ >= thresh)
    Ny, Nx = allowed.shape
    idx = -np.ones((Ny, Nx), dtype=np.int64)
    idx[allowed] = np.arange(allowed.sum())
    n = int(allowed.sum())

    rows, cols, vals = [], [], []

    def add(a, b, val):
        rows.append(a); cols.append(b); vals.append(val)

    # Standard 5-point with Dirichlet: offdiag = -1 between allowed neighbors, diag = degree
    deg = np.zeros(n, dtype=np.float64)

    for y in range(Ny):
        for x in range(Nx):
            a = idx[y, x]
            if a < 0:
                continue
            # 4-neighbors
            for dy, dx in ((0,1),(0,-1),(1,0),(-1,0)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < Ny and 0 <= xx < Nx and idx[yy, xx] >= 0:
                    b = idx[yy, xx]
                    add(a, b, -1.0)   # off-diagonal
                    deg[a] += 1.0     # count degree

    # Diagonal
    rows += list(range(n)); cols += list(range(n)); vals += list(deg)

    H = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    # No soft potential term; Dirichlet is enforced by removing outside nodes entirely.

    def pack(img):
        v = np.zeros(n, dtype=np.float64)
        v[:] = img[allowed].reshape(-1)
        return v

    def unpack(vec):
        out = np.zeros_like(allowed, dtype=np.float64)
        out[allowed] = vec.reshape(-1)
        return out

    return H, n, pack, unpack, allowed




# ----------------------------------------------
# 3) Compute eigenpairs (shift-invert, low band)
# ----------------------------------------------


def low_band_eigs(H, k=60, sigma=0.0, tol=1e-8):
    """
    Compute k eigenpairs of Hermitian H closest to sigma using ARPACK shift-invert.
    Returns eigenvalues (ascending, real) and eigenvectors (n x k), both complex128.
    """
    n = H.shape[0]
    if k >= n - 2:
        raise ValueError(f"k={k} too large for n={n} (need k <= n-3).")

    # Build shifted operator in COMPLEX and factor once
    I = identity(n, format="csc", dtype=np.complex128)
    Hc = H.astype(np.complex128)
    A = (Hc - sigma * I).tocsc()

    LU = splu(A)  # complex LU

    def OP(x):
        # ARPACK can pass complex RHS; ensure dtype matches the factorization
        x = np.asarray(x, dtype=np.complex128, order="C")
        y = LU.solve(x)
        # guarantee contiguous complex128 for ARPACK
        return np.asarray(y, dtype=np.complex128, order="C")

    OPinv = LinearOperator(H.shape, matvec=OP, dtype=np.complex128)

    # Ask for largest magnitude of (H - sigma I)^{-1}, i.e. closest to sigma in H
    evals, evecs = eigsh(Hc, k=k, sigma=sigma, which='LM', OPinv=OPinv, tol=tol)
    order = np.argsort(evals.real)
    return evals[order].real, evecs[:, order]



def batched_low_eigs(H, total_k=500, batch_k=80, sigma0=0.0, tol=1e-8,
                     step_factor=0.6, dedup_tol=1e-7, verbose=True):
    """
    Get the lowest `total_k` eigenpairs of H using shift-invert in batches.
    - Starts near sigma0 (usually 0.0), then slides sigma upward so each
      batch returns the next chunk.
    - Deduplicates across batches and finally sorts by energy.

    Returns:
        evals (K,), evecs (n, K) with K <= total_k (exactly total_k if converged).
    """
    n = H.shape[0]
    got_vals = []
    got_vecs = []
    sigma = float(sigma0)
    remain = total_k
    last_E = None

    while remain > 0:
        k_here = min(batch_k, remain)
        if verbose:
            print(f"[batch] requesting k={k_here} near sigma={sigma:.6g}")

        ev, U = low_band_eigs(H, k=k_here, sigma=sigma, tol=tol)  # complex evecs
        # Drop anything that is below what we already have (duplicate/overlap), keep new
        for j, E in enumerate(ev):
            if last_E is not None and E <= last_E + dedup_tol:
                continue
            got_vals.append(E)
            got_vecs.append(U[:, j])

        got = len(got_vals)
        if got == 0:
            # advance sigma a bit and try again
            sigma += 0.5
            continue

        # Prepare next sigma ~ just above the last new eigenvalue
        last_batch_spacing = np.diff(ev).mean() if len(ev) > 1 else 0.2
        bump = max(1e-3, step_factor * max(last_batch_spacing, 0.1))
        last_E = got_vals[-1]
        sigma = last_E + bump

        remain = total_k - got
        if verbose:
            print(f"  collected {got}/{total_k}; next sigma≈{sigma:.6g}")

    # Stack & sort
    E = np.asarray(got_vals, dtype=np.float64)
    V = np.column_stack(got_vecs)
    order = np.argsort(E)
    E = E[order]
    V = V[:, order]
    if verbose:
        print(f"Done: {len(E)} eigenpairs collected.")
    return E, V




# -----------------------------------------
# 4) Quick plot of eigenmodes over the mask
# -----------------------------------------

def show_modes(occ, evals, evecs, ncols=6, nshow=24, cmap_real=cm.RdBu_r):
    Ny, Nx = occ.shape
    fig, axes = plt.subplots(int(np.ceil(nshow/ncols)), ncols, figsize=(1.9*ncols, 1.9*np.ceil(nshow/ncols)))
    axes = np.array(axes).ravel()
    bg = np.stack([occ, occ, occ], axis=-1)  # white where allowed
    for i in range(nshow):
        ax = axes[i]
        psi = evecs[:, i].reshape(Ny, Nx)
        re = psi.real
        s = np.max(np.abs(re)) + 1e-12
        x = 0.5*(re/s + 1.0)
        img = (cmap_real(x)[:, :, :3] * 255).astype(np.uint8)
        # overlay on occupancy
        blend = 0.65*img + 0.35*(255*np.dstack([occ,occ,occ])).astype(np.uint8)
        ax.imshow(blend.astype(np.uint8), origin='lower')
        ax.set_title(f"{i}: E≈{evals[i]:.3f}", fontsize=8)
        ax.axis('off')
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()



def save_modes_montage(occ, evals, evecs, out_dir, start=0, count=60, ncols=10,
                       cmap_real=cm.RdBu_r, alpha=0.65, dpi=140, which="real",
                       embed=None):
    """
    Save a grid of `count` modes starting at index `start` to a PNG.
    which: "real" -> show Re(psi); "prob" -> show |psi|^2
    embed: callable(vec) -> (Ny,Nx) image of the eigenvector on the full grid.
           For soft-wall (full-grid) builds: embed = lambda v: v.reshape(Ny,Nx)
           For Dirichlet (allowed-only) builds: embed = the `unpack` function.
    """
    Ny, Nx = occ.shape
    os.makedirs(out_dir, exist_ok=True)
    stop = min(start + count, evecs.shape[1])
    rows = int(np.ceil((stop - start) / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(1.6*ncols, 1.6*rows), dpi=dpi)
    axes = np.array(axes).ravel()

    bg = (255*np.dstack([occ,occ,occ])).astype(np.uint8)

    if embed is None:
        # default: assume full-grid vectors
        embed = lambda v: v.reshape(Ny, Nx)

    for i, k in enumerate(range(start, stop)):
        ax = axes[i]
        psi_img = embed(evecs[:, k])  # <<< this is the important change

        if which == "real":
            re = psi_img.real
            s = np.max(np.abs(re)) + 1e-12
            x = 0.5*(re/s + 1.0)
            img = (cmap_real(x)[:, :, :3] * 255).astype(np.uint8)
        else:
            p = np.abs(psi_img)**2
            p = p / (p.max() + 1e-12)
            # use same palette as real, or swap to a sequential map if you prefer
            img = (cmap_real(p)[:, :, :3] * 255).astype(np.uint8)

        blend = (alpha*img + (1-alpha)*bg).astype(np.uint8)
        ax.imshow(blend, origin='lower')
        ax.set_title(f"#{k}  E≈{evals[k]:.4g}", fontsize=7)
        ax.axis('off')

    for j in range(i+1, axes.size):
        axes[j].axis('off')

    plt.tight_layout()
    out = os.path.join(out_dir, f"modes_{which}_{start:04d}_{stop-1:04d}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

# -----------------------
# 5) Example entry point
# -----------------------

if __name__ == "__main__":
    IMG = "potentials/potential5.png"   # white=allowed, black=wall (your domain)

    # # A) Build smooth fractional occupancy
    # occ, mask_preview = load_smooth_occupancy(
    #     IMG,
    #     target_N=384,
    #     supersample=10,      # raise to 12–16 for razor edges (slower once at startup)
    #     blur_sigma=0.6,
    #     spline_smooth=0.002, # increase a bit if the input outline is very jaggy
    #     spline_pts=4000
    # )



    # # B) Build soft-wall Hamiltonian
    # H, n, pack, unpack = build_weighted_hamiltonian(
    #     occ,
    #     V0=8e3,  # wall strength; can raise to 1e4–2e4 for harder boundary
    #     p=2.0
    # )

    # Build smooth occupancy from image (same as before)
    occ, _ = load_smooth_occupancy(IMG, target_N=384, supersample=10, blur_sigma=0.6,
                                   spline_smooth=0.002, spline_pts=4000)

    # Preview of occupancy and rough mask
    plt.figure(figsize=(4,4)); plt.imshow(occ, origin='lower', cmap='gray'); plt.title("fractional occupancy (anti-aliased)"); plt.show()

    # Build *hard-wall* Dirichlet Hamiltonian on a thresholded mask
    H, n, pack, unpack, allowed = build_dirichlet_hamiltonian_from_occ(occ, thresh=0.5)



    # C) Solve many modes in batches
    TOTAL = 500  # target total number of eigenpairs
    BATCH = 50  # size per ARPACK call (50–120 is typical)
    # Now H is positive-definite; its eigenvalues are the "energies" (up to your Δx, ħ=1 units).
    E, V = batched_low_eigs(H, total_k=TOTAL, batch_k=BATCH, sigma0=0.0, tol=1e-8)



    # Save eigenvalues to CSV
    dir="eigs2"
    os.makedirs(dir, exist_ok=True)
    np.savetxt(f"{dir}eigenvalues_lowest500.csv", E, delimiter=",", header="E", comments="")
    print(f"Saved eigenvalues to {dir}/eigenvalues_lowest500.csv")

    # D) Save montages in batches (both real part and probability)
    for start in range(0, min(TOTAL, V.shape[1]), 60):
        save_modes_montage(occ, E, V, out_dir=dir, start=start, count=60, which="real", embed=unpack)
        save_modes_montage(occ, E, V, out_dir=dir, start=start, count=60, which="prob",  embed=unpack)
