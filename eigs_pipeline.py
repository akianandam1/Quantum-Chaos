"""
eigs_pipeline_params.py

Set parameters below; run this file directly (no CLI flags).
- Smooth boundary from a black/white image (white=allowed, black=wall).
- Hard (Dirichlet) Laplacian on the smoothed domain.
- Batched shift–invert eigen solve with persistent storage + resume.
- Montages for newly added modes (Re ψ and |ψ|^2) with a clear diverging palette.

Install once:
  pip install numpy scipy pillow matplotlib scikit-image
  # optional, recommended for single-file storage:
  pip install h5py
"""

# ======= USER PARAMETERS (EDIT HERE) =========================================
SDF_PATH       = "potentials/potential5.npz"   # produced by png_to_sdf_params.py
IMG_PATH        = "potentials/potential5.png"   # white=allowed, black=wall
trial="trial6"
OUT_DIR         = "eigensdf/"+trial        # all outputs saved here

# Geometry / smoothing (higher supersample => smoother edges; a bit slower once)
GRID_N          = 512               # final grid is GRID_N x GRID_N (e.g., 384, 512)
SUPERSAMPLE     = 10                # 8–12 typical
BLUR_SIGMA      = 0.6               # 0.4–0.9 tames pixel jaggies
SPLINE_SMOOTH   = 0.002             # 0..~0.01 (raise if input outline is noisy)

# Eigen solve targets
TARGET_K_TOTAL  = 3000              # total modes desired in the store (e.g., 5000)
BATCH_K         = 50                # modes per ARPACK call (50–120 typical)
ARPACK_TOL      = 1e-8              # relax to 1e-7 if convergence is slow
RESUME          = True              # if True, continue from existing store

# Storage backend: "auto" -> prefer HDF5, else NPZ shards
STORE_BACKEND   = "auto"            # "auto", "h5", or "npz"

# Montage rendering
MONTAGE_COUNT   = 60                # modes per montage image
MONTAGE_COLS    = 10                # image grid columns
MONTAGE_DPI     = 140
ALPHA_OVERLAY   = 0.65              # blend with mask
# ============================================================================

import os, json, math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, csr_matrix, identity
from scipy.sparse.linalg import eigsh, splu, LinearOperator
from skimage import measure
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# ---------- Smooth occupancy (anti-aliased boundary) ----------
def load_smooth_occupancy(path, target_N, supersample, blur_sigma,
                          contour_level=0.5, spline_smooth=0.002, spline_pts=4000,
                          keep_largest=True):
    im = Image.open(path).convert("L")
    Hi = target_N * supersample
    arr = np.asarray(im.resize((Hi, Hi), Image.LANCZOS), dtype=np.float32)/255.0
    if blur_sigma > 0:
        arr = ndi.gaussian_filter(arr, blur_sigma)
    contours = measure.find_contours(arr, level=contour_level)
    if not contours:
        raise RuntimeError("No contours found; check image polarity/level.")
    def interior_mean(c):
        yx = np.mean(c, axis=0)
        y = int(np.clip(yx[0], 0, arr.shape[0]-1))
        x = int(np.clip(yx[1], 0, arr.shape[1]-1))
        return arr[y, x]
    contours.sort(key=lambda c: -interior_mean(c))
    c = contours[0]
    y, x = c[:,0], c[:,1]
    if np.hypot(x[0]-x[-1], y[0]-y[-1]) > 1e-6:
        x = np.r_[x, x[0]]; y = np.r_[y, y[0]]
    tck, _ = splprep([x, y], s=spline_smooth*len(x), per=True, k=3)
    u = np.linspace(0, 1, spline_pts, endpoint=False)
    xs, ys = splev(u, tck)
    poly = [(float(xx), float(yy)) for xx, yy in zip(xs, ys)]
    canvas = Image.new("L", (Hi, Hi), 0)
    ImageDraw.Draw(canvas, "L").polygon(poly, outline=255, fill=255)
    filled = np.asarray(canvas, dtype=np.float32)/255.0
    if keep_largest:
        lab, n = ndi.label(filled > 0.5)
        if n >= 1:
            sizes = ndi.sum(filled > 0.5, lab, index=np.arange(1, n+1))
            keep = 1 + int(np.argmax(sizes))
            filled = (lab == keep).astype(np.float32)
    Hc = Hi // supersample * supersample
    filled = filled[:Hc, :Hc]
    occ = filled.reshape(Hc//supersample, supersample, Hc//supersample, supersample).mean(axis=(1,3))
    return occ.astype(np.float32), (occ >= 0.5)

# ---------- Dirichlet Laplacian on thresholded mask ----------
# --- Shortley–Weller Dirichlet builder (smooth hard wall) ---
from scipy.ndimage import distance_transform_edt

def build_dirichlet_hamiltonian_from_occ(occ, thresh=0.5, alpha_clip=1e-2):
    """
    Shortley–Weller finite-difference Laplacian on a smooth Dirichlet boundary.
    occ : float array in [0,1] (your anti-aliased occupancy)
    thresh : inside/outside split; level set φ=0 is near this threshold
    alpha_clip : minimum fractional distance (prevents huge 1/α coefficients)

    Returns:
      H (CSR), n, pack, unpack, allowed
    """
    Ny, Nx = occ.shape
    inside = occ >= thresh
    # signed distance φ: >0 inside, <0 outside (in grid units)
    # (edt gives distance to the nearest False; we need both sides)
    d_in  = distance_transform_edt(inside)
    d_out = distance_transform_edt(~inside)
    phi = d_in - d_out  # >0 inside

    allowed = inside.copy()
    idx = -np.ones((Ny, Nx), np.int64)
    idx[allowed] = np.arange(allowed.sum())
    n = int(allowed.sum())

    rows, cols, vals = [], [], []
    diag = np.zeros(n, np.float64)

    def add(a,b,v):
        rows.append(a); cols.append(b); vals.append(v)

    # helper: add SW contribution for a single neighbor direction
    def sw_link(a, y, x, yn, xn):
        """Cell (y,x) inside. Neighbor (yn,xn) may be inside (standard link)
        or outside (place boundary at fractional distance α along the segment)."""
        if 0 <= yn < Ny and 0 <= xn < Nx and allowed[yn, xn]:
            b = idx[yn, xn]
            add(a, b, -1.0)   # standard off-diagonal
            diag[a] += 1.0    # counts one neighbor
        else:
            # neighbor is outside => boundary lies between centers
            # φ(y,x)>0 (inside), φ(yn,xn)<0 (outside) approximately.
            phi_i = phi[y, x]
            # sample φ just across the face; if out-of-bounds, assume negative
            phi_o = phi[yn, xn] if (0 <= yn < Ny and 0 <= xn < Nx) else -phi_i
            denom = (phi_i - phi_o)
            if denom <= 0:
                # fall back to half-cell if something degenerate happens
                alpha = 0.5
            else:
                alpha = np.clip(phi_i / denom, alpha_clip, 1.0)  # 0<α≤1
            # Shortley–Weller: replace missing link by a diagonal term 1/α
            diag[a] += 1.0/alpha

    # loop interior allowed cells and add 4 directions
    for y in range(Ny):
        for x in range(Nx):
            a = idx[y, x]
            if a < 0:
                continue
            sw_link(a, y, x, y,   x+1)   # +x
            sw_link(a, y, x, y,   x-1)   # -x
            sw_link(a, y, x, y+1, x  )   # +y
            sw_link(a, y, x, y-1, x  )   # -y

    # assemble
    # diagonal last
    rows += list(range(n)); cols += list(range(n)); vals += list(diag)
    from scipy.sparse import coo_matrix
    H = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    def pack(img):
        return img[allowed].reshape(-1)

    def unpack(vec):
        out = np.zeros_like(allowed, dtype=np.complex128)
        out[allowed] = vec.reshape(-1)
        return out

    return H, n, pack, unpack, allowed


# --- Shortley–Weller Dirichlet builder from a smooth signed distance field φ ---
def build_dirichlet_hamiltonian_from_phi(occ, phi, thresh=0.0, alpha_clip=5e-3):
    """
    Shortley–Weller 5-point Laplacian with Dirichlet on a smooth boundary:
      - phi: (Ny,Nx) signed distance (>0 inside, <0 outside), in *cell units*
      - occ: (Ny,Nx) fractional occupancy (only used for previews; solver uses phi)
      - thresh: level-set used to define "inside" (usually 0.0)
      - alpha_clip: minimum fractional distance to avoid huge coefficients near tangencies
    Returns: H (CSR), n, pack, unpack, allowed
    """
    Ny, Nx = phi.shape
    allowed = phi > thresh
    idx = -np.ones((Ny, Nx), np.int64)
    idx[allowed] = np.arange(allowed.sum())
    n = int(allowed.sum())

    rows, cols, vals = [], [], []
    diag = np.zeros(n, np.float64)

    def add(a,b,v): rows.append(a); cols.append(b); vals.append(v)

    def sw_link(a, y, x, yn, xn):
        if 0 <= yn < Ny and 0 <= xn < Nx and allowed[yn, xn]:
            b = idx[yn, xn]
            add(a, b, -1.0)
            diag[a] += 1.0
        else:
            phi_i = phi[y, x]
            phi_o = phi[yn, xn] if (0 <= yn < Ny and 0 <= xn < Nx) else -phi_i
            denom = (phi_i - phi_o)
            if denom <= 0:
                alpha = 0.5
            else:
                alpha = np.clip(phi_i / denom, alpha_clip, 1.0)  # fractional dist to wall
            diag[a] += 1.0/alpha

    for y in range(Ny):
        for x in range(Nx):
            a = idx[y, x]
            if a < 0: continue
            sw_link(a, y, x, y,   x+1)
            sw_link(a, y, x, y,   x-1)
            sw_link(a, y, x, y+1, x  )
            sw_link(a, y, x, y-1, x  )

    from scipy.sparse import coo_matrix
    rows += list(range(n)); cols += list(range(n)); vals += list(diag)
    H = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    def pack(img):  return img[allowed].reshape(-1)
    def unpack(vec):
        out = np.zeros((Ny, Nx), dtype=np.complex128)
        out[allowed] = vec.reshape(-1)
        return out

    return H, n, pack, unpack, allowed

# ---------- Shift–invert (robust types) ----------
def low_band_eigs(H, k=60, sigma=0.0, tol=1e-8):
    n = H.shape[0]
    if k >= n-2:
        raise ValueError(f"k={k} too large for n={n}")
    I = identity(n, format="csc", dtype=np.complex128)
    Hc = H.astype(np.complex128)
    A = (Hc - sigma*I).tocsc()
    LU = splu(A)
    def OP(x):
        x = np.asarray(x, dtype=np.complex128, order="C")
        y = LU.solve(x)
        return np.asarray(y, dtype=np.complex128, order="C")
    OPinv = LinearOperator(H.shape, matvec=OP, dtype=np.complex128)
    evals, evecs = eigsh(Hc, k=k, sigma=sigma, which='LM', OPinv=OPinv, tol=tol)
    order = np.argsort(evals.real)
    return evals[order].real, evecs[:,order]

# ---------- Persistent store with resume ----------
class EigenStore:
    def __init__(self, out_dir, n, backend="auto"):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.n = int(n)
        self.backend = None
        self._h5 = None
        self.manifest_path = os.path.join(out_dir, "manifest.json")
        if backend in ("auto","h5"):
            try:
                import h5py  # noqa
                self.backend = "h5"
            except Exception:
                if backend=="h5":
                    raise
        if self.backend is None:
            self.backend = "npz"
        if self.backend == "h5":
            import h5py
            path = os.path.join(out_dir, "eigenpairs.h5")
            if os.path.exists(path):
                self._h5 = h5py.File(path, "a")
                if "evecs" in self._h5 and self._h5["evecs"].shape[0] != self.n:
                    raise RuntimeError("Existing HDF5 has different n.")
            else:
                self._h5 = h5py.File(path, "w")
                self._h5.create_dataset("evals", shape=(0,), maxshape=(None,), dtype="f8", chunks=True)
                self._h5.create_dataset("evecs", shape=(self.n,0), maxshape=(self.n,None), dtype="c16", chunks=(self.n,32))
                self._h5.attrs["n"] = self.n
        else:
            if not os.path.exists(self.manifest_path):
                with open(self.manifest_path, "w") as f:
                    json.dump({"n": self.n, "shards": []}, f, indent=2)
            else:
                with open(self.manifest_path,"r") as f:
                    man=json.load(f)
                if int(man.get("n",-1)) != self.n:
                    raise RuntimeError("Existing NPZ manifest has different n.")

    def count(self):
        if self.backend == "h5":
            return int(self._h5["evals"].shape[0])
        with open(self.manifest_path,"r") as f:
            man=json.load(f)
        return int(sum(sh["stop"]-sh["start"] for sh in man["shards"]))

    def last_e(self):
        c = self.count()
        if c == 0: return None
        if self.backend == "h5":
            return float(self._h5["evals"][-1])
        with open(self.manifest_path,"r") as f:
            man=json.load(f)
        if not man["shards"]: return None
        sh = man["shards"][-1]
        data = np.load(os.path.join(self.out_dir, sh["path"]))
        return float(np.asarray(data["evals"])[-1])

    def append(self, evals, evecs):
        m = int(evals.shape[0])
        assert evecs.shape == (self.n, m)
        if self.backend == "h5":
            dsE, dsV = self._h5["evals"], self._h5["evecs"]
            old = dsE.shape[0]
            dsE.resize((old+m,)); dsV.resize((self.n, old+m))
            dsE[old:old+m] = evals
            dsV[:,old:old+m] = evecs
            self._h5.flush()
        else:
            start = self.count()
            stop = start + m
            shard = os.path.join(self.out_dir, f"eigs_{start:06d}_{stop-1:06d}.npz")
            np.savez_compressed(shard, evals=evals.astype(np.float64), evecs=evecs.astype(np.complex128))
            with open(self.manifest_path,"r") as f:
                man=json.load(f)
            man["shards"].append({"start":start,"stop":stop,"path":os.path.basename(shard)})
            with open(self.manifest_path,"w") as f:
                json.dump(man,f,indent=2)

    def close(self):
        if self._h5 is not None: self._h5.close()

# ---------- Batched solver that resumes above last_E ----------
def batched_low_eigs_resume(H, total_k, batch_k, tol, store: EigenStore,
                            start_count=0, step_factor=0.6, dedup_tol=1e-7, verbose=True):
    collected = start_count
    remain = max(0, total_k - collected)
    if remain == 0:
        if verbose: print(f"Already have {collected}/{total_k}. Nothing to do.")
        return np.empty((0,), float), np.empty((H.shape[0], 0), complex)
    last_E = store.last_e()
    sigma = (float(last_E) + 0.2) if (last_E is not None) else 0.0
    got_vals, got_vecs = [], []
    while remain > 0:
        k_here = min(batch_k, remain)
        if verbose: print(f"[batch] requesting k={k_here} near sigma={sigma:.6g}")
        ev, U = low_band_eigs(H, k=k_here, sigma=sigma, tol=tol)
        keep = [j for j,E in enumerate(ev) if not (last_E is not None and E <= last_E + dedup_tol)]
        if not keep:
            sigma += 0.5
            continue
        evk, Uk = ev[keep], U[:, keep]
        store.append(evk, Uk)
        collected += len(evk); remain = total_k - collected
        last_E = float(evk[-1])
        got_vals.append(evk); got_vecs.append(Uk)
        spacing = np.diff(ev).mean() if len(ev)>1 else 0.2
        bump = max(1e-3, step_factor*max(spacing, 0.1))
        sigma = last_E + bump
        if verbose: print(f"  collected {collected}/{total_k}; next sigma≈{sigma:.6g}")
    Enew = np.concatenate(got_vals, axis=0)
    Vnew = np.column_stack(got_vecs)
    return Enew, Vnew

# ---------- Montages (for newly added modes only) ----------
def save_modes_montage(occ, evals, evecs, out_dir, start_index_label,
                       which="real", cmap=cm.RdBu_r, alpha=ALPHA_OVERLAY,
                       per_image_count=MONTAGE_COUNT, ncols=MONTAGE_COLS, dpi=MONTAGE_DPI,
                       embed=None):
    Ny, Nx = occ.shape
    os.makedirs(out_dir, exist_ok=True)
    if embed is None:
        embed = lambda v: v.reshape(Ny, Nx)
    total = evecs.shape[1]
    for offset in range(0, total, per_image_count):
        stop = min(offset+per_image_count, total)
        rows = int(np.ceil((stop-offset)/ncols))
        fig, axes = plt.subplots(rows, ncols, figsize=(1.6*ncols, 1.6*rows), dpi=dpi)
        axes = np.array(axes).ravel()
        bg = (255*np.dstack([occ,occ,occ])).astype(np.uint8)
        for i,k in enumerate(range(offset, stop)):
            ax = axes[i]
            psi_img = embed(evecs[:, k])
            if which=="real":
                re = psi_img.real
                s = np.max(np.abs(re)) + 1e-12
                x = 0.5*(re/s + 1.0)
                img = (cmap(x)[:, :, :3]*255).astype(np.uint8)
            else:
                p = np.abs(psi_img)**2
                p = p / (p.max()+1e-12)
                img = (cmap(p)[:, :, :3]*255).astype(np.uint8)
            blend = (alpha*img + (1-alpha)*bg).astype(np.uint8)
            gidx = start_index_label + k
            ax.imshow(blend, origin='lower')
            ax.set_title(f"#{gidx}  E≈{evals[k]:.6g}", fontsize=7)
            ax.axis('off')
        for j in range(i+1, axes.size):
            axes[j].axis('off')
        plt.tight_layout()
        tag = f"{which}_{start_index_label+offset:06d}_{start_index_label+stop-1:06d}"
        out = os.path.join(out_dir, f"modes_{tag}.png")
        plt.savefig(out, bbox_inches="tight"); plt.close(fig)
        print("wrote", out)

# ---------- Main (no CLI; uses parameters at the top) ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # ===== Load smooth signed distance φ (generated by png_to_sdf_params.py) =====
    if not os.path.exists(SDF_PATH):
        raise FileNotFoundError(
            f"SDF_PATH '{SDF_PATH}' not found. Run png_to_sdf_params.py first to create it."
        )

    geo = np.load(SDF_PATH)
    phi = geo["phi"].astype(np.float32)  # signed distance (>0 inside), in cell units
    occ = geo["occ"].astype(np.float32)  # fractional occupancy (for previews)
    Ny, Nx = phi.shape
    print(f"Loaded SDF: {SDF_PATH}  (phi shape {phi.shape})")

    # Build Shortley–Weller Dirichlet directly from φ
    H, n, pack, unpack, allowed = build_dirichlet_hamiltonian_from_phi(
        occ=occ, phi=phi, thresh=0.0, alpha_clip=5e-3
    )
    print(f"Grid: {Ny}x{Nx}   DOFs: {n}")

    # Save a preview that matches what the solver 'sees' (smooth φ=0 boundary)
    from skimage import measure as skmeasure
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), dpi=150)
    ax[0].imshow(occ, origin='lower', cmap='gray', vmin=0, vmax=1)
    ax[0].set_title("fractional occupancy");
    ax[0].axis('off')
    im1 = ax[1].imshow(phi, origin='lower', cmap='coolwarm');
    ax[1].axis('off')
    ax[1].set_title("signed distance ϕ (cells)");
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    levels = [phi.min() - 1e-9, 0.0, phi.max() + 1e-9]
    ax[2].contourf(phi, levels=levels, colors=['black', 'white'], antialiased=True)
    for c in skmeasure.find_contours(phi, level=0.0):
        ax[2].plot(c[:, 1], c[:, 0], 'r-', lw=1.2, alpha=0.9)
    ax[2].set_title("smooth Dirichlet boundary (φ=0)");
    ax[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "potential_preview_from_phi.png"), bbox_inches="tight")
    plt.close()
    print("Saved potential_preview_from_phi.png")

    # Also store geometry we are actually using (for reproducibility/resume)
    np.savez_compressed(os.path.join(OUT_DIR, "geometry_used.npz"),
                        phi=phi, occ=occ, allowed=allowed.astype(np.uint8),
                        shape=np.array([Ny, Nx], dtype=np.int64))

    # Open/create store and see how many we already have
    store = EigenStore(OUT_DIR, n=n, backend=STORE_BACKEND)
    print(f"DOFs this run (from builder) n = {n}")
    print(f"Store expects n = {store.n}")
    if store.n != n:
        raise RuntimeError(
            f"Geometry mismatch: store.n={store.n}, builder n={n}. "
            f"Pass n from the builder to EigenStore or start a fresh OUT_DIR."
        )
    have = store.count()
    target = int(TARGET_K_TOTAL)
    print(f"Target K={target}. Already stored: {have}.")

    # Solve remaining in batches, appending as we go
    Enew, Vnew = batched_low_eigs_resume(
        H, total_k=target, batch_k=BATCH_K, tol=ARPACK_TOL,
        store=store, start_count=have, step_factor=0.6, dedup_tol=1e-7, verbose=True
    )
    store.close()

    # Append/Save eigenvalues CSV (only new)
    csv_path = os.path.join(OUT_DIR, "eigenvalues.csv")
    if Enew.size > 0:
        mode = "ab" if os.path.exists(csv_path) else "wb"
        with open(csv_path, mode) as f:
            if mode == "wb":
                np.savetxt(f, Enew.reshape(-1,1), delimiter=",", header="E", comments="")
            else:
                np.savetxt(f, Enew.reshape(-1,1), delimiter=",")
        print("Eigenvalues appended/saved to eigenvalues.csv.")
    else:
        print("No new eigenvalues written (already at or above target).")

    # Montages only for NEW modes
    if Enew.size > 0:
        base = have
        save_modes_montage(occ, Enew, Vnew, out_dir=OUT_DIR, start_index_label=base,
                           which="real", embed=lambda v: unpack(v))
        save_modes_montage(occ, Enew, Vnew, out_dir=OUT_DIR, start_index_label=base,
                           which="prob", embed=lambda v: unpack(v))
    print("Done.")

if __name__ == "__main__":
    start=time.time()
    main()
    end=time.time()
    print(f"Finished in {end-star} seconds")