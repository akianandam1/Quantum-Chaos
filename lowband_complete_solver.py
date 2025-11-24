#!/usr/bin/env python3
# lowband_complete_solver.py
#
# Collect ALL eigenpairs up to a chosen energy cutoff E_CUT (or until N_TARGET),
# with no low-energy gaps, using adaptive shift–invert windows + dedup.
#
# Output: eigenpairs.h5  (datasets: evals [K], evecs [n,K]),
#         manifest.json  (metadata), geometry_used.npz (for projection)
#
# Requirements: numpy, scipy, h5py  (optional: scikit-image for a nicer perimeter later)

import os, json, math, time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, eye
from scipy.sparse.linalg import splu, eigsh

# ========= USER PARAMS =========
SDF_PATH      = "potentials/potential12.npz"   # your φ (signed distance) NPZ (from png_to_sdf)
OUT_DIR       = "eigensdf/doublewell/roland_narrow1"           # new output directory (do NOT reuse old one)
E_CUT         = 1                          # collect all modes with E <= E_CUT
N_TARGET      = 8000                          # or set e.g. 12000 to stop at count instead of E_CUT
SIGMA_GRID    = np.arange(0.0, 3.0, 0.25)     # initial σ sweep (you can widen to E_CUT+margin)
K_BATCH       = 256                           # how many to ask per ARPACK call
TOL_E         = 1e-8                          # energy tolerance for dedup
TOL_CORR      = 0.90                          # if |<v_new, v_old>| > TOL_CORR and |E-E'|<TOL_E -> duplicate
ARPACK_TOL    = 1e-8                          # internal eigsh tolerance
MAX_REFINES   = 40                            # extra sigmas inserted adaptively for gap-filling
SAVE_THIN     = 100                          # flush to disk every this many NEW modes
# =================================
os.makedirs(OUT_DIR,exist_ok=True)

# ======= GEOMETRY / HAMILTONIAN =======
def load_phi_allowed(sdf_path):
    g = np.load(sdf_path)
    phi = g["phi"].astype(np.float32)
    allowed = phi > 0.0
    return phi, allowed

def build_dirichlet_sw(phi, allowed, alpha_clip=5e-3):
    Ny, Nx = phi.shape
    idx = -np.ones((Ny, Nx), np.int64)
    idx[allowed] = np.arange(allowed.sum())
    n = int(allowed.sum())

    rows, cols, vals = [], [], []
    diag = np.zeros(n, np.float64)
    def add(a,b,v): rows.append(a); cols.append(b); vals.append(v)
    def link(a, y, x, yn, xn):
        if 0 <= yn < Ny and 0 <= xn < Nx and allowed[yn, xn]:
            b = idx[yn, xn]; add(a, b, -1.0); diag[a] += 1.0
        else:
            phi_i = phi[y, x]
            phi_o = phi[yn, xn] if (0 <= yn < Ny and 0 <= xn < Nx) else -phi_i
            denom = (phi_i - phi_o)
            alpha = 0.5 if denom <= 0 else np.clip(phi_i/denom, alpha_clip, 1.0)
            diag[a] += 1.0/alpha
    for y in range(Ny):
        for x in range(Nx):
            a = idx[y, x]
            if a < 0: continue
            link(a,y,x,y,x+1); link(a,y,x,y,x-1); link(a,y,x,y+1,x); link(a,y,x,y-1,x)

    rows += list(range(n)); cols += list(range(n)); vals += list(diag)
    H = coo_matrix((vals,(rows,cols)), shape=(n,n)).tocsr()

    def pack(img2d):  return img2d[allowed].reshape(-1)
    def unpack(vec):
        out = np.zeros((Ny, Nx), dtype=np.complex128)
        out[allowed] = vec.reshape(-1)
        return out

    return H, n, pack, unpack, allowed, (Ny, Nx)

# ======= STORAGE =======
def h5_open(out_dir, n, mode="a"):
    os.makedirs(out_dir, exist_ok=True)
    h5p = os.path.join(out_dir, "eigenpairs.h5")
    if (mode == "w") or (not os.path.exists(h5p)):
        with h5py.File(h5p, "w") as f:
            f.create_dataset("evals", shape=(0,), maxshape=(None,), dtype="f8")
            f.create_dataset("evecs", shape=(n,0), maxshape=(n,None), dtype="c16", chunks=(min(n,8192),1))
    return h5py.File(h5p, "a")

def h5_append(f, evals_new, evecs_new):
    K0 = f["evals"].shape[0]
    K1 = K0 + evals_new.shape[0]
    f["evals"].resize((K1,))
    f["evecs"].resize((f["evecs"].shape[0], K1))
    f["evals"][K0:K1] = np.asarray(evals_new, dtype=np.float64)
    # 👇 ensure complex128 going into a c16 dataset
    f["evecs"][:, K0:K1] = np.asarray(evecs_new, dtype=np.complex128)

# ======= DEDUP / CLUSTER HANDLING =======
def dedup_merge(existing_E, existing_sel_idx, E_new, V_new, store_reader, tol_E=1e-8, tol_corr=0.90):
    """
    existing_E: 1D numpy array of all evals already stored
    existing_sel_idx: numpy array of indices (0..len(existing_E)-1) whose E within [0, E_CUT+margin] for correlation checks
    E_new: 1D (k,), V_new: (n,k), dense
    store_reader: function to read eigenvectors by indices from store when needed
    Returns (keep_mask, dup_count)
    """
    keep = np.ones(E_new.shape[0], dtype=bool)
    if existing_E.size == 0:
        return keep, 0
    # for each new eigenvalue, check energy-dedup +/- tol_E
    for j, En in enumerate(E_new):
        # candidates in store with |E- En| < tol_E
        near = np.where(np.abs(existing_E[existing_sel_idx] - En) <= tol_E)[0]
        if near.size == 0:
            continue
        # load those candidate vectors and check correlation
        idxs = existing_sel_idx[near]
        V_old = store_reader(idxs)   # shape (n, m)
        vj = V_new[:, j]
        # Normalize (just in case)
        vj = vj / (np.linalg.norm(vj) + 1e-20)
        # correlation with each old vector
        corr = np.abs(V_old.conj().T @ vj)
        if np.max(corr) >= tol_corr:
            keep[j] = False
    return keep, int((~keep).sum())

# ======= SHIFT-INVERT ARPACK =======
def solve_near_sigma(H, sigma, k, tol=1e-8):
    # factor (H - sigma I)
    n = H.shape[0]
    A = (H - sigma*eye(n, format="csr")).tocsc()
    LU = splu(A, permc_spec="COLAMD")
    # operator: (H - sigma I)^{-1} for ARPACK
    from scipy.sparse.linalg import LinearOperator
    def op(x): return LU.solve(x)
    OP = LinearOperator((n,n), matvec=op, dtype=np.complex128)
    # ARPACK wants 'which=LM' on OP to get closest to sigma in original
    evals, evecs = eigsh(H, k=k, which="LM", sigma=sigma, OPinv=OP, tol=tol)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order].astype(np.complex128, copy=False)  # 👈 add this
    return evals, evecs

    # sort by energy
    order = np.argsort(evals)
    return evals[order], evecs[:, order]

# ======= MAIN DRIVER =======
def main():
    t0 = time.time()
    phi, allowed = load_phi_allowed(SDF_PATH)
    H, n, pack, unpack, allowed_mask, (Ny, Nx) = build_dirichlet_sw(phi, allowed)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Save geometry snapshot used (for projection later)
    np.savez_compressed(os.path.join(OUT_DIR, "geometry_used.npz"),
                        phi=phi, allowed=allowed_mask, shape=np.array([Ny, Nx], np.int32))
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump({"note":"lowband_complete_solver","E_CUT":E_CUT,"N_TARGET":N_TARGET}, f, indent=2)

    f = h5_open(OUT_DIR, n)
    E_store = np.array(f["evals"][:])  # maybe zero
    K_store = E_store.size

    # helper to read vectors by indices from HDF5
    def read_vecs(idxs):
        with h5py.File(os.path.join(OUT_DIR,"eigenpairs.h5"),"r") as fr:
            return np.array(fr["evecs"][:, idxs], dtype=np.complex128)

    def flush_plot():
        E_sorted = np.sort(np.array(f["evals"][:]))
        plt.figure(figsize=(6,3), dpi=140)
        xs = np.arange(E_sorted.size)
        plt.plot(xs, E_sorted)
        plt.axhline(E_CUT, color='r', ls='--', lw=1)
        plt.xlabel("#"); plt.ylabel("E")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "progress_E_sorted.png"), bbox_inches="tight")
        plt.close()

    print(f"[geom] n={n}, grid={Ny}x{Nx}, target E_CUT={E_CUT}, N_TARGET={N_TARGET}")
    new_total = 0
    sigmas = list(SIGMA_GRID.astype(float))
    refines = 0

    # selection of indices below E_CUT for faster correlation checks
    def update_sel_idx(E_arr):
        return np.where(E_arr <= (E_CUT + 5*max(1e-6, TOL_E)))[0]

    sel_idx = update_sel_idx(E_store)

    while True:
        if (N_TARGET is not None) and (K_store >= N_TARGET):
            print(f"[stop] reached N_TARGET={N_TARGET}")
            break
        # if we already cover all below E_CUT (by Weyl headcount), stop
        if E_CUT is not None:
            area = allowed_mask.sum()
            weyl_est = (area/(4*np.pi))*E_CUT
            have_below = int((E_store <= E_CUT).sum())
            print(f"[weyl] have_below={have_below}, est≈{weyl_est:.1f}")
            if have_below >= 0.95*weyl_est and refines > 5:
                # heuristic: close enough; you can tighten to 0.99 if you prefer
                print("[stop] below-E_CUT coverage consistent with Weyl; stopping.")
                break

        if not sigmas:
            if refines >= MAX_REFINES:
                print("[stop] no more sigmas and max refines reached.")
                break
            # adaptive refinement: find largest gap below E_CUT and target its midpoint
            Es = np.sort(E_store[E_store <= max(E_CUT, np.inf)])
            if Es.size == 0:
                sigmas.append(0.0)
            else:
                gaps = np.diff(Es)
                j = int(np.argmax(gaps))
                s_mid = float(0.5*(Es[j] + Es[j+1])) if j+1 < Es.size else float(Es[-1] + 0.25)
                sigmas.append(s_mid)
                refines += 1
            continue

        sigma = float(sigmas.pop(0))
        print(f"\n[solve] sigma={sigma:.6f}, K_BATCH={K_BATCH}")
        try:
            E_new, V_new = solve_near_sigma(H, sigma=sigma, k=K_BATCH, tol=ARPACK_TOL)
        except Exception as ex:
            print(f"[warn] eigsh failed at sigma={sigma}: {ex}")
            continue

        # Dedup vs existing
        keep_mask, dups = dedup_merge(E_store, sel_idx, E_new, V_new, read_vecs, tol_E=TOL_E, tol_corr=TOL_CORR)
        E_keep = E_new[keep_mask]
        V_keep = V_new[:, keep_mask]
        if E_keep.size == 0:
            print(f"[info] nothing new at sigma={sigma} (dups={dups}).")
        else:
            # keep only finite / positive
            ok = np.isfinite(E_keep) & (E_keep > 0)
            E_keep = E_keep[ok]; V_keep = V_keep[:, ok]

        if E_keep.size:
            # sort kept by energy ascending
            order = np.argsort(E_keep)
            E_keep = E_keep[order]; V_keep = V_keep[:, order]
            h5_append(f, E_keep, V_keep)
            f.flush()
            E_store = np.array(f["evals"][:])
            K_store = E_store.size
            sel_idx = update_sel_idx(E_store)
            new_total += E_keep.size
            print(f"[store] +{E_keep.size} (now K={K_store}); min={E_store.min():.6g}, max={E_store.max():.6g}")
            if (new_total // SAVE_THIN) != ((new_total - E_keep.size) // SAVE_THIN):
                flush_plot()

        # Push next sigma to march upward (overlap windows)
        if E_keep.size:
            s_next = float(min(E_keep.max() + 0.15, max(E_keep.max(), sigma) + 0.25))
        else:
            s_next = float(sigma + 0.25)
        if (E_CUT is None) or (s_next <= E_CUT + 0.75):
            sigmas.append(s_next)

    # Final sort by energy (global)
    print("[final] sorting store by energy …")
    E_all = np.array(f["evals"][:])
    order = np.argsort(E_all)
    # reorder evecs without loading everything at once
    evecs = f["evecs"][:]  # careful: may be large, but K below ~12k is OK on most machines
    evecs = evecs[:, order]
    f["evals"][:] = E_all[order]
    f["evecs"][:] = evecs
    f.flush(); f.close()

    # Save a quick coverage plot vs E
    E_sorted = np.sort(E_all)
    plt.figure(figsize=(6,3), dpi=140)
    plt.plot(np.arange(E_sorted.size), E_sorted)
    plt.axhline(E_CUT, color='r', ls='--', lw=1)
    plt.xlabel("#"); plt.ylabel("E"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "final_E_sorted.png"), bbox_inches="tight")
    plt.close()

    print(f"[done] K={E_sorted.size} stored; E_min={E_sorted.min():.6g}, E_max={E_sorted.max():.6g}")
    print(f"Elapsed {time.time()-t0:.1f} s -> {OUT_DIR}")

if __name__ == "__main__":
    start=time.time()
    main()
    end=time.time()
    print(f"Finished in {end-start} seconds")
