#!/usr/bin/env python3
"""
extract_sorted_subset_h5.py

Load an eigenpairs HDF5 database, select all eigenstates with energy <= E_CUT,
sort them by energy, and save into a new HDF5 database.

Works with:
  - eigenpairs.h5 (real evals)
  - eigenpairs_complex.h5 (complex evals; uses Re(E) for cutoff/sorting by default)

This copies evecs in chunks to avoid loading everything into RAM.
"""

import os
import json
import time
import numpy as np
import h5py


# ========= USER CONFIG (edit these) =========
IN_H5_PATH = r"eigensdf/doublewell/final2/eigenpairs.h5"
E_CUT = 3.0

# For complex evals: cutoff/sort key
USE_REAL_PART = True  # True -> use Re(E), False -> use |E|

# Output
OUT_H5_PATH = None  # None -> next to input: <basename>_E<=X_sorted.h5
COPY_GEOMETRY_USED = True  # if geometry_used.npz exists next to input, copy it too
WRITE_MANIFEST = True

# Performance
COPY_CHUNK = 256  # number of eigenvectors (columns) per read/write chunk
# ===========================================


def _energy_key(evals: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(evals):
        return np.real(evals) if USE_REAL_PART else np.abs(evals)
    return np.asarray(evals, dtype=np.float64)


def main() -> None:
    t0 = time.time()

    in_path = IN_H5_PATH
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    out_path = OUT_H5_PATH
    if out_path is None:
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_dir = os.path.dirname(os.path.abspath(in_path))
        # Windows filesystem does not allow characters like '<' or '>' in filenames.
        # Use a safe tag like "E_le_3" instead of "E<=3".
        tag = f"E_le_{float(E_CUT):g}"
        # sanitize any remaining problematic characters just in case
        safe = []
        for ch in tag:
            if ch in '<>:"/\\|?*':
                safe.append("_")
            else:
                safe.append(ch)
        tag = "".join(safe)
        out_path = os.path.join(out_dir, f"{base}_{tag}_sorted.h5")

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    # --- load evals and decide indices ---
    with h5py.File(in_path, "r") as h5:
        if "evals" not in h5 or "evecs" not in h5:
            raise KeyError("Input H5 must contain datasets 'evals' and 'evecs'.")
        evals = np.array(h5["evals"][:])
        n = int(h5["evecs"].shape[0])
        K = int(h5["evecs"].shape[1])

    key = _energy_key(evals)
    finite = np.isfinite(key)
    sel = finite & (key <= float(E_CUT))
    idx = np.where(sel)[0].astype(np.int64)
    if idx.size == 0:
        raise RuntimeError(f"No eigenvalues found with key(E) <= {E_CUT}.")

    # sort selected indices by key
    order = idx[np.argsort(key[idx])]
    K_out = int(order.size)

    print(f"[in]  {in_path}")
    print(f"[out] {out_path}")
    print(f"[shape] n={n}, K_in={K}")
    print(f"[select] key(E) <= {E_CUT}: K_out={K_out}")

    # --- create output H5 ---
    # Overwrite if exists
    if os.path.exists(out_path):
        os.remove(out_path)

    with h5py.File(in_path, "r") as src, h5py.File(out_path, "w") as dst:
        # Dtypes
        evals_dtype = src["evals"].dtype
        evecs_dtype = src["evecs"].dtype

        dst.create_dataset("evals", shape=(K_out,), dtype=evals_dtype)
        dst.create_dataset(
            "evecs",
            shape=(n, K_out),
            dtype=evecs_dtype,
            chunks=(min(n, 8192), 1),
        )
        dst.create_dataset("orig_indices", data=order.astype(np.int64))

        # write evals in sorted order
        dst["evals"][:] = evals[order].astype(evals_dtype, copy=False)

        # copy evecs in chunks; h5py requires increasing indices when fancy-indexing
        chunk = max(int(COPY_CHUNK), 1)
        for start in range(0, K_out, chunk):
            stop = min(start + chunk, K_out)
            idxs = order[start:stop]

            # read from src using increasing indices
            sort_perm = np.argsort(idxs)
            idxs_sorted = idxs[sort_perm]
            V_sorted = np.array(src["evecs"][:, idxs_sorted], dtype=evecs_dtype)
            inv_perm = np.argsort(sort_perm)
            V = V_sorted[:, inv_perm]

            dst["evecs"][:, start:stop] = V

            if (start == 0) or (stop == K_out) or ((start // chunk) % 10 == 0):
                print(f"[copy] {stop}/{K_out} eigenvectors")

        dst.flush()

    # --- copy geometry_used.npz (optional) ---
    geom_src = os.path.join(os.path.dirname(os.path.abspath(in_path)), "geometry_used.npz")
    geom_dst = os.path.join(out_dir, "geometry_used.npz")
    if COPY_GEOMETRY_USED and os.path.exists(geom_src):
        # simple byte copy
        with open(geom_src, "rb") as fsrc, open(geom_dst, "wb") as fdst:
            fdst.write(fsrc.read())
        print(f"[copy] geometry_used.npz -> {geom_dst}")

    # --- manifest (optional) ---
    if WRITE_MANIFEST:
        man = {
            "note": "extract_sorted_subset_h5",
            "in_h5": os.path.abspath(in_path),
            "out_h5": os.path.abspath(out_path),
            "E_CUT": float(E_CUT),
            "USE_REAL_PART": bool(USE_REAL_PART),
            "K_in": int(K),
            "K_out": int(K_out),
            "time_sec": float(time.time() - t0),
        }
        man_path = os.path.join(out_dir, "manifest_extract_sorted_subset.json")
        with open(man_path, "w") as f:
            json.dump(man, f, indent=2)
        print(f"[save] {man_path}")

    print(f"[done] wrote {K_out} modes in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()


