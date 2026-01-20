#!/usr/bin/env python3
"""
sort_eigenpairs_by_energy.py

Reads an existing eigenpairs.h5 file (datasets: evals [K], evecs [n,K]),
sorts the eigenpairs by energy (ascending), and writes a NEW HDF5 file
containing only the lowest MAX_STATES eigenpairs.

The original file is never modified.

User parameters are in the CONFIG dict below.
"""

import os
import numpy as np
import h5py

# ===================== USER CONFIG =====================

CONFIG = {
    # Path to the original eigenpairs file (will be opened read-only)
    "H5_IN_PATH": r"eigensdf/doublewell/trial2/eigenpairs.h5",

    # Path for the new, sorted eigenpairs file
    "H5_OUT_PATH": r"eigensdf/doublewell/trial2/eigenpairs_sorted_15000.h5",

    # How many lowest-energy eigenstates to keep
    "MAX_STATES": 15000,

    # If the output file already exists:
    #   - if True: overwrite it
    #   - if False: abort with an error
    "OVERWRITE_OUT": False,
}

# =======================================================


def main():
    cfg = CONFIG
    h5_in_path = cfg["H5_IN_PATH"]
    h5_out_path = cfg["H5_OUT_PATH"]
    max_states = int(cfg["MAX_STATES"])
    overwrite_out = bool(cfg["OVERWRITE_OUT"])

    if not os.path.exists(h5_in_path):
        raise FileNotFoundError(f"H5_IN_PATH not found: {h5_in_path}")

    if os.path.exists(h5_out_path) and not overwrite_out:
        raise FileExistsError(
            f"H5_OUT_PATH already exists: {h5_out_path}\n"
            f"Set OVERWRITE_OUT=True in CONFIG if you want to overwrite it."
        )

    print(f"[info] Opening input file (read-only): {h5_in_path}")
    with h5py.File(h5_in_path, "r") as fin:
        # --- Read evals and basic shapes ---
        if "evals" not in fin or "evecs" not in fin:
            raise KeyError("Input file must contain datasets 'evals' and 'evecs'.")

        evals = np.array(fin["evals"][:], dtype=np.float64)
        evecs_ds = fin["evecs"]

        if evecs_ds.shape[1] != evals.shape[0]:
            raise ValueError(
                f"Shape mismatch: evecs.shape={evecs_ds.shape}, "
                f"evals.shape={evals.shape}"
            )

        n, K_total = evecs_ds.shape
        print(f"[info] Found {K_total} eigenvalues, matrix size n={n}")

        # --- Handle NaNs / Infs just in case ---
        finite_mask = np.isfinite(evals)
        if not finite_mask.all():
            n_bad = np.count_nonzero(~finite_mask)
            print(f"[warn] {n_bad} non-finite eigenvalues detected; ignoring those.")
            idx_good = np.where(finite_mask)[0]
            evals_good = evals[idx_good]
        else:
            idx_good = np.arange(K_total, dtype=int)
            evals_good = evals

        # --- Sort by energy (ascending) among the good ones ---
        sort_order = np.argsort(evals_good)
        idx_sorted_all = idx_good[sort_order]

        # number of states to keep
        M = min(max_states, idx_sorted_all.size)
        idx_sorted = idx_sorted_all[:M]
        evals_sorted = evals[idx_sorted]

        print(f"[info] Keeping {M} lowest-energy states out of {K_total}.")
        print(f"[info] Lowest energies (first 5): {evals_sorted[:5]}")
        if M > 5:
            print(f"[info] Highest of kept energies: {evals_sorted[-1]}")

        # --- Create output file and datasets ---
        if os.path.exists(h5_out_path) and overwrite_out:
            print(f"[warn] Overwriting existing output file: {h5_out_path}")
            os.remove(h5_out_path)

        os.makedirs(os.path.dirname(h5_out_path), exist_ok=True)
        print(f"[info] Creating output file: {h5_out_path}")

        with h5py.File(h5_out_path, "w") as fout:
            # evals: 1D array [M]
            d_evals = fout.create_dataset(
                "evals",
                shape=(M,),
                dtype=np.float64
            )
            d_evals[:] = evals_sorted

            # evecs: 2D array [n, M], same dtype as input evecs
            d_evecs = fout.create_dataset(
                "evecs",
                shape=(n, M),
                dtype=evecs_ds.dtype,
                chunks=(min(n, 8192), 1),
            )

            # --- Copy eigenvectors one column at a time to avoid huge RAM usage ---
            print("[info] Copying eigenvectors in sorted order...")
            for j_new, j_old in enumerate(idx_sorted):
                if j_new % 100 == 0:
                    print(f"  [copy] column {j_new}/{M} (source index {j_old})")
                # read column j_old from input and write to column j_new in output
                d_evecs[:, j_new] = evecs_ds[:, j_old]

            print("[info] Done writing sorted eigenpairs.")

    print("[info] Finished. Original file was NOT modified.")


if __name__ == "__main__":
    main()
