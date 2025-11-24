#!/usr/bin/env python3
# filter_states_by_density_enrichment.py
#
# Read the per‑state region density CSV (from region_density_by_state*.py)
# and find all eigenstates whose region density is enriched above a chosen
# multiple of the *uniform* baseline (i.e., ratio_vs_uniform_* > CUTOFF_MULT).
#
# Reports:
#   1) Percent of states with ANY region above cutoff.
#   2) Among those "special" states, the share that triggered in SMALL / CHANNEL / LARGE.
#      By default this is **non‑exclusive** (a state can count toward multiple regions).
#   3) Also shows an **exclusive** breakdown by the region with the largest ratio
#      (ARGMAX), which sums to ~100% by construction.
#
# It also saves a CSV of the matching state indices, with which region(s) exceeded.
#
# --------------------- CONFIG: edit here ---------------------
CONFIG = {
    "CSV_PATH": r"eigensdf/doublewell/potential9/trial1/region_avg_densities_by_state.csv",
    "OUT_DIR":  r"eigensdf/doublewell/potential9/trial1",
    "CUTOFF_MULT": 1.3,          # e.g. 1.5 means 50% above uniform baseline
    "CLASSIFY_MODE": "nonexclusive",  # "nonexclusive" or "argmax" (sets which headline to print first)
    "SAVE_MATCHES_CSV": True,
}
# ------------------------------------------------------------

import os, csv, json

REGIONS = ("small", "channel", "large")
RATIO_COLS = {
    "small":  "ratio_vs_uniform_small",
    "channel":"ratio_vs_uniform_channel",
    "large":  "ratio_vs_uniform_large",
}

def load_rows(csv_path):
    with open(csv_path, "r", newline="") as fp:
        r = csv.DictReader(fp)
        for row in r:
            yield row

def to_float(s, default=0.0):
    try:
        return float(s)
    except Exception:
        return default

def main():
    cfg = CONFIG
    csv_path = cfg["CSV_PATH"]
    out_dir  = cfg["OUT_DIR"]
    cutoff   = float(cfg["CUTOFF_MULT"])
    mode     = cfg["CLASSIFY_MODE"].lower()
    os.makedirs(out_dir, exist_ok=True)

    total_rows = 0
    special_rows = 0

    # nonexclusive tallies (how many special states included each region)
    hits = {reg: 0 for reg in REGIONS}

    # argmax exclusive tallies (which region had the largest ratio in that state)
    argmax_hits = {reg: 0 for reg in REGIONS}

    # store matches for CSV
    matches = []  # (index, energy, hit_small, hit_channel, hit_large, argmax_region)

    # Iterate CSV
    for row in load_rows(csv_path):
        total_rows += 1
        idx = row.get("index")
        energy = row.get("energy")

        ratios = {reg: to_float(row.get(RATIO_COLS[reg], "0")) for reg in REGIONS}
        flags = {reg: (ratios[reg] > cutoff) for reg in REGIONS}

        if any(flags.values()):
            special_rows += 1

            # nonexclusive count
            for reg in REGIONS:
                if flags[reg]:
                    hits[reg] += 1

            # argmax label
            max_region = max(REGIONS, key=lambda r: ratios[r])
            argmax_hits[max_region] += 1

            matches.append((idx, energy, int(flags["small"]), int(flags["channel"]), int(flags["large"]), max_region))

    # Percentages
    pct_special = (100.0 * special_rows / total_rows) if total_rows > 0 else 0.0

    # Nonexclusive shares among special states (note: may sum >100%)
    if special_rows > 0:
        pct_small_ne = 100.0 * hits["small"]   / special_rows
        pct_chan_ne  = 100.0 * hits["channel"] / special_rows
        pct_large_ne = 100.0 * hits["large"]   / special_rows
    else:
        pct_small_ne = pct_chan_ne = pct_large_ne = 0.0

    # Exclusive (argmax) shares among special states — these do sum to ~100%
    if special_rows > 0:
        pct_small_ax = 100.0 * argmax_hits["small"]   / special_rows
        pct_chan_ax  = 100.0 * argmax_hits["channel"] / special_rows
        pct_large_ax = 100.0 * argmax_hits["large"]   / special_rows
    else:
        pct_small_ax = pct_chan_ax = pct_large_ax = 0.0

    # Print summary
    print("=== Enrichment scan ===")
    print(f"CSV        : {csv_path}")
    print(f"Cutoff mult: {cutoff:.3f} (ratio_vs_uniform > cutoff)")
    print(f"Total states           : {total_rows}")
    print(f"States above cutoff    : {special_rows}  ({pct_special:.2f}%)")

    if mode == "argmax":
        print("\n-- Among states above cutoff (EXCLUSIVE by ARGMAX; sums to ~100%) --")
        print(f"small   : {argmax_hits['small']:6d}  ({pct_small_ax:6.2f}%)")
        print(f"channel : {argmax_hits['channel']:6d}  ({pct_chan_ax:6.2f}%)")
        print(f"large   : {argmax_hits['large']:6d}  ({pct_large_ax:6.2f}%)")

        print("\n-- Also (NON‑EXCLUSIVE; can sum >100%) --")
        print(f"small   : {hits['small']:6d}  ({pct_small_ne:6.2f}%)")
        print(f"channel : {hits['channel']:6d}  ({pct_chan_ne:6.2f}%)")
        print(f"large   : {hits['large']:6d}  ({pct_large_ne:6.2f}%)")
    else:
        print("\n-- Among states above cutoff (NON‑EXCLUSIVE; can sum >100%) --")
        print(f"small   : {hits['small']:6d}  ({pct_small_ne:6.2f}%)")
        print(f"channel : {hits['channel']:6d}  ({pct_chan_ne:6.2f}%)")
        print(f"large   : {hits['large']:6d}  ({pct_large_ne:6.2f}%)")

        print("\n-- Also (EXCLUSIVE by ARGMAX; sums to ~100%) --")
        print(f"small   : {argmax_hits['small']:6d}  ({pct_small_ax:6.2f}%)")
        print(f"channel : {argmax_hits['channel']:6d}  ({pct_chan_ax:6.2f}%)")
        print(f"large   : {argmax_hits['large']:6d}  ({pct_large_ax:6.2f}%)")

    # Save JSON
    summary = {
        "csv": csv_path,
        "cutoff_mult": cutoff,
        "classify_mode": mode,
        "total_states": total_rows,
        "above_cutoff": special_rows,
        "percent_above_cutoff": pct_special,
        "nonexclusive_counts": hits,
        "nonexclusive_percentages": {
            "small": pct_small_ne,
            "channel": pct_chan_ne,
            "large": pct_large_ne,
        },
        "argmax_counts": argmax_hits,
        "argmax_percentages": {
            "small": pct_small_ax,
            "channel": pct_chan_ax,
            "large": pct_large_ax,
        },
    }
    js_path = os.path.join(out_dir, "enrichment_summary.json")
    with open(js_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\n[Saved] JSON summary -> {js_path}")

    # Save matches CSV
    if CONFIG.get("SAVE_MATCHES_CSV", True):
        out_csv = os.path.join(out_dir, "enriched_state_indices.csv")
        with open(out_csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["index", "energy", "hit_small", "hit_channel", "hit_large", "argmax_region"])
            for rec in matches:
                w.writerow(rec)
        print(f"[Saved] matches CSV -> {out_csv}")

if __name__ == "__main__":
    main()
