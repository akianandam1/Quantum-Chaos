# import numpy as np, matplotlib.pyplot as plt, csv
# opath ="videos/potential4/trial2/standing_spectrum.csv"  # adjust tag if using "gaussian"
# omega, Smag = np.loadtxt(opath, delimiter=",", skiprows=1, unpack=True)
# plt.plot(omega, Smag)
# plt.xlabel("ω (≈ energy)"); plt.ylabel("|FT{C}| (normalized)")
# plt.title("Energy spectrum from Fourier transform of autocorrelation")
# plt.show()

import sys, os, csv
import numpy as np
import matplotlib.pyplot as plt

def main(csv_path):
    if not os.path.isfile(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")
    out_dir = os.path.dirname(csv_path)
    out_png = os.path.join(out_dir, "autocorr_abs.png")

    # Read header to find columns
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # normalize header names (strip spaces)
        cols = [h.strip() for h in header]

    # Helper to find column index by a few possible names
    def find_col(names):
        for name in names:
            if name in cols:
                return cols.index(name)
        return None

    idx_t   = find_col(["t", "time"])
    idx_abs = find_col(["|C|", "|C_aa|", "|C_aa| (abs)", "abs_C"])
    # If |C| not present, we’ll compute from Re and Im:
    idx_re  = find_col(["Re<C>", "Re<C_aa>", "Re<C_aa"])
    idx_im  = find_col(["Im<C>", "Im<C_aa>", "Im<C_aa"])

    if idx_t is None:
        raise SystemExit("Couldn't find time column ('t' or 'time') in CSV header.")
    if idx_abs is None and (idx_re is None or idx_im is None):
        raise SystemExit("CSV must contain either '|C|' or both 'Re<C>' and 'Im<C>'.")

    # Load data rows
    t_list, abs_list = [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row or len(row) <= idx_t:
                continue
            try:
                t = float(row[idx_t])
            except:
                continue
            if idx_abs is not None and len(row) > idx_abs and row[idx_abs] != "":
                val = float(row[idx_abs])
            else:
                # compute |C| = sqrt(Re^2 + Im^2)
                if len(row) <= max(idx_re, idx_im):
                    continue
                re = float(row[idx_re])
                im = float(row[idx_im])
                val = (re**2 + im**2) ** 0.5
            t_list.append(t)
            abs_list.append(val)

    if not t_list:
        raise SystemExit("No data rows parsed; check your CSV contents.")

    t = np.array(t_list, dtype=float)
    Cabs = np.array(abs_list, dtype=float)

    # Optional: light smoothing for visualization (comment out if not desired)
    # win = 5
    # if len(Cabs) > win:
    #     Cabs = np.convolve(Cabs, np.ones(win)/win, mode="same")
    # t = t[:200]
    # Cabs = Cabs[:200]
    # Plot


    plt.figure(figsize=(7, 3.5), dpi=150)
    plt.plot(t, Cabs, linewidth=1.2)
    plt.xlabel("time t")
    plt.ylabel(r"$|C(t)| = |\langle \psi(0) | \psi(t) \rangle|$")
    plt.title("Autocorrelation magnitude vs time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_png}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_autocorr_abs.py <path/to/autocorr.csv>")
        sys.exit(1)
    main(sys.argv[1])