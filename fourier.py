# analyze_autocorr_solved.py
# Full replacement. Edit CONFIG only.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# ========= CONFIG ========
# =========================
CSV_PATH = "eigensdf/trial12/cn_autocorr.csv"   # CSV with columns: t, ReC, ImC
T_MIN    = 100                  # analysis window start
T_MAX    = 200.0                # analysis window end (use early times for clean phase)
XMAX     = 4.0                  # x-axis limit for ω plots
USE_DISPLAY_HIGHPASS = True     # zero out very-low-ω in the display spectrum
W_CUT    = 1.1                  # display high-pass cutoff (rad/time), e.g., 0.5–1.2
DO_DENSE_SCAN = True            # run dense scan around Ė to reveal nearby lines
DENSE_WIN    = 3.0              # scan Ė ± DENSE_WIN
DENSE_DW     = 0.0005           # ω step for dense scan
SHOW_DERIVATIVE_FFT = True      # optional: FFT of dC/dt (suppresses DC/envelope)
# =========================

OUT_DIR = os.path.dirname(CSV_PATH) or "."

def _pick_column(df, candidates, fallback_idx=None):
    cols = {str(c).strip().lower(): c for c in df.columns}
    for key in candidates:
        if key in cols:
            return cols[key]
    if fallback_idx is not None and fallback_idx < len(df.columns):
        return df.columns[fallback_idx]
    raise KeyError(f"Missing any of {candidates}. CSV has {list(df.columns)}")

def _norm_ignore_zero(y):
    y = np.asarray(y, float).copy()
    if y.size > 1 and np.max(y[1:]) > 0:
        y /= np.max(y[1:])
    elif y.size and y.max() > 0:
        y /= y.max()
    return y

def _smooth_envelope(a, frac=0.02):
    n = len(a)
    win = max(5, int(max(3, round(frac * n))))
    if win % 2 == 0:
        win += 1
    w = np.hanning(win); w /= w.sum()
    env = np.convolve(np.abs(a), w, mode="same")
    return np.maximum(env, 1e-12)

def main():
    # ---- Load CSV ----
    df = pd.read_csv(CSV_PATH)
    t_col  = _pick_column(df, ["t","time","time (s)"], 0)
    re_col = _pick_column(df, ["rec","re","real","real(c)"], 1)
    im_col = _pick_column(df, ["imc","im","imag","imag(c)"], 2)

    t  = df[t_col].to_numpy(float)
    Re = df[re_col].to_numpy(float)
    Im = df[im_col].to_numpy(float)
    Ct_full = (Re + 1j*Im).astype(np.complex128)

    # ---- Window selection ----
    m = (t >= T_MIN) & (t <= T_MAX)
    t  = t[m]
    Ct = Ct_full[m]
    if t.size < 64:
        raise ValueError(f"Selected window too short: N={t.size}")

    # ---- Sampling ----
    dt_vals = np.diff(t)
    dt = float(np.median(dt_vals))
    if np.max(np.abs(dt_vals - dt)) > 1e-6 * max(1.0, abs(dt)):
        print("[warn] time spacing slightly non-uniform; using median dt.")
    N = t.size
    nyq = np.pi / dt
    domega_fft = 2*np.pi / (N*dt)
    print(f"[info] N={N}, dt={dt:.6g}, Nyquist ω≈{nyq:.3f}, FFT Δω≈{domega_fft:.6f}")

    # Normalize phase reference (helps numerics)
    if Ct[0] != 0:
        Ct = Ct / Ct[0]

    # ---- Sanity plot: Re/Im/|C| + unwrapped phase ----
    phi = np.unwrap(np.angle(Ct))
    plt.figure(figsize=(10,5))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(t, Ct.real, lw=1.0, label="Re C")
    ax1.plot(t, Ct.imag, lw=1.0, label="Im C")
    ax1.plot(t, np.abs(Ct), lw=1.0, alpha=0.8, label="|C|")
    ax1.legend(); ax1.set_ylabel("C components")
    ax2 = plt.subplot(2,1,2)
    ax2.plot(t, phi, lw=1.0)
    ax2.set_xlabel("t"); ax2.set_ylabel("unwrapped phase")
    ax2.set_title("Early-window sanity check")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cn_C_early_sanity.png"), dpi=150)
    plt.close()

    # ---- Weighted phase-slope energy estimate  Ė ----
    w_amp = (np.abs(Ct)**2).astype(float)
    A = np.vstack([np.ones_like(t), t]).T
    W = np.sqrt(w_amp)
    Aw = A * W[:, None]
    phiw = phi * W
    a_hat, b_hat = np.linalg.lstsq(Aw, phiw, rcond=None)[0]
    E_hat = -b_hat
    print(f"[info] weighted phase slope -> E_hat ≈ {E_hat:.6f} rad/time")

    # ---- Complex FFT of C(t) (with display-only high-pass) ----
    taper = np.hanning(N).astype(float)
    CtW = Ct * taper
    Nfft  = 1 << (N - 1).bit_length()
    Spec  = np.fft.fft(CtW, n=Nfft)
    P     = np.abs(Spec)**2
    omega = 2*np.pi*np.fft.fftfreq(Nfft, d=dt)
    pos   = omega >= 0
    om_p  = omega[pos]
    P_p   = P[pos]

    if USE_DISPLAY_HIGHPASS:
        P_plot = P_p.copy()
        P_plot[om_p < W_CUT] = 0.0
    else:
        P_plot = P_p.copy()
    P_plot = _norm_ignore_zero(P_plot)

    # save + plot
    view = (om_p >= 0) & (om_p <= XMAX)
    np.savetxt(os.path.join(OUT_DIR, "cn_fft_C_complex.csv"),
               np.column_stack([om_p[view], P_plot[view]]),
               delimiter=",", header="omega,P_norm(display)", comments="")
    plt.figure(figsize=(10,3.2))
    plt.plot(om_p, P_plot, lw=1.2)
    plt.axvline(E_hat, ls="--", lw=1.0, alpha=0.9, label=fr"$\hat E \approx {E_hat:.3f}$")
    plt.xlim(0, XMAX)
    plt.xlabel(r"$\omega$ (≈ Energy)"); plt.ylabel("Power (norm.)")
    plt.title(fr"FFT of $C(t)$, window {T_MIN:g}–{T_MAX:g}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cn_fft_C_complex.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # log-scale view
    plt.figure(figsize=(10,3.2))
    plt.semilogy(om_p, np.maximum(P_plot, 1e-16), lw=1.2)
    plt.axvline(E_hat, ls="--", lw=1.0, alpha=0.9, label=fr"$\hat E \approx {E_hat:.3f}$")
    plt.xlim(0, XMAX)
    plt.xlabel(r"$\omega$"); plt.ylabel("Power (log)")
    plt.title("FFT of $C(t)$ (log scale)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cn_fft_C_complex_log.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Demodulate by Ė, de-envelope, FFT (peak should be at 0; side peaks = other energies) ----
    Ct_demod = Ct * np.exp(1j * E_hat * t)         # carrier -> 0
    env = _smooth_envelope(Ct_demod, frac=0.02)    # remove slow amplitude
    Ct_flat = Ct_demod / env
    X = Ct_flat * taper

    Spec2 = np.fft.fft(X, n=Nfft)
    P2    = np.abs(Spec2)**2
    P2_p  = P2[pos]
    P2_plot = _norm_ignore_zero(P2_p)

    np.savetxt(os.path.join(OUT_DIR, "cn_fft_demod_deenv.csv"),
               np.column_stack([om_p[view], P2_plot[view]]),
               delimiter=",", header="omega,P_norm_demod_deenv", comments="")
    plt.figure(figsize=(10,3.2))
    plt.plot(om_p, P2_plot, lw=1.2)
    plt.xlim(0, 3.0)  # look near 0 after demod
    plt.xlabel(r"$\omega$ after mix by $\hat E$")
    plt.ylabel("Power (norm.)")
    plt.title(r"FFT of $C(t)\,e^{+i\hat E t}$ after de-envelope")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cn_fft_demod_deenv.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Dense scan around Ė to list actual lines ----
    if DO_DENSE_SCAN:
        d = np.arange(-DENSE_WIN, DENSE_WIN + 0.5*DENSE_DW, DENSE_DW)  # offsets around 0
        t0 = t[0]
        # use demod+de-envelope here so carrier=0 and envelope suppressed
        Y = Ct_flat * taper
        E = np.exp(-1j * np.outer(d, (t - t0)))
        Sd = np.abs(E @ Y)**2
        Sd = _norm_ignore_zero(Sd)

        # peak report: top offsets and absolute energies
        k0 = 1 if d.size > 1 else 0
        order = np.argsort(Sd[k0:])[-6:][::-1] + k0
        offsets = d[order]
        E_lines = E_hat + offsets
        print("[lines] Energy offsets δ (rad/time):", np.round(offsets, 6))
        print("[lines] Absolute energies E = Ė + δ :", np.round(E_lines, 6))

        # save + plot dense scan
        np.savetxt(os.path.join(OUT_DIR, "cn_dense_scan_offsets.csv"),
                   np.column_stack([d, Sd]),
                   delimiter=",", header="delta_omega,Power_norm", comments="")
        plt.figure(figsize=(10,3.2))
        plt.plot(d, Sd, lw=1.2)
        plt.axvline(0.0, ls="--", lw=1.0, alpha=0.9, label="carrier (Ė)")
        plt.xlabel(r"offset δ (so E = Ė + δ)")
        plt.ylabel("Power (norm.)")
        plt.title(r"Dense scan around Ė (demod + de-envelope)")
        plt.xlim(-DENSE_WIN, DENSE_WIN)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "cn_dense_scan_offsets.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ---- Optional: derivative-FFT (suppresses DC/envelope) ----
    if SHOW_DERIVATIVE_FFT:
        dCt = np.gradient(Ct, dt)              # ~ dC/dt
        dCtW = dCt * taper
        Specd = np.fft.fft(dCtW, n=Nfft)
        Pd = np.abs(Specd)**2
        Pd_p = Pd[pos]
        # compensate by ω to compare shape (avoid divide-by-zero at 0)
        om_safe = om_p.copy()
        om_safe[om_safe == 0] = 1.0
        Pd_comp = Pd_p / (om_safe**2)
        Pd_plot = _norm_ignore_zero(Pd_comp)
        np.savetxt(os.path.join(OUT_DIR, "cn_fft_derivative.csv"),
                   np.column_stack([om_p[view], Pd_plot[view]]),
                   delimiter=",", header="omega,P_norm_dCdt(compensated)", comments="")
        plt.figure(figsize=(10,3.2))
        plt.plot(om_p, Pd_plot, lw=1.2)
        plt.axvline(E_hat, ls="--", lw=1.0, alpha=0.9, label=fr"$\hat E \approx {E_hat:.3f}$")
        plt.xlim(0, XMAX)
        plt.xlabel(r"$\omega$ (≈ Energy)")
        plt.ylabel("Power (norm.)")
        plt.title("FFT of dC/dt (DC suppressed; compensated by $\\omega^2$)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "cn_fft_dCdt.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print("Done. Outputs written to:", os.path.abspath(OUT_DIR))

if __name__ == "__main__":
    main()