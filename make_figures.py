"""
Regenerates every figure from data produced by this repository (nothing hard-coded).

  python nist_validate.py gen && python nist_validate.py test   # produces results/
  python make_figures.py                                        # produces figures/

Fig1-3  noise model illustration (simulated Gaussian detector noise, see README)
Fig4-7  phase / balance / autocorrelation / attractor of the HGPhi pipeline
Fig8-12 NIST SP 800-22 results (chaos vs chaos-OFF control vs numpy reference)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hgphi_core as h

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
STREAMS = ["chaos", "control", "numpy"]
LABEL = {"chaos": "HG$\\Phi$-QRNG (chaos on)", "control": "Control (chaos off)",
         "numpy": "numpy PCG64 reference"}
COL = {"chaos": "#2166AC", "control": "#E66101", "numpy": "#1B7837"}
RED = "#B2182B"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11, "axes.labelsize": 13, "legend.fontsize": 9, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.2, "mathtext.fontset": "cm",
})


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved", name)


def load_bits(stream):
    meta = json.load(open(f"{RES}/gen_meta.json"))
    P = np.load(f"{RES}/seqs_{stream}.npy")
    return np.unpackbits(P, axis=1)[:, :meta["n_bits"]]


def acf(b, lags):
    x = b.astype(np.float64) - b.mean()
    v = x.var()
    return np.array([np.dot(x[:-l], x[l:]) / (len(x) - l) / v for l in lags])


summary = json.load(open(f"{RES}/nist_summary.json"))
bits = {s: load_bits(s) for s in STREAMS}
n_seq, n_bits = bits["chaos"].shape
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- metrics (all real)
LAGS = np.array([1, 2, 5, 10, 20, 50])
metrics = {}
for s in STREAMS:
    allb = bits[s]
    p1 = float(allb.mean())
    ac_seq = np.array([acf(allb[i], LAGS) for i in range(n_seq)])
    ac_pool = ac_seq.mean(axis=0)
    metrics[s] = dict(
        n_bits_total=int(allb.size), balance=p1,
        shannon=float(-(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1))),
        min_entropy=float(-np.log2(max(p1, 1 - p1))),
        ac_lags=LAGS.tolist(), ac_mean=ac_pool.tolist(),
        ac_rms=np.sqrt((ac_seq ** 2).mean(axis=0)).tolist(),
        ac1_per_seq=ac_seq[:, 0].tolist())
metrics["ci95_single_seq"] = float(1.96 / np.sqrt(n_bits))
json.dump(metrics, open(f"{RES}/metrics.json", "w"), indent=1)

# one extra trajectory for the phase / attractor panels
wm = h.window_means(h.initial_states(1, 6, 1), 60000)
phases = wm[0].astype(np.float64)

# ---------------------------------------------------------------- Fig 1-3 noise model
sig_shot, sig_el, N = 0.18, 0.035, 500_000
q = lambda x, nb, V=1.0: np.round(np.clip(x, -V, V - 2 * V / 2 ** nb) / (2 * V / 2 ** nb)) * (2 * V / 2 ** nb)
for name, sig, col, xl, bins in [("Fig1_shot_noise_distribution", sig_shot, "#2166AC", 0.65, 300),
                                 ("Fig2_electronic_noise_distribution", sig_el, RED, 0.14, 150)]:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    raw = rng.normal(0, sig, N)
    for nb, ls in [(12, "--"), (16, ":")]:
        hh, e = np.histogram(q(raw, nb), bins=bins * (1 if nb == 12 else 2), density=True)
        ax.plot(0.5 * (e[:-1] + e[1:]), hh, ls, color=col, label=f"{nb}-bit ADC")
    x = np.linspace(-xl, xl, 2000)
    ax.plot(x, norm.pdf(x, 0, sig), "k-", alpha=0.6, label="Gaussian model")
    ax.set_xlim(-xl, xl); ax.set_ylim(bottom=0)
    ax.set_xlabel("Simulated detector output (V)"); ax.set_ylabel("Probability density")
    ax.legend(loc="upper left")
    save(fig, name)
fig, ax = plt.subplots(figsize=(9, 5.5))
for sig, col, lab in [(sig_shot, "#2166AC", "Shot-noise model"), (sig_el, RED, "Electronic-noise model")]:
    hh, e = np.histogram(q(rng.normal(0, sig, N), 16), bins=500, density=True)
    c = 0.5 * (e[:-1] + e[1:])
    ax.fill_between(c, hh, alpha=0.25, color=col); ax.plot(c, hh, color=col, label=f"{lab} ($\\sigma$ = {sig} V)")
ax.set_xlim(-0.7, 0.7); ax.set_ylim(bottom=0)
ax.set_xlabel("Simulated detector output (V)"); ax.set_ylabel("Probability density"); ax.legend()
ax.text(0.03, 0.9, f"$20\\log_{{10}}(\\sigma_{{shot}}/\\sigma_{{elec}})$ = {20*np.log10(sig_shot/sig_el):.1f} dB",
        transform=ax.transAxes, bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))
save(fig, "Fig3_shot_vs_electronic_overlay")

# ---------------------------------------------------------------- Fig 4 phase
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.hist(phases, bins=200, density=True, color=COL["chaos"], alpha=0.7, edgecolor="white", lw=0.3)
a1.axhline(1 / (2 * np.pi), color=RED, ls="--", label="uniform on [0, 2$\\pi$)")
a1.set_xlabel("Geometric phase $\\gamma_{SB}$ (rad)"); a1.set_ylabel("Probability density")
a1.set_title("(a)", fontweight="bold"); a1.legend()
ac = acf(phases, np.arange(1, 51))
a2.bar(np.arange(1, 51), ac, color=COL["chaos"], alpha=0.7)
ci = 1.96 / np.sqrt(len(phases))
a2.axhline(ci, color=RED, ls="--", label="95% band (white noise)"); a2.axhline(-ci, color=RED, ls="--")
a2.set_xlabel("Lag (windows)"); a2.set_ylabel("Phase autocorrelation"); a2.set_title("(b)", fontweight="bold"); a2.legend()
save(fig, "Fig4_geometric_phase_distribution")

# ---------------------------------------------------------------- Fig 5 balance / entropy
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
b0 = bits["chaos"][0]
win = 10000
run = np.array([b0[i:i + win].mean() for i in range(0, len(b0) - win, win)])
x = np.arange(len(run)) * win / 1e3
a1.plot(x, run, color=COL["chaos"], lw=0.8)
a1.axhline(0.5, color=RED, ls="--", label="ideal 0.5")
a1.fill_between(x, 0.5 - 3 / np.sqrt(4 * win), 0.5 + 3 / np.sqrt(4 * win), color=RED, alpha=0.1, label="$\\pm 3\\sigma$ band")
a1.set_xlabel("Position ($\\times 10^3$ bits)"); a1.set_ylabel("Proportion of 1s"); a1.set_title("(a)", fontweight="bold"); a1.legend()
n8 = len(b0) // 8 * 8
cnt = np.bincount(np.packbits(b0[:n8]), minlength=256) / (n8 // 8)
a2.bar(np.arange(256), cnt, color=COL["chaos"], alpha=0.6, width=1)
a2.axhline(1 / 256, color=RED, ls="--", label="uniform = 1/256")
a2.set_xlabel("Byte value"); a2.set_ylabel("Probability"); a2.set_title("(b)", fontweight="bold"); a2.legend()
save(fig, "Fig5_bit_balance_entropy")

# ---------------------------------------------------------------- Fig 6 bit autocorrelation
fig, ax = plt.subplots(figsize=(9, 5))
lags = np.arange(1, 101)
ac_all = np.mean([acf(bits["chaos"][i], lags) for i in range(n_seq)], axis=0)
ax.bar(lags, ac_all, color=COL["chaos"], alpha=0.7)
ci = 1.96 / np.sqrt(bits["chaos"].size)
ax.axhline(ci, color=RED, ls="--", label=f"95% band for {n_seq} sequences ($\\pm${ci:.1e})"); ax.axhline(-ci, color=RED, ls="--")
ax.set_xlabel("Lag"); ax.set_ylabel("Autocorrelation (mean over sequences)"); ax.legend()
save(fig, "Fig6_bit_autocorrelation")

# ---------------------------------------------------------------- Fig 7 attractor
X = np.array(h.X0, dtype=float)
ip1, im1, im2 = (np.array([(i + k) % 6 for i in range(6)]) for k in (1, -1, -2))
att = np.empty((75000, 6))
for k in range(75000):
    X = X + h.DT * (h.A * X[ip1] - h.B * X[im2]) * X[im1]
    m = np.max(np.abs(X))
    if m > h.BOUND:
        X *= (h.BOUND - 0.01) / m
    att[k] = X
att = att[5000:]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (i, j), t in zip(axes, [(0, 1), (2, 3), (4, 5)], "abc"):
    ax.plot(att[::2, i], att[::2, j], ",", color=COL["chaos"], alpha=0.2)
    ax.set_xlabel(f"$X_{i+1}$"); ax.set_ylabel(f"$X_{j+1}$"); ax.set_title(f"({t})", fontweight="bold")
fig.tight_layout()
save(fig, "Fig7_hyperchaotic_attractor")

# ---------------------------------------------------------------- Fig 8 NIST proportions (chaos)
tests = list(summary["chaos"].keys())
fig, ax = plt.subplots(figsize=(11, 6.5))
y = np.arange(len(tests))
prop = [summary["chaos"][t]["passed"] / summary["chaos"][t]["n_pvalues"] for t in tests]
lo = [summary["chaos"][t]["min_required"] / summary["chaos"][t]["n_pvalues"] for t in tests]
ok = [summary["chaos"][t]["prop_ok"] and summary["chaos"][t]["unif_ok"] for t in tests]
ax.barh(y, prop, color=[COL["chaos"] if o else RED for o in ok], alpha=0.85, height=0.7)
ax.plot(lo, y, "|", color="k", ms=14, mew=2, label="minimum acceptable proportion")
for yi, t in zip(y, tests):
    d = summary["chaos"][t]
    ax.text(1.002, yi, f"{d['passed']}/{d['n_pvalues']}   $P_T$={d['P_T']:.3f}", va="center", fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(tests); ax.invert_yaxis()
ax.set_xlim(0.9, 1.0); ax.set_xlabel("Proportion of p-values $\\geq$ 0.01"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
ax.set_title(f"{n_seq} sequences $\\times$ {n_bits:,} bits", fontsize=11)
save(fig, "Fig8_NIST_pvalues_barchart")

# ---------------------------------------------------------------- Fig 9 grouped proportions
fig, ax = plt.subplots(figsize=(15, 6.5))
w = 0.27
for k, s in enumerate(STREAMS):
    v = [summary[s][t]["passed"] / summary[s][t]["n_pvalues"] for t in tests]
    ax.bar(np.arange(len(tests)) + (k - 1) * w, v, w, color=COL[s], alpha=0.85, label=LABEL[s])
ax.plot(np.arange(len(tests)), lo, "_", color="k", ms=30, mew=2, label="minimum acceptable")
ax.set_xticks(np.arange(len(tests))); ax.set_xticklabels(tests, rotation=40, ha="right")
ax.set_ylim(0.93, 1.005); ax.set_ylabel("Pass proportion ($\\alpha$ = 0.01)"); ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
fig.tight_layout()
save(fig, "Fig9_NIST_comparison_grouped_bars")

# ---------------------------------------------------------------- Fig 10 metrics
fig = plt.figure(figsize=(14, 5.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.3)
ax = fig.add_subplot(gs[0])
data = [np.array(metrics[s]["ac1_per_seq"]) for s in STREAMS]
bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True)
for p, s in zip(bp["boxes"], STREAMS):
    p.set_facecolor(COL[s]); p.set_alpha(0.6)
ax.axhline(metrics["ci95_single_seq"], color=RED, ls="--", label="95% band (one sequence)")
ax.axhline(-metrics["ci95_single_seq"], color=RED, ls="--")
ax.set_xticklabels(["chaos on", "chaos off", "numpy"]); ax.set_ylabel("lag-1 autocorrelation per sequence")
ax.set_title("(a)", fontweight="bold"); ax.legend()
ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
rows = [["Metric", "chaos on", "chaos off", "numpy"],
        ["Bits tested", *[f"{metrics[s]['n_bits_total']:.2e}" for s in STREAMS]],
        ["Balance", *[f"{metrics[s]['balance']:.6f}" for s in STREAMS]],
        ["Shannon entropy", *[f"{metrics[s]['shannon']:.8f}" for s in STREAMS]],
        ["Min-entropy", *[f"{metrics[s]['min_entropy']:.6f}" for s in STREAMS]],
        ["mean AC(1)", *[f"{metrics[s]['ac_mean'][0]:+.1e}" for s in STREAMS]],
        ["NIST tests passed", *[f"{sum(summary[s][t]['prop_ok'] and summary[s][t]['unif_ok'] for t in tests)}/{len(tests)}" for s in STREAMS]]]
tb = ax2.table(cellText=rows[1:], colLabels=rows[0], cellLoc="center", loc="center")
tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.8)
for j in range(4):
    tb[0, j].set_facecolor("#2C3E50"); tb[0, j].set_text_props(color="white", fontweight="bold")
ax2.set_title("(b)", fontweight="bold")
save(fig, "Fig10_comprehensive_comparison")

# ---------------------------------------------------------------- Fig 11 p-value uniformity
raw = json.load(open(f"{RES}/nist_raw.json"))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, s in zip(axes, STREAMS):
    ps = np.array([p for r in raw[s] for t, v in r.items() if v is not None and t != "Non-overlapping Template" for p in v])
    ax.hist(ps, bins=10, range=(0, 1), color=COL[s], alpha=0.8, edgecolor="white", density=True)
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set_title(f"{LABEL[s]}\n({len(ps)} p-values, all tests except non-overlapping)", fontsize=10)
    ax.set_xlabel("p-value")
axes[0].set_ylabel("Density")
fig.tight_layout()
save(fig, "Fig11_pvalue_scatter_comparison")

# ---------------------------------------------------------------- Fig 12 autocorrelation profiles
fig, ax = plt.subplots(figsize=(10, 5.5))
for s in STREAMS:
    ax.plot(LAGS, metrics[s]["ac_rms"], "o-", color=COL[s], label=LABEL[s])
ax.axhline(1 / np.sqrt(n_bits), color=RED, ls="--", label="expected rms for ideal bits ($1/\\sqrt{n}$)")
ax.set_yscale("log"); ax.set_xlabel("Lag"); ax.set_ylabel("rms autocorrelation over sequences"); ax.legend()
save(fig, "Fig12_autocorrelation_comparison")
print("done")
