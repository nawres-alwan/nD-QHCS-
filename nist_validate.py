"""
NIST SP 800-22 validation of the HGPhi-QRNG output, done the way the standard asks:
many sequences, per-test pass *proportion* and p-value *uniformity*.

Three streams are tested side by side
  chaos    : full pipeline (nD-HCS -> geometric phase -> interferometer -> von Neumann)
  control  : same pipeline with the chaotic phase replaced by a constant (chaos OFF)
  numpy    : numpy PCG64 bits (calibration of the test implementation itself)

usage:  python nist_validate.py gen  [--n-seq 100] [--n-bits 1100000]
        python nist_validate.py test
        python nist_validate.py report
"""
import argparse, json, math, os, sys, time, warnings
from multiprocessing import Pool
import numpy as np
from scipy.special import gammaincc

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
STREAMS = ["chaos", "control", "numpy"]
ALPHA = 0.01


def cmd_gen(a):
    import hgphi_core as h
    n_windows = int(a.n_bits * 4.5) + 1000
    t = time.time()
    print(f"integrating {a.n_seq} trajectories x {n_windows} windows ...", flush=True)
    ph = h.window_means(h.initial_states(a.n_seq, 6, a.seed), n_windows)
    print(f"  done in {time.time()-t:.0f}s; phase std {ph.std():.3f}", flush=True)
    chaos, _ = h.generate_sequences(a.n_seq, a.n_bits, a.seed, wm=ph)
    ctrl, _ = h.generate_sequences(a.n_seq, a.n_bits, a.seed, constant_phase=0.0, wm=ph)
    ref = np.random.default_rng(a.seed + 7).integers(0, 2, chaos.shape, dtype=np.uint8)
    for name, arr in zip(STREAMS, [chaos, ctrl, ref]):
        np.save(os.path.join(OUT, f"seqs_{name}.npy"), np.packbits(arr, axis=1))
    json.dump({"n_seq": a.n_seq, "n_bits": a.n_bits, "seed": a.seed,
               "mean_bit": {n: float(x.mean()) for n, x in zip(STREAMS, [chaos, ctrl, ref])}},
              open(os.path.join(OUT, "gen_meta.json"), "w"), indent=1)
    print("saved sequences", flush=True)


def _worker(args):
    import nist_sts
    stream, i, packed, n_bits = args
    return stream, i, nist_sts.run_all(np.unpackbits(packed)[:n_bits])


def cmd_test(a):
    meta = json.load(open(os.path.join(OUT, "gen_meta.json")))
    jobs = []
    for s in STREAMS:
        P = np.load(os.path.join(OUT, f"seqs_{s}.npy"))
        jobs += [(s, i, P[i], meta["n_bits"]) for i in range(P.shape[0])]
    results = {s: [None] * meta["n_seq"] for s in STREAMS}
    t = time.time()
    with Pool(a.procs) as pool:
        for k, (s, i, r) in enumerate(pool.imap_unordered(_worker, jobs), 1):
            results[s][i] = r
            if k % 25 == 0:
                print(f"{k}/{len(jobs)} sequences  ({time.time()-t:.0f}s)", flush=True)
    json.dump(results, open(os.path.join(OUT, "nist_raw.json"), "w"))


def summarize(results):
    """Per stream/test: pooled p-values -> pass proportion + uniformity P_T."""
    table = {}
    for s, seqs in results.items():
        pool = {}
        for r in seqs:
            for name, ps in r.items():
                if ps is not None:
                    pool.setdefault(name, []).extend(ps)
        table[s] = {}
        for name, ps in pool.items():
            ps = np.array(ps)
            m = len(ps)
            npass = int((ps >= ALPHA).sum())
            ph = 1 - ALPHA
            lo = math.ceil((ph - 3 * math.sqrt(ph * ALPHA / m)) * m)
            hist = np.histogram(ps, bins=10, range=(0, 1))[0]
            chi = float(((hist - m / 10) ** 2 / (m / 10)).sum())
            pt = float(gammaincc(4.5, chi / 2))
            table[s][name] = dict(n_pvalues=m, passed=npass, min_required=lo,
                                  prop_ok=bool(npass >= lo), P_T=pt, unif_ok=bool(pt >= 1e-4),
                                  n_eligible_seqs=sum(1 for r in seqs if r.get(name) is not None))
    return table


def cmd_report(a):
    res = json.load(open(os.path.join(OUT, "nist_raw.json")))
    tab = summarize(res)
    json.dump(tab, open(os.path.join(OUT, "nist_summary.json"), "w"), indent=1)
    names = list(tab["chaos"].keys())
    print(f"{'test':28s}" + "".join(f"{s:>30s}" for s in STREAMS))
    for n in names:
        row = f"{n:28s}"
        for s in STREAMS:
            d = tab[s].get(n)
            ok = d["prop_ok"] and d["unif_ok"]
            row += f"   {d['passed']:5d}/{d['n_pvalues']:<5d} P_T={d['P_T']:.3f} {'OK ' if ok else 'BAD'}"
        print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("--n-seq", type=int, default=100)
    g.add_argument("--n-bits", type=int, default=1_100_000)
    g.add_argument("--seed", type=int, default=1)
    t = sp.add_parser("test")
    t.add_argument("--procs", type=int, default=6)
    sp.add_parser("report")
    a = ap.parse_args()
    {"gen": cmd_gen, "test": cmd_test, "report": cmd_report}[a.cmd](a)
