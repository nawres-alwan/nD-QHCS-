"""
Self-contained NIST SP 800-22 rev.1a implementation (all 15 tests), written to follow
the reference STS parameters for n ~ 1e6 bits.  Every test returns a *list* of p-values
(one for most tests; several for Serial, CuSum, Non-overlapping, Excursions) or None
when the test is not applicable to that sequence (Random Excursions with J < 500).

The implementation is validated in nist_validate.py against iid numpy bits:
the pass proportion and the p-value uniformity on that control must be normal.
"""
import math
import numpy as np
from scipy.fft import rfft
from scipy.special import erfc, gammaincc
from scipy.stats import norm


def monobit(b):
    n = len(b)
    s = abs(int(2 * int(b.sum()) - n)) / math.sqrt(n)
    return [float(erfc(s / math.sqrt(2)))]


def block_frequency(b, M=10000):
    N = len(b) // M
    pi = b[: N * M].reshape(N, M).mean(axis=1)
    chi2 = 4.0 * M * float(((pi - 0.5) ** 2).sum())
    return [float(gammaincc(N / 2, chi2 / 2))]


def runs(b):
    n = len(b)
    pi = b.mean()
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return [0.0]
    V = 1 + int((b[1:] != b[:-1]).sum())
    return [float(erfc(abs(V - 2 * n * pi * (1 - pi)) / (2 * math.sqrt(2 * n) * pi * (1 - pi))))]


def longest_run(b, M=10000, N=75):
    pi = np.array([0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727])
    nu = np.zeros(7)
    for i in range(N):
        blk = b[i * M:(i + 1) * M].astype(np.int8)
        d = np.diff(np.concatenate([[0], blk, [0]]))
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        L = int((en - st).max()) if len(st) else 0
        nu[min(max(L, 10), 16) - 10] += 1
    chi2 = float(((nu - N * pi) ** 2 / (N * pi)).sum())
    return [float(gammaincc(3.0, chi2 / 2))]


def _rank32(rows):
    rows = list(rows)
    r = 0
    for bit in range(31, -1, -1):
        mask = 1 << bit
        piv = next((i for i in range(r, len(rows)) if rows[i] & mask), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i] & mask:
                rows[i] ^= rows[r]
        r += 1
    return r


def binary_matrix_rank(b):
    """Rank over GF(2) of disjoint 32x32 matrices (NOT the real-valued rank)."""
    N = len(b) // 1024
    pk = np.packbits(b[: N * 1024].reshape(N, 32, 32), axis=2).astype(np.uint64)  # (N,32,4)
    ints = (pk[..., 0] << 24) | (pk[..., 1] << 16) | (pk[..., 2] << 8) | pk[..., 3]
    ranks = np.array([_rank32([int(v) for v in m]) for m in ints])
    F = [(ranks == 32).sum(), (ranks == 31).sum(), (ranks <= 30).sum()]
    P = [0.2888, 0.5776, 0.1336]
    chi2 = sum((f - N * p) ** 2 / (N * p) for f, p in zip(F, P))
    return [float(math.exp(-chi2 / 2))]


def dft(b):
    n = len(b)
    S = np.abs(rfft(2.0 * b - 1.0))[: n // 2]
    T = math.sqrt(math.log(1 / 0.05) * n)
    N0, N1 = 0.95 * n / 2, float((S < T).sum())
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4)
    return [float(erfc(abs(d) / math.sqrt(2)))]


def _aperiodic_templates(m=9):
    out = []
    for t in range(2 ** m):
        s = format(t, "0%db" % m)
        if all(s[k:] != s[: m - k] for k in range(1, m)):
            out.append(t)
    return out


_TEMPL9 = _aperiodic_templates(9)          # 148 templates


def non_overlapping_template(b, m=9, N=8):
    """All 148 aperiodic 9-bit templates; aperiodic => matches never overlap."""
    n = len(b)
    M = n // N
    ext = b[: N * M].astype(np.int64).reshape(N, M)
    val = np.zeros((N, M - m + 1), dtype=np.int64)
    for j in range(m):
        val = (val << 1) | ext[:, j: j + M - m + 1]
    mu = (M - m + 1) / 2 ** m
    s2 = M * (1 / 2 ** m - (2 * m - 1) / 2 ** (2 * m))
    ps = []
    for t in _TEMPL9:
        W = (val == t).sum(axis=1)
        ps.append(float(gammaincc(N / 2, float(((W - mu) ** 2 / s2).sum()) / 2)))
    return ps


_OT_PI = np.array([0.364091, 0.185659, 0.139381, 0.100571, 0.0704323, 0.139865])


def overlapping_template(b, m=9, M=1032, N=968):
    x = b[: M * N].astype(np.int32).reshape(N, M)
    cs = np.concatenate([np.zeros((N, 1), np.int32), np.cumsum(x, axis=1)], axis=1)
    cnt = ((cs[:, m:] - cs[:, :-m]) == m).sum(axis=1)
    nu = np.bincount(np.minimum(cnt, 5), minlength=6)
    chi2 = float(((nu - N * _OT_PI) ** 2 / (N * _OT_PI)).sum())
    return [float(gammaincc(2.5, chi2 / 2))]


def universal(b, L=7, Q=1280):
    n = len(b)
    K = n // L - Q
    blk = b[: (Q + K) * L].reshape(Q + K, L).astype(np.int64)
    idx = blk @ (1 << np.arange(L - 1, -1, -1))
    last = np.zeros(2 ** L, dtype=np.int64)
    for i in range(Q):
        last[idx[i]] = i + 1
    s = 0.0
    for i in range(Q, Q + K):
        s += math.log2(i + 1 - last[idx[i]])
        last[idx[i]] = i + 1
    fn = s / K
    c = 0.7 - 0.8 / L + (4 + 32 / L) * K ** (-3 / L) / 15
    sigma = c * math.sqrt(3.125 / K)
    return [float(erfc(abs(fn - 6.1962507) / (math.sqrt(2) * sigma)))]


def _bm(s):
    """Berlekamp-Massey linear complexity over GF(2), bit-parallel with Python ints."""
    C, B, L, mm, R = 1, 1, 0, -1, 0
    for k, bit in enumerate(s):
        R = (R << 1) | int(bit)
        if (C & R).bit_count() & 1:
            T = C
            C ^= B << (k - mm)
            if 2 * L <= k:
                L, mm, B = k + 1 - L, k, T
    return L


def linear_complexity(b, M=500):
    N = len(b) // M
    mu = M / 2 + (9 + (-1) ** (M + 1)) / 36 - (M / 3 + 2 / 9) / 2 ** M
    T = np.array([(-1) ** M * (_bm(b[i * M:(i + 1) * M]) - mu) + 2 / 9 for i in range(N)])
    nu = np.bincount(np.digitize(T, [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], right=True), minlength=7)
    pi = np.array([0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833])
    return [float(gammaincc(3.0, float(((nu - N * pi) ** 2 / (N * pi)).sum()) / 2))]


def _counts(b, m):
    n = len(b)
    if m == 0:
        return np.array([n])
    ext = np.concatenate([b, b[: m - 1]]).astype(np.int64)
    v = np.zeros(n, dtype=np.int64)
    for j in range(m):
        v = (v << 1) | ext[j: j + n]
    return np.bincount(v, minlength=2 ** m)


def serial(b, m=16):
    n = len(b)

    def psi(k):
        return 0.0 if k <= 0 else float((_counts(b, k).astype(float) ** 2).sum()) * 2 ** k / n - n

    a, c, d = psi(m), psi(m - 1), psi(m - 2)
    return [float(gammaincc(2 ** (m - 2), (a - c) / 2)),
            float(gammaincc(2 ** (m - 3), (a - 2 * c + d) / 2))]


def approximate_entropy(b, m=10):
    n = len(b)

    def phi(k):
        c = _counts(b, k).astype(float)
        c = c[c > 0] / n
        return float((c * np.log(c)).sum())

    return [float(gammaincc(2 ** (m - 1), n * (math.log(2) - (phi(m) - phi(m + 1)))))]


def cumulative_sums(b):
    n = len(b)
    x = 2 * b.astype(np.int64) - 1

    def p(seq):
        z = int(np.abs(np.cumsum(seq)).max())
        sn = math.sqrt(n)
        k1 = np.arange(math.floor((-n / z + 1) / 4), math.floor((n / z - 1) / 4) + 1)
        k2 = np.arange(math.floor((-n / z - 3) / 4), math.floor((n / z - 1) / 4) + 1)
        s1 = (norm.cdf((4 * k1 + 1) * z / sn) - norm.cdf((4 * k1 - 1) * z / sn)).sum()
        s2 = (norm.cdf((4 * k2 + 3) * z / sn) - norm.cdf((4 * k2 + 1) * z / sn)).sum()
        return float(1 - s1 + s2)

    return [p(x), p(x[::-1])]


def _walk(b):
    S = np.concatenate([[0], np.cumsum(2 * b.astype(np.int64) - 1), [0]])
    zero = S == 0
    return S, int(zero.sum()) - 1, np.cumsum(zero)           # S', J, cycle index


def random_excursions(b):
    S, J, cyc = _walk(b)
    if J < 500:
        return None
    ps = []
    for x in (-4, -3, -2, -1, 1, 2, 3, 4):
        ax = abs(x)
        per_cycle = np.bincount(cyc[S == x] - 1, minlength=J)[:J]
        nu = np.bincount(np.minimum(per_cycle, 5), minlength=6)
        pi = np.empty(6)
        pi[0] = 1 - 1 / (2 * ax)
        for k in range(1, 5):
            pi[k] = (1 / (4 * ax * ax)) * (1 - 1 / (2 * ax)) ** (k - 1)
        pi[5] = (1 / (2 * ax)) * (1 - 1 / (2 * ax)) ** 4
        ps.append(float(gammaincc(2.5, float(((nu - J * pi) ** 2 / (J * pi)).sum()) / 2)))
    return ps


def random_excursions_variant(b):
    S, J, _ = _walk(b)
    if J < 500:
        return None
    return [float(erfc(abs(int((S == x).sum()) - J) / math.sqrt(2 * J * (4 * abs(x) - 2))))
            for x in list(range(-9, 0)) + list(range(1, 10))]


TESTS = {
    "Frequency (Monobit)": monobit,
    "Block Frequency": block_frequency,
    "Runs": runs,
    "Longest Run of Ones": longest_run,
    "Binary Matrix Rank": binary_matrix_rank,
    "DFT (Spectral)": dft,
    "Non-overlapping Template": non_overlapping_template,
    "Overlapping Template": overlapping_template,
    "Universal": universal,
    "Linear Complexity": linear_complexity,
    "Serial": serial,
    "Approximate Entropy": approximate_entropy,
    "Cumulative Sums": cumulative_sums,
    "Random Excursions": random_excursions,
    "Random Excursions Variant": random_excursions_variant,
}


def run_all(bits):
    b = np.asarray(bits, dtype=np.uint8)
    return {name: f(b) for name, f in TESTS.items()}
