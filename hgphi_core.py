"""
HGPhi-QRNG core pipeline (simulation).

nD-HCS (Euler)  ->  window means  ->  Lemma-1 geometric phase
 -> interferometric Bernoulli sampling (random reference phase) -> von Neumann.

NOTE: this is a *simulation*. The reference phase delta_r and the Bernoulli draw
are produced by a software PRNG (numpy Generator); see README, section
"What the simulation does and does not show".
"""
import numpy as np

A, B, DT = 1.0, 0.3, 0.01
B0, BETA, TAU = 3.0, 4.0, 0.3
CPW = 5            # chaotic steps per window
TRANSIENT = 3000
BOUND = 10.0
X0 = np.array([0.5, -0.3, 0.7, -0.1, 0.4, -0.6])


def initial_states(n_seq, n_dim=6, seed=0, spread=0.05):
    """Distinct initial conditions: X0 plus a small Gaussian offset per sequence."""
    rng = np.random.default_rng(seed)
    x0 = X0[:n_dim][None, :] + spread * rng.standard_normal((n_seq, n_dim))
    x0[0] = X0[:n_dim]
    return x0


def window_means(x0, n_windows, n_dim=6, chunk=20000):
    """
    Integrate the nD-HCS dX_i/dt = (A X_{i+1} - B X_{i-2}) X_{i-1} for a batch of
    initial states (shape (S, n_dim)), return Lemma-1 phases per window, shape (S, n_windows).
    Same Euler step / soft clamp as the original script, vectorised over the batch.
    """
    X = np.array(x0, dtype=np.float64)
    S = X.shape[0]
    ip1 = np.array([(i + 1) % n_dim for i in range(n_dim)])
    im1 = np.array([(i - 1) % n_dim for i in range(n_dim)])
    im2 = np.array([(i - 2) % n_dim for i in range(n_dim)])

    def step(X):
        X = X + DT * (A * X[:, ip1] - B * X[:, im2]) * X[:, im1]
        mx = np.max(np.abs(X), axis=1)
        big = mx > BOUND
        if big.any():
            X[big] *= ((BOUND - 0.01) / mx[big])[:, None]
        return X

    for _ in range(TRANSIENT):
        X = step(X)
    out = np.empty((S, n_windows), dtype=np.float32)      # phases, not raw means
    buf = np.empty((S, chunk, n_dim))
    w = 0
    while w < n_windows:
        m = min(chunk, n_windows - w)
        for j in range(m):
            acc = np.zeros_like(X)
            for _ in range(CPW):
                X = step(X)
                acc += X
            buf[:, j, :] = acc / CPW
        out[:, w:w + m] = geometric_phases(buf[:, :m, :])[0]
        w += m
    return out


def geometric_phases(wm):
    """Lemma-1 phase  dgamma = C_theta * Omega^2 * tau^2  (mod 2 pi), wm: (..., n_dim)."""
    s_k = np.tanh(wm[..., 0].astype(np.float64))
    ax = wm[..., 1:4].astype(np.float64)
    nrm = np.maximum(np.linalg.norm(ax, axis=-1, keepdims=True), 1e-8)
    theta = np.arccos(np.clip((ax / nrm)[..., 2], -0.999, 0.999))
    omega = B0 + BETA * s_k
    c_theta = 0.5 * np.sin(theta) ** 2 * np.cos(theta)
    return (c_theta * omega ** 2 * TAU ** 2) % (2 * np.pi), s_k, theta


def sample_bits(phases, rng, constant_phase=None):
    """Interferometric sampling with one random reference phase per window."""
    if constant_phase is not None:          # control: chaos switched off
        phases = np.full_like(phases, constant_phase)
    delta = rng.uniform(0, 2 * np.pi, phases.shape)
    p = 0.5 * (1 + np.cos(phases + delta))
    return (rng.random(phases.shape) < p).astype(np.uint8)


def von_neumann(bits):
    n = len(bits) // 2 * 2
    pr = bits[:n].reshape(-1, 2)
    return pr[pr[:, 0] != pr[:, 1], 0]


def generate_sequences(n_seq, n_bits, seed=1, n_dim=6, constant_phase=None, wm=None):  # wm = precomputed phases
    """Return (n_seq, n_bits) uint8 array. Each row from its own trajectory."""
    n_windows = int(n_bits * 4.5) + 1000        # VN keeps ~1/4 of windows
    if wm is None:
        wm = window_means(initial_states(n_seq, n_dim, seed), n_windows, n_dim)
    phases = wm
    rng = np.random.default_rng(seed + 1000)
    seqs = np.empty((n_seq, n_bits), dtype=np.uint8)
    for i in range(n_seq):
        vn = von_neumann(sample_bits(phases[i], rng, constant_phase))
        if len(vn) < n_bits:
            raise RuntimeError("not enough VN bits; increase window margin")
        seqs[i] = vn[:n_bits]
    return seqs, phases
