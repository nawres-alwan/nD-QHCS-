# nD-QHCS / HGΦ-QRNG — simulation code and figures

Code and figures for the hyperchaotic geometric-phase QRNG (HGΦ-QRNG): an nD hyperchaotic system
drives a Samuel–Bhandari open-path geometric phase (Lemma 1), which sets the output probability of a
simulated interferometer; bits are debiased with a von Neumann extractor.

## Reproduce (everything below is generated, nothing is hard-coded)

```bash
pip install -r requirements.txt
python nist_validate.py gen      # 100 independent trajectories -> 3 x 100 sequences of 1.1 Mbit (~15 min)
python nist_validate.py test     # NIST SP 800-22, all 15 tests, multiprocessing (~2-3 min)
python nist_validate.py report   # summary table -> results/nist_summary.json
python make_figures.py           # figures/Fig1..Fig12 (.png + .pdf) and results/metrics.json
```

| file | content |
|---|---|
| `hgphi_core.py` | nD-HCS (Euler), Lemma-1 phase, interferometric sampling, von Neumann |
| `nist_sts.py` | self-contained NIST SP 800-22 rev.1a implementation (15 tests, STS parameters for n ≈ 10⁶) |
| `nist_validate.py` | sequence generation, test driver, proportion + uniformity analysis |
| `make_figures.py` | all figures from the data above |
| `results/` | `nist_summary.json`, `nist_raw.json` (every p-value), `metrics.json`, `gen_meta.json` |

## NIST SP 800-22 method

* 100 sequences × 1,100,000 bits, each from its own trajectory (initial state = X0 + 0.05·N(0,1)).
* Per test: proportion of p-values ≥ 0.01 against the minimum acceptable proportion
  (p̂ − 3√(p̂(1−p̂)/m), p̂ = 0.99), and uniformity of the p-values (10-bin χ², P_T ≥ 10⁻⁴).
  Tests that give many p-values per sequence (non-overlapping template: 148, excursions: 8 / 18)
  are pooled over sequences. Random Excursions are evaluated only on sequences with ≥ 500 cycles.
* **The test code is calibrated**: on 300 sequences of numpy PCG64 bits all 15 tests give a normal
  proportion and uniform p-values (`nist_validate.py` is run on that reference stream every time).
* Three streams are tested side by side: `chaos` (this generator), `control` (same pipeline with the
  chaotic phase replaced by a constant) and `numpy` (reference PRNG).

## Results (as produced by the commands above)

| | chaos on | chaos off (control) | numpy reference |
|---|---|---|---|
| NIST tests meeting proportion + uniformity | 15 / 15 | 14 / 15 (DFT: 96/100, limit 97) | 15 / 15 |
| balance | 0.499935 | 0.499916 | 0.499945 |
| min-entropy (bit/bit) | 0.999811 | 0.999757 | 0.999840 |
| mean AC(1) | +7.6e-5 | +6.0e-5 | −2.4e-6 |

## What the simulation does and does not show — please read

* The output bits are statistically indistinguishable from the numpy reference **and from the
  chaos-off control**. The control passes the same battery, so the NIST result does **not** show that the
  hyperchaotic driving or the geometric phase contributes randomness.
* In the simulation the reference phase δ_r and the Bernoulli outcome are drawn from a software PRNG
  (`numpy.random.Generator`), i.e. the final randomness comes from that PRNG. The simulation validates the
  *pipeline* (phase statistics, bias removal, readout model); it is not a certification of quantum
  randomness. A physical implementation would replace these two draws by interferometer measurements.
* Figs 1–3 are model illustrations (Gaussian detector noise with fixed σ), not measurements.
* Lemma 1 (Δγ = C_θ Ω² τ²) is a small-angle formula; with B0 = 3, β = 4, τ = 0.3 the product Ωτ reaches ≈ 2.1.
* The "conservative" system is integrated with an Euler step and a soft rescaling whenever |X| > 10, which
  is not volume-preserving; the Lyapunov spectrum is not part of this repository.
* No comparison with other published QRNGs is made here: their per-test values must be taken from the
  cited papers, which this repository does not reproduce.
