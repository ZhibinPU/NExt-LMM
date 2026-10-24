# NExt-LMM

Implementation of

> Zhibin Pu, Shufei Ge, Shijia Wang, "NExt-LMM: a two-stage LMM framework
> with near-exact variance components for genome-wide association
> studies", arXiv:2508.05278v2.

```python
from NExt import NExt

model = NExt(y).fit(W, K)        # W: (n, P) genotypes, K: (n, n) GSM
p_values = model.association()
evidence = model.evidence(K)     # what the approximation cost
```

## What changed since the released version

The original files are archived as `HodlrInverse_published.py` and
`NExt_published.py`. Three defects were fixed; all three are invisible
in a trace plot, which is why they lasted.

**The recursion had never run.** `HInverse.__init__` called `HoDLR(...)`
where the class is named `HInverse`, so every call with a block above
the leaf size raised `NameError`. Only the dense leaf path had ever
executed — the paper's algorithm had not been run by the code released
with it.

**The Schur complement was inverted densely.** Algorithm 1 line 10 reads
`S^-1 <- HODLR-inverse(S, eps, s)`. The released code used
`np.linalg.inv(S)`, which collapses the recursion to a single level
whatever the matrix size, and the `O(k^2 n log^2 n)` claim with it.

**The association statistic divided by the wrong thing.**

```python
T = self.beta / np.std(self.beta)     # released
```

The denominator is the spread of effects *across markers*, not the
standard error of any one of them. Under a pure null this is calibrated,
which is presumably why it went unnoticed; but real signal enters its own
denominator. On simulated data with 24 causal markers out of 60, it found
17.0% of them against 29.7% for the Wald statistic implied by Equations
(7) and (8) — and the gap widens as the architecture gets denser. The
paper attributes falling power at high causal fractions to genetic
architecture; some of it is this.

Two smaller corrections: PCGC now solves Equation (5) as written (no
intercept, clamped to `[0, 1]`; the released code used
`LinearRegression()` with its default intercept and no clamp), and the
covariance is `lambda K + I` with `lambda = h2/(1-h2)` rather than
`h2 K + (1-h2) I`. The two differ by a constant factor, which GLS ignores
for `beta` and does not ignore for `sigma_e^2` — at `h2 = 0.5` the
released residual variance was 2x too large.

## The rank dial

HODLR is a **lossy** format. The truncation rank is chosen, not
discovered: a matrix whose off-diagonal blocks happen to be low rank
compresses for free, and one whose blocks are full rank still compresses
and pays in accuracy instead.

Genetic similarity matrices are the second kind. Measured on a simulated
GSM (n = 512, m = 2560) and on real ADNI genotypes, the off-diagonal
blocks retain essentially full numerical rank. So `max_rank` is a dial
the caller sets against a budget, and `evidence()` reports what the
setting cost.

What it costs, at n = 256 over 120 null replicates and 40 power
replicates:

| `max_rank` | ‖Σ − Σ̃‖₂ | storage / n² | type-I at 0.05 | power |
|---|---|---|---|---|
| 2 | 0.185 | 0.137 | 0.0495 | 0.302 |
| 8 | 0.165 | 0.172 | 0.0512 | 0.295 |
| 32 | 0.107 | 0.312 | 0.0496 | — |
| none | 0.065 | 0.562 | 0.0537 | 0.297 |

The spectral error looks alarming at rank 2 and the inference is almost
untouched: type-I error stays at nominal across the whole range, and
power moves by about 2%.

The reason is worth knowing, because it says when this will and will not
hold. The approximate inverse stays **symmetric and positive definite**
(asymmetry ~1e-17, smallest eigenvalue 0.65 where the exact one gives
0.65), so generalised least squares is still using a *valid* weight
matrix — just not the optimal one. GLS with any positive-definite weight
is unbiased; a suboptimal one costs efficiency. And because `sigma_e^2`
is computed through the same approximate inverse, the reported standard
error stays honest rather than drifting from the realised spread.

This also explains why the paper's Proposition 1 bound is loose. It is a
worst case over all quadratic forms, while an association test only ever
probes a two-dimensional subspace of them.

## Performance

The scan is where the time goes. At n = 2000 with 100,000 markers, a
dense inverse of Σ takes about 0.4 s while the per-marker tests take
about 4 minutes — the scan is 99.9% of the work, so accelerating only
the setup accelerates almost nothing.

Nothing in `NExt.fit` materialises the inverse. The covariance is applied
through `HInverse.matmat`, with markers pushed through in blocks so the
scan reaches BLAS-3 rather than doing two dense `n x n` products per
marker:

| n | P | rewritten | per-marker dense | speedup |
|---|---|---|---|---|
| 256 | 2000 | 0.04 s | 0.54 s | 13.6x |
| 512 | 2000 | 0.12 s | 1.27 s | 10.7x |
| 1024 | 2000 | 0.62 s | 4.51 s | 7.3x |

### Making it fast

The format promises `O(k n log n)` per solve at fixed `k`. Getting
there took four fixes, three of them in the Python and one in the
algorithm itself.

**The recursion was visiting subtrees twice.** Every internal node
needs both `N11 b1` and `N11 U`. Requesting them in two separate
descents makes the call count grow as `3^depth` rather than `2^depth`
— 1093 calls over 127 nodes at n = 4096 — so the cost was
`O(n^1.585)`, not `O(k n log n)`. Stacking the two right-hand sides
and descending once fixes it; a test now asserts exactly one visit per
node.

**Every level allocated and copied the full vector.** `np.vstack` at
the end of each node: 10,920 calls, 21% of runtime. The destination
buffer is sliced instead.

**The scan batch was too wide.** 512 columns spilled the per-level
working set out of cache; 128 sits near the bottom of a flat region
between 64 and 256.

**Construction, not solving, was the bottleneck.** Profiling a scan at
n = 2048: building the factors took 89% of the time and every solve
together took 10%. Each internal node was taking a full SVD of its
off-diagonal block. Passing `randomized=True` projects onto a small
random subspace first, which is about 10x cheaper to build.

That last one deserves its caveat spelled out, because the textbook
objection is real and does not apply here. Randomised SVD wants a
decaying spectrum, and a GSM's off-diagonal blocks have none — the
singular values fall from 1.00 to 0.91 across the first eight, which
hold 8% of the spectral energy. But *because* the spectrum is flat, no
8-dimensional subspace is much better than another: on a 512 x 512
block the optimal rank-8 truncation gave `||M - UV||_2 = 2.67` and the
randomised one 2.88, 8% worse for 34x less work. On a matrix whose
spectrum does decay, the optimal truncation would be much better and
this would be the wrong switch — which is why it is opt-in.

### What it adds up to

One solve, against a dense `Si @ B` (128 columns, per column):

| n | HODLR | dense | |
|---|---|---|---|
| 512 | 0.0052 ms | 0.0113 ms | **2.2x faster** |
| 1024 | 0.0142 ms | 0.0467 ms | **3.3x faster** |
| 2048 | 0.0384 ms | 0.1941 ms | **5.1x faster** |
| 4096 | 0.0916 ms | 0.7737 ms | **8.4x faster** |

The ratio grows with n, which is what an `O(n log n)` method against an
`O(n^2)` one should do. Before these fixes the same comparison ran 1.4x
to 10.9x *slower* at every size tested.

End to end, scanning 2000 markers:

| n | full SVD | randomised | per-marker dense | |
|---|---|---|---|---|
| 1024 | 0.18 s | 0.06 s | 4.12 s | **65x** |
| 2048 | 0.94 s | 0.18 s | 16.46 s | **92x** |

The randomised build changes the reported effect sizes by a median of
0.06 standard errors — statistically nothing, and pinned by a test.

### The compiled kernel

`_hodlr_fast.pyx` flattens the tree into arrays and calls `dgemm`
through `scipy.linalg.cython_blas`, removing a Python frame and a
numpy dispatch per node. Build it with:

```bash
pip install cython
python setup.py build_ext --inplace
```

It is optional — without it `HInverse` uses the Python path and gets
the same answers, which a test checks by switching the extension off.
On the solve alone it is worth 1.3-1.6x. That is less than it sounds,
because after the randomised build the solve is no longer where the
time goes; it matters when the same factorisation is reused across
many more markers than the 2000 benchmarked here.

Two bugs while writing it, both worth recording because neither would
show up in a test of the extension alone. `V` is stored `(k, lower)`
and the product wanted is `V^T tmp`, which came out transposed. And
the first version dodged the BLAS layout question by writing three
products as explicit triple loops — 6.3 million scalar operations per
node against the 12.9 GFLOPS BLAS reaches on the same work, which made
the compiled path three times *slower* than the interpreter it was
meant to replace. Getting the `dgemm` flags right was the whole job.

## API

### `HInverse(matrix, min_block=64, tolerance=1e-6, max_rank=None)`

HODLR representation of a symmetric positive definite matrix's inverse.

- `matvec(v)`, `matmat(B)` — apply the inverse without forming it.
- `to_dense()` — materialise it. O(n²) storage, and it discards the
  format's advantage; for tests and for callers that need the entries.
- `error_bound(matrix)` — relative residual by random probing, retained
  rank, storage against dense.
- `max_rank_retained()`, `storage()`.

### `NExt(Y, min_block=64, tolerance=1e-6, max_rank=None)`

- `pcgc(K)` — Equation (5), clamped to `[0, 1]`.
- `factorise(K, h2)` — build `Sigma = lambda K + I` and factor it.
- `fit(W, K, h=None)` — scan. `h=None` estimates it by PCGC.
- `association()` — two-sided Wald p-values. A monomorphic marker gives
  NaN rather than a spurious effect.
- `evidence(K)` — the approximation's cost alongside the fit.

## Tests

```bash
pytest -q                  # 52 tests, ~5 s
pytest -m "not slow" -q    # skipping the Monte Carlo checks
```

They assert behaviour against known truths — an exact dense solve, a null
that is true by construction, identities that hold at any rank — rather
than against stored numbers, which would only pin the implementation to
itself.
