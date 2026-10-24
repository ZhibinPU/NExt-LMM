"""
Regression tests for NExt-LMM.

The published release had three defects that these pin shut:

* ``HInverse``'s recursive branch called an undefined name, so only the
  leaf path ever executed — the paper's algorithm had never run;
* Algorithm 1 line 10 recurses on the Schur complement, and the code
  inverted it densely, collapsing the recursion to one level;
* the association statistic divided each marker's effect by the spread
  of effects across markers rather than by a standard error.

What is asserted here is mostly behaviour against a known truth: an
exact dense solve, a null that is true by construction, an identity
that has to hold whatever the rank. Stored numbers would only pin the
implementation to itself.

Run ``pytest -m "not slow"`` to skip the Monte Carlo checks.
"""

import numpy as np
import pytest
from scipy import stats

from HodlrInverse import HInverse, hodlr_approximation, low_rank
from NExt import NExt


# ======================================================================
# Helpers
# ======================================================================
def gsm(n, m_factor=5, seed=0):
    """A genetic similarity matrix, built the way a real one is."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, m_factor * n))
    return Z @ Z.T / (m_factor * n)


def covariance(K, h2):
    """``Sigma = lambda K + I`` with ``lambda = h2 / (1 - h2)``."""
    return h2 / (1 - h2) * K + np.eye(K.shape[0])


def dense_gls(Sigma_inv, x, y):
    """Exact two-parameter GLS, as the reference for the fast path."""
    n = y.size
    design = np.column_stack([x, np.ones(n)])
    gram = np.linalg.inv(design.T @ Sigma_inv @ design)
    theta = gram @ design.T @ Sigma_inv @ y
    residual = y - design @ theta
    s2 = float(residual @ Sigma_inv @ residual / (n - 2))
    return theta[0], np.sqrt(s2 * gram[0, 0]), s2


# ======================================================================
# 1. The recursion exists and is correct
# ======================================================================
class TestHodlrRecursion:
    """
    The released code's recursive branch raised ``NameError``: it called
    ``HoDLR`` where the class is ``HInverse``. Every published result
    therefore came from the leaf path alone, which is a dense inverse
    with extra steps.
    """

    @pytest.mark.parametrize("n", [128, 256, 512])
    def test_the_recursion_runs_and_has_the_expected_depth(self, n):
        model = HInverse(covariance(gsm(n), 0.5), min_block=64)
        depth, node = 0, model
        while not node.is_leaf():
            depth += 1
            node = node.A11
        assert depth == int(np.log2(n / 64)), f"depth {depth} at n={n}"

    def test_the_schur_complement_recurses(self):
        """
        Algorithm 1 line 10 is ``S^-1 <- HODLR-inverse(S)``. Inverting S
        densely leaves the recursion one level deep whatever the size,
        and the complexity claim with it.
        """
        model = HInverse(covariance(gsm(256), 0.5), min_block=64)
        assert not model.A22.is_leaf(), (
            "the Schur complement was inverted densely, so the recursion "
            "is one level deep")

    def test_a_dense_block_reproduces_the_exact_inverse(self):
        """With no compression the result must be exact, not merely close."""
        Sigma = covariance(gsm(100), 0.5)
        model = HInverse(Sigma, min_block=200)       # one leaf, no splitting
        assert model.is_leaf()
        assert np.allclose(model.to_dense(), np.linalg.inv(Sigma),
                           rtol=1e-10, atol=1e-12)

    def test_a_singular_block_is_refused_with_its_reason(self):
        with pytest.raises(ValueError, match="singular"):
            HInverse(np.zeros((8, 8)), min_block=16)

    def test_a_non_square_matrix_is_refused(self):
        with pytest.raises(ValueError, match="square"):
            HInverse(np.ones((4, 6)))


# ======================================================================
# 2. The factored form is kept
# ======================================================================
class TestFactoredApplication:
    """
    The format's whole value is applying the inverse in O(k n log n)
    without forming it. ``to_dense`` is O(n^2) in storage and throws
    that away, so the fast path has to be the one the scan uses — and
    it has to agree with the slow one.
    """

    @pytest.fixture
    def model_and_dense(self):
        Sigma = covariance(gsm(256), 0.5)
        model = HInverse(Sigma, min_block=64)
        return model, model.to_dense()

    def test_matvec_matches_the_materialised_inverse(self, model_and_dense):
        model, dense = model_and_dense
        v = np.random.default_rng(1).standard_normal(model.size)
        assert np.allclose(model.matvec(v), dense @ v, rtol=1e-10,
                           atol=1e-12)

    def test_matmat_matches_column_by_column(self, model_and_dense):
        model, dense = model_and_dense
        B = np.random.default_rng(2).standard_normal((model.size, 7))
        assert np.allclose(model.matmat(B), dense @ B, rtol=1e-10,
                           atol=1e-12)

    def test_storage_is_below_dense(self):
        """
        The claim the format makes. An aggressive rank has to actually
        store less, or the recursion is pure overhead.
        """
        Sigma = covariance(gsm(512), 0.5)
        model = HInverse(Sigma, min_block=64, max_rank=2)
        assert model.storage() < 0.3 * model.size ** 2

    def test_each_node_is_visited_exactly_once(self):
        """
        Both ``N11 b1`` and ``N11 U`` are needed at every internal node.
        Asking for them in two separate descents makes the call count
        grow as ``3^depth`` instead of ``2^depth`` — measured at 1093
        calls against 127 nodes for n = 4096 — which turns the cost
        into O(n^1.585) and quietly gives up the complexity the format
        exists for. Stacking the two right-hand sides and descending
        once restores it.
        """
        Sigma = covariance(gsm(1024), 0.5)
        model = HInverse(Sigma, min_block=64, max_rank=8)

        def count_nodes(node):
            if node.is_leaf():
                return 1
            return 1 + count_nodes(node.A11) + count_nodes(node.A22)

        calls = [0]
        original = HInverse._apply

        def counting(self, B, out, _original=original):
            calls[0] += 1
            return _original(self, B, out)

        HInverse._apply = counting
        try:
            # The Python path explicitly: matvec prefers the compiled
            # kernel when it is built, and this test is about the
            # traversal, which both paths share.
            buffer = np.empty((1024, 1))
            model._apply(np.ones((1024, 1)), buffer)
        finally:
            HInverse._apply = original

        assert calls[0] == count_nodes(model), (
            f"{calls[0]} descents over {count_nodes(model)} nodes — the "
            f"recursion is revisiting subtrees")

    def test_a_wrong_shaped_operand_is_refused(self, model_and_dense):
        model, _ = model_and_dense
        with pytest.raises(ValueError, match="rows"):
            model.matmat(np.ones((model.size + 1, 2)))


# ======================================================================
# 2b. The compiled kernel
# ======================================================================
class TestCompiledKernel:
    """
    The Cython path is an optimisation, so the only thing that makes it
    safe is agreeing with the Python one element for element. Both bugs
    found while writing it were invisible to any test of the extension
    alone: a transposed operand (V is stored ``(k, lower)``, and the
    product wanted is ``V^T tmp``) and three hand-written triple loops
    put in to dodge the layout question, which made the compiled path
    three times slower than the interpreter.
    """

    @pytest.mark.parametrize("n", [128, 512, 1024])
    @pytest.mark.parametrize("width", [1, 13, 128])
    def test_it_agrees_with_the_python_path(self, n, width):
        from HodlrInverse import HAVE_FAST

        if not HAVE_FAST:
            pytest.skip("compiled extension not built")

        Sigma = covariance(gsm(n, seed=n), 0.5)
        model = HInverse(Sigma, min_block=64, max_rank=8)
        B = np.ascontiguousarray(
            np.random.default_rng(n + width).standard_normal((n, width)))

        compiled = model.matmat(B)
        interpreted = np.empty_like(B)
        model._apply(B, interpreted)
        assert np.allclose(compiled, interpreted, rtol=1e-12, atol=1e-13)

    def test_a_tree_of_one_leaf_still_works(self):
        """The degenerate flattening: no internal nodes at all."""
        from HodlrInverse import HAVE_FAST

        if not HAVE_FAST:
            pytest.skip("compiled extension not built")

        Sigma = covariance(gsm(64, seed=1), 0.5)
        model = HInverse(Sigma, min_block=128)
        assert model.is_leaf()
        v = np.random.default_rng(1).standard_normal(64)
        assert np.allclose(model.matvec(v), np.linalg.inv(Sigma) @ v,
                           rtol=1e-10)

    def test_the_package_works_without_the_extension(self, monkeypatch):
        """
        The build is optional. A user without a compiler has to get the
        same answers, just slower — so the fallback cannot be a path
        that is never exercised.
        """
        import HodlrInverse

        Sigma = covariance(gsm(256, seed=2), 0.5)
        v = np.random.default_rng(2).standard_normal(256)

        compiled = HInverse(Sigma, min_block=64, max_rank=8).matvec(v)

        monkeypatch.setattr(HodlrInverse, "HAVE_FAST", False)
        fallback_model = HInverse(Sigma, min_block=64, max_rank=8)
        assert fallback_model._fast is None

        # Same answer, not merely a plausible one: both paths carry the
        # same rank-8 approximation error, so they have to agree to
        # machine precision rather than to the compression tolerance.
        assert np.allclose(fallback_model.matvec(v), compiled,
                           rtol=1e-12, atol=1e-13)


# ======================================================================
# 3. The rank dial
# ======================================================================
class TestAdaptiveRank:
    """
    Algorithm 1 lines 11-15 grow the rank until the spectral error of
    the truncation meets a tolerance. The released code took one SVD
    and kept whatever cleared a relative threshold, which gives the
    caller no way to ask for an accuracy.
    """

    def test_the_tolerance_is_met_when_it_can_be(self):
        """A genuinely low-rank block should compress to its true rank."""
        rng = np.random.default_rng(3)
        block = rng.standard_normal((64, 5)) @ rng.standard_normal((5, 64))
        U, V = low_rank(block, tolerance=1e-8)
        assert U.shape[1] <= 6
        assert np.linalg.norm(U @ V - block, 2) < 1e-6

    def test_a_full_rank_block_stops_at_the_ceiling(self):
        """
        A GSM's off-diagonal blocks are not low rank. The search must
        stop rather than growing until the factorisation costs more
        than the block it replaces.
        """
        block = np.random.default_rng(4).standard_normal((64, 64))
        U, V = low_rank(block, tolerance=1e-12)
        assert U.shape[1] <= 32          # MAX_RANK_FRACTION of 64

    def test_a_smaller_rank_costs_accuracy_monotonically(self):
        """The dial has to actually turn, in the direction advertised."""
        Sigma = covariance(gsm(256), 0.5)
        residuals = [
            HInverse(Sigma, min_block=64, max_rank=k
                     ).error_bound(Sigma)["relative_residual"]
            for k in (2, 8, 32)
        ]
        assert residuals[0] > residuals[1] > residuals[2]

    def test_the_cost_of_the_setting_is_reported(self):
        """
        A compressed covariance is a deliberate trade, so both sides of
        it belong in the output rather than in a footnote.
        """
        Sigma = covariance(gsm(256), 0.5)
        bound = HInverse(Sigma, min_block=64,
                         max_rank=4).error_bound(Sigma)
        assert bound["max_rank_retained"] == 4
        assert 0 < bound["compression"] < 1
        assert bound["relative_residual"] > 0

    def test_the_approximation_of_sigma_is_symmetric(self):
        """
        ``Sigma~`` is what Assumption 1 bounds; a non-symmetric one
        would not be a covariance.
        """
        Sigma = covariance(gsm(128), 0.5)
        approx = hodlr_approximation(Sigma, min_block=32, max_rank=4)
        assert np.allclose(approx, approx.T)


# ======================================================================
# 3b. Randomised construction
# ======================================================================
class TestRandomisedBuild:
    """
    Building the factors dominates a scan — 89% of the runtime at
    n = 2048, against 10% for every solve put together — because each
    internal node takes a full SVD of its off-diagonal block. A random
    projection gets the same job done about ten times cheaper.

    The textbook objection is that randomised SVD wants a decaying
    spectrum and a GSM has none. That objection does not bite here,
    and the reason is the point: when the spectrum is flat, no
    low-dimensional subspace is much better than another, so the
    optimal truncation is barely better than a sampled one.
    """

    def test_it_costs_little_accuracy(self):
        Sigma = covariance(gsm(512, seed=20), 0.5)
        exact = HInverse(Sigma, min_block=64, max_rank=8)
        sampled = HInverse(Sigma, min_block=64, max_rank=8,
                           randomized=True)
        exact_residual = exact.error_bound(Sigma)["relative_residual"]
        sampled_residual = sampled.error_bound(Sigma)["relative_residual"]
        assert sampled_residual < 1.3 * exact_residual, (
            f"randomised build gave residual {sampled_residual:.4f} "
            f"against {exact_residual:.4f}")

    def test_it_retains_the_requested_rank(self):
        Sigma = covariance(gsm(256, seed=21), 0.5)
        model = HInverse(Sigma, min_block=64, max_rank=4,
                         randomized=True)
        assert model.max_rank_retained() == 4

    def test_it_is_reproducible(self):
        """
        The projection is seeded from the block size, so two builds of
        the same matrix agree. A build that moved between runs would
        make a reported effect size irreproducible.
        """
        Sigma = covariance(gsm(256, seed=22), 0.5)
        v = np.random.default_rng(22).standard_normal(256)
        first = HInverse(Sigma, min_block=64, max_rank=8,
                         randomized=True).matvec(v)
        second = HInverse(Sigma, min_block=64, max_rank=8,
                          randomized=True).matvec(v)
        assert np.array_equal(first, second)

    def test_the_effect_sizes_barely_move(self):
        """
        The measurement that decides whether the trade is acceptable:
        not matrix norms, but how far the reported effects shift
        relative to their own standard errors.
        """
        rng = np.random.default_rng(23)
        n, P = 512, 100
        K = gsm(n, seed=23)
        y = np.linalg.cholesky(covariance(K, 0.5)) @ rng.standard_normal(n)
        X = rng.binomial(2, 0.3, size=(n, P)).astype(float)

        exact = NExt(y, min_block=64, max_rank=8).fit(X, K, h=0.5)
        sampled = NExt(y, min_block=64, max_rank=8,
                       randomized=True).fit(X, K, h=0.5)

        shift = np.abs(sampled.beta - exact.beta) / np.abs(exact.se)
        assert np.median(shift) < 0.25, (
            f"median shift {np.median(shift):.3f} standard errors")


# ======================================================================
# 4. The approximate inverse stays a valid weight matrix
# ======================================================================
class TestApproximateInverseIsBenign:
    """
    Why an error of 0.2 in spectral norm costs almost nothing in
    inference: the approximate inverse is still symmetric positive
    definite, so GLS is still using a *valid* weight — just not the
    optimal one. GLS with any positive-definite weight is unbiased, and
    the residual variance computed through the same matrix keeps the
    standard error honest.
    """

    @pytest.mark.parametrize("max_rank", [2, 8, 32])
    def test_it_is_symmetric(self, max_rank):
        Sigma = covariance(gsm(256), 0.5)
        dense = HInverse(Sigma, min_block=64, max_rank=max_rank).to_dense()
        assert np.abs(dense - dense.T).max() < 1e-10

    @pytest.mark.parametrize("max_rank", [2, 8, 32])
    def test_it_is_positive_definite(self, max_rank):
        Sigma = covariance(gsm(256), 0.5)
        dense = HInverse(Sigma, min_block=64, max_rank=max_rank).to_dense()
        assert np.linalg.eigvalsh((dense + dense.T) / 2).min() > 0


# ======================================================================
# 5. The scan
# ======================================================================
class TestScan:
    def test_the_fast_path_matches_an_exact_dense_solve(self):
        """
        With compression switched off, the batched algebra has to be
        exact — otherwise a later disagreement cannot be attributed to
        the approximation.
        """
        rng = np.random.default_rng(6)
        n, P = 100, 20
        K = gsm(n, seed=6)
        y = rng.standard_normal(n)
        X = rng.binomial(2, 0.3, size=(n, P)).astype(float)

        model = NExt(y, min_block=2 * n).fit(X, K, h=0.4)
        Sigma_inv = np.linalg.inv(covariance(K, 0.4))
        for j in range(P):
            beta, se, _ = dense_gls(Sigma_inv, X[:, j], y)
            assert model.beta[j] == pytest.approx(beta, rel=1e-8)
            assert model.se[j] == pytest.approx(se, rel=1e-8)

    def test_a_monomorphic_marker_is_nan_not_a_finding(self):
        """
        A constant column is collinear with the intercept, so it has no
        slope. Returning a number there would put a spurious effect in
        the output.
        """
        rng = np.random.default_rng(7)
        n = 100
        X = rng.binomial(2, 0.3, size=(n, 3)).astype(float)
        X[:, 1] = 1.0
        model = NExt(rng.standard_normal(n), min_block=2 * n).fit(
            X, gsm(n, seed=7), h=0.3)
        assert np.isnan(model.beta[1])
        assert np.isnan(model.association()[1])
        assert not np.isnan(model.beta[0])

    def test_association_before_fit_is_refused(self):
        with pytest.raises(RuntimeError, match="fit"):
            NExt(np.ones(10)).association()


# ======================================================================
# 6. PCGC
# ======================================================================
class TestPcgc:
    """
    Equation (5) is a no-intercept least squares through the origin
    with the solution clamped to [0, 1]. The released code used
    ``LinearRegression()`` with its default intercept and no clamp.
    """

    def test_it_matches_the_closed_form(self):
        rng = np.random.default_rng(8)
        n = 150
        K = gsm(n, seed=8)
        y = rng.standard_normal(n)

        standardised = (y - y.mean()) / y.std()
        i, j = np.triu_indices(n, k=1)
        expected = ((K[i, j] @ (standardised[i] * standardised[j]))
                    / (K[i, j] @ K[i, j]))
        assert NExt(y).pcgc(K) == pytest.approx(
            float(np.clip(expected, 0, 1)))

    def test_the_estimate_is_clamped_to_a_valid_range(self):
        """
        A variance ratio is only defined on [0, 1]. An unclamped fit can
        leave it, and the covariance built from it is then not a
        covariance.
        """
        rng = np.random.default_rng(9)
        n = 80
        K = gsm(n, seed=9)
        for seed in range(10):
            y = np.random.default_rng(seed).standard_normal(n) * 50
            assert 0.0 <= NExt(y).pcgc(K) <= 1.0

    def test_a_mismatched_k_is_refused(self):
        with pytest.raises(ValueError, match="to match Y"):
            NExt(np.ones(10)).pcgc(np.eye(5))

    def test_a_constant_phenotype_is_refused(self):
        with pytest.raises(ValueError, match="constant"):
            NExt(np.ones(20)).pcgc(gsm(20))


# ======================================================================
# 7. The variance-ratio parameterisation
# ======================================================================
class TestVarianceRatio:
    """
    The paper uses ``Sigma = lambda K + I`` with
    ``lambda = h2 / (1 - h2)``; the released code used
    ``Sigma = h2 K + (1 - h2) I``. The two differ by a constant factor,
    which GLS ignores for ``beta`` and does not ignore for
    ``sigma_e^2``.
    """

    def test_beta_is_invariant_to_the_scaling_but_sigma_is_not(self):
        rng = np.random.default_rng(10)
        n = 120
        K = gsm(n, seed=10)
        y = rng.standard_normal(n)
        x = rng.binomial(2, 0.3, size=n).astype(float)
        h2 = 0.5

        paper = np.linalg.inv(covariance(K, h2))
        released = np.linalg.inv(h2 * K + (1 - h2) * np.eye(n))

        b_paper, _, s2_paper = dense_gls(paper, x, y)
        b_released, _, s2_released = dense_gls(released, x, y)

        assert b_paper == pytest.approx(b_released, rel=1e-10)
        assert s2_released / s2_paper == pytest.approx(1 / (1 - h2),
                                                       rel=1e-6)

    def test_h2_of_one_is_refused(self):
        with pytest.raises(ValueError, match="undefined"):
            NExt(np.ones(10)).factorise(np.eye(10), 1.0)


# ======================================================================
# 8. Calibration and power  (slow)
# ======================================================================
class TestInference:
    """
    The checks that would have caught the association statistic. A
    wrong denominator leaves the point estimates untouched and the
    trace plots clean; only a known-null simulation and a known-signal
    one separate it from the right answer.
    """

    @staticmethod
    def _simulate(n=256, P=60, h2=0.5, n_causal=0, effect=0.0, seed=0,
                  reps=60):
        rng = np.random.default_rng(seed)
        K = gsm(n, seed=seed)
        chol = np.linalg.cholesky(covariance(K, h2))
        X = rng.binomial(2, 0.3, size=(n, P)).astype(float)
        for _ in range(reps):
            y = chol @ rng.standard_normal(n)
            if n_causal:
                y = y + X[:, :n_causal].sum(axis=1) * effect
            yield K, X, y

    @pytest.mark.slow
    @pytest.mark.parametrize("max_rank", [2, 32, None])
    def test_the_null_is_calibrated_at_every_rank(self, max_rank):
        """
        The claim the rank dial rests on: compression costs accuracy in
        the matrix, not validity in the test.
        """
        p_values = np.concatenate([
            NExt(y, min_block=64, max_rank=max_rank
                 ).fit(X, K, h=0.5).association()
            for K, X, y in self._simulate(seed=11)
        ])
        rate = float((p_values < 0.05).mean())
        assert 0.03 < rate < 0.08, f"type-I {rate:.4f} at rank {max_rank}"
        assert stats.kstest(p_values, "uniform").pvalue > 0.01

    @pytest.mark.slow
    def test_the_wald_statistic_beats_dividing_by_the_spread(self):
        """
        The released statistic put real signal into its own denominator.
        Measured here on 24 causal markers out of 60; on a denser
        architecture the gap widens, which matters because the paper
        attributes the power drop at high causal fractions to genetic
        architecture alone.
        """
        n, P, n_causal = 256, 60, 24
        wald_hits = spread_hits = 0
        for K, X, y in self._simulate(n=n, P=P, n_causal=n_causal,
                                      effect=0.25, seed=12, reps=40):
            model = NExt(y, min_block=64).fit(X, K, h=0.5)
            causal = slice(0, n_causal)
            wald_hits += int((model.association()[causal] < 0.05).sum())

            spread = model.beta / np.std(model.beta)
            released = 2 * stats.t.sf(np.abs(spread), n - 1)
            spread_hits += int((released[causal] < 0.05).sum())

        assert wald_hits > 1.4 * spread_hits, (
            f"Wald found {wald_hits}, the released statistic "
            f"{spread_hits}")

    @pytest.mark.slow
    def test_a_coarse_rank_keeps_most_of_the_power(self):
        """What the dial costs where it matters, not in matrix norm."""
        powers = {}
        for max_rank in (2, None):
            hits = 0
            for K, X, y in self._simulate(n_causal=24, effect=0.25,
                                          seed=13, reps=40):
                p_values = NExt(y, min_block=64, max_rank=max_rank
                                ).fit(X, K, h=0.5).association()
                hits += int((p_values[:24] < 0.05).sum())
            powers[max_rank] = hits
        assert powers[2] > 0.85 * powers[None], powers
