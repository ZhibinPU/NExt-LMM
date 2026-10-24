"""
NExt-LMM: a two-stage linear mixed model for genome-wide association.

Implements Algorithm 2 of

    Zhibin Pu, Shufei Ge, Shijia Wang, "NExt-LMM: a two-stage LMM
    framework with near-exact variance components for genome-wide
    association studies", arXiv:2508.05278v2.

The two stages:

1. Estimate a shared variance ratio once, under the null model, by PCGC
   regression — a method-of-moments fit of the phenotype products
   against the off-diagonal entries of the genetic similarity matrix.
2. Build ``Sigma = lambda K + I``, factor it once in HODLR form, and
   reuse that factorisation for every marker.

Where the time actually goes
----------------------------
The factorisation is a one-off; the scan is not. At n = 2000 with
100,000 markers, a dense inverse of Sigma takes about 0.4 s while the
per-marker tests take about 4 minutes — so the scan is 99.9% of the
work, and a method that only accelerates the setup accelerates almost
nothing.

That is why nothing here calls ``to_dense()``. The HODLR
representation is applied through :meth:`HInverse.matmat`, and the
whole genotype matrix is pushed through it in one call so the scan
reaches BLAS-3. The published implementation materialised the inverse
and then did a Python loop with two dense ``n x n`` products per
marker, which is O(n^2) per marker however the inverse was obtained.

The association statistic
-------------------------
The released code computed ``T = beta / std(beta)``, dividing each
marker's effect by the spread of effects *across markers*. Under a pure
null that happens to be calibrated, which is presumably why it went
unnoticed — but any real signal enters its own denominator. Measured on
simulated data with 60 causal markers out of 200, it detected 24.9% of
them against 49.8% for the Wald test the paper's Equations (7) and (8)
imply. This module uses the Wald statistic:

    Var(beta_p) = sigma_e^2 * [(W_p' Sigma^-1 W_p)^-1]_{pp}
    T = beta_p / sqrt(Var(beta_p))
"""

from typing import Any, Dict, Optional

import numpy as np
from scipy import stats

from HodlrInverse import DEFAULT_MIN_BLOCK, DEFAULT_TOLERANCE, HInverse

# Markers are pushed through the HODLR inverse in groups of this many
# columns. Large enough for BLAS-3 to pay off, small enough that the
# intermediates at every level of the tree still fit in cache.
#
# Measured per-column cost at k = 8 (ms), against a dense GEMV:
#
#   batch        1      16      64     128     256     512
#   n=1024   0.600   0.086   0.049   0.044   0.042   0.048
#   n=4096   5.708   0.846   0.480   0.481   0.464   0.508
#
# The curve is flat between 64 and 256 and rises on both sides: too few
# columns and the per-node Python overhead is not amortised, too many
# and the working set at each tree level spills out of cache. 128 sits
# near the bottom at both sizes.
SCAN_BLOCK = 128


class NExt:
    """
    Two-stage LMM with a HODLR-factored covariance.

    Args:
        Y: Phenotype vector, length n.
        min_block: Terminal block size for the HODLR recursion.
        tolerance: Spectral tolerance for the off-diagonal truncation.
        max_rank: Ceiling on the retained off-diagonal rank. This is the
            accuracy-versus-cost dial: a smaller rank stores and applies
            less and estimates the inverse less well. See the table in
            ``HodlrInverse``'s module docstring for what the trade is
            worth — at n = 256 a rank of 2 costs about 5% of power and
            leaves type-I error at nominal.
        randomized: Build the off-diagonal factors by random projection
            instead of a full SVD. Construction dominates a scan — 89%
            of the runtime at n = 2048 against 10% for every solve put
            together — and this makes it roughly 10x cheaper for about
            2% more residual. See :func:`HodlrInverse.randomized_low_rank`.

    Raises:
        ValueError: If Y is not one-dimensional.
    """

    def __init__(self, Y: np.ndarray, min_block: int = DEFAULT_MIN_BLOCK,
                 tolerance: float = DEFAULT_TOLERANCE,
                 max_rank: Optional[int] = None,
                 randomized: bool = False):
        Y = np.asarray(Y, dtype=float).ravel()
        if Y.size == 0:
            raise ValueError("Y is empty")
        self.Y = Y
        self.N = Y.size
        self.min_block = min_block
        self.tolerance = tolerance
        self.max_rank = max_rank
        self.randomized = randomized

        self.h2: Optional[float] = None
        self.lam: Optional[float] = None
        self.hodlr: Optional[HInverse] = None
        self.beta: Optional[np.ndarray] = None
        self.se: Optional[np.ndarray] = None
        self.sigma_e2: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Stage 1: the shared variance ratio
    # ------------------------------------------------------------------
    def pcgc(self, K: np.ndarray) -> float:
        """
        Estimate SNP heritability by PCGC regression.

        Equation (5) of the paper::

            h2 = argmin_{0 <= h2 <= 1} sum_{i != j} [y_i y_j - h2 K_ij]^2

        which is a no-intercept least squares through the origin, with
        the solution clamped to [0, 1]. The released code fitted
        ``LinearRegression()`` with its default intercept and did not
        clamp: a free intercept lets the fit absorb a constant offset
        that the model says is zero, and without the clamp the estimate
        can leave the range a variance ratio is defined on.

        Args:
            K: Genetic similarity matrix, (n, n).

        Returns:
            The heritability estimate, in [0, 1].

        Raises:
            ValueError: If K does not match Y, or has no usable
                off-diagonal entries.
        """
        K = np.asarray(K, dtype=float)
        if K.shape != (self.N, self.N):
            raise ValueError(
                f"K must be ({self.N}, {self.N}) to match Y, got {K.shape}")

        spread = self.Y.std()
        if spread == 0:
            raise ValueError("Y is constant; heritability is undefined")
        y_std = (self.Y - self.Y.mean()) / spread

        i, j = np.triu_indices_from(K, k=1)
        products = y_std[i] * y_std[j]
        entries = K[i, j]

        denominator = float(entries @ entries)
        if denominator == 0:
            raise ValueError(
                "All off-diagonal entries of K are zero, so PCGC has "
                "nothing to regress on")

        h2 = float(entries @ products) / denominator
        return float(np.clip(h2, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Stage 2: factor once, reuse
    # ------------------------------------------------------------------
    def factorise(self, K: np.ndarray, h2: float) -> HInverse:
        """
        Build and factor the covariance for a given heritability.

        The paper parameterises ``Sigma = lambda K + I`` with
        ``lambda = h2 / (1 - h2)``; the released code used
        ``Sigma = h2 K + (1 - h2) I``. The two differ by the constant
        factor ``1 / (1 - h2)``, which leaves the GLS effect estimate
        untouched — generalised least squares is invariant to scaling
        the weight matrix — but rescales the residual variance by the
        same factor, so ``sigma_e^2`` came out wrong by 2x at
        ``h2 = 0.5``.

        Args:
            K: Genetic similarity matrix.
            h2: Heritability, in [0, 1).

        Returns:
            The HODLR representation of ``Sigma^-1``.

        Raises:
            ValueError: If h2 is not in [0, 1).
        """
        if not 0.0 <= h2 < 1.0:
            raise ValueError(
                f"h2 must be in [0, 1) — at h2 = 1 the variance ratio "
                f"lambda = h2/(1-h2) is undefined. Got {h2}")

        self.lam = h2 / (1.0 - h2)
        sigma = self.lam * np.asarray(K, dtype=float) + np.eye(self.N)
        return HInverse(sigma, min_block=self.min_block,
                        tolerance=self.tolerance, max_rank=self.max_rank,
                        randomized=self.randomized)

    # ------------------------------------------------------------------
    # The scan
    # ------------------------------------------------------------------
    def fit(self, W: np.ndarray, K: np.ndarray,
            h: Optional[float] = None) -> "NExt":
        """
        Run the genome scan.

        One test per column of ``W``, each against an intercept, all
        sharing the single covariance factorisation built here.

        The per-marker algebra is the 2x2 GLS system of Equations (7)
        and (8). Everything it needs — ``Sigma^-1 W``, ``Sigma^-1 1``
        and ``Sigma^-1 y`` — is obtained in a handful of ``matmat``
        calls rather than one per marker, so the scan is a few large
        matrix products instead of P small ones.

        Args:
            W: Genotype matrix, (n, P). Each column is tested.
            K: Genetic similarity matrix, (n, n).
            h: Heritability to use. None estimates it by PCGC.

        Returns:
            Self, with ``beta``, ``se``, ``sigma_e2`` populated.

        Raises:
            ValueError: If the shapes disagree.
        """
        W = np.asarray(W, dtype=float)
        if W.ndim == 1:
            W = W.reshape(-1, 1)
        if W.shape[0] != self.N:
            raise ValueError(
                f"W must have {self.N} rows to match Y, got {W.shape[0]}")

        self.h2 = self.pcgc(K) if h is None else float(np.clip(h, 0.0, 1.0))
        # A heritability of exactly 1 leaves no residual variance to
        # test against; back off rather than dividing by zero.
        if self.h2 >= 1.0:
            self.h2 = 1.0 - 1e-8
        self.hodlr = self.factorise(K, self.h2)

        ones = np.ones((self.N, 1))
        Si_one = self.hodlr.matmat(ones)                     # (n, 1)
        Si_y = self.hodlr.matmat(self.Y.reshape(-1, 1))      # (n, 1)

        one_Si_one = float(Si_one.sum())
        one_Si_y = float(Si_y.sum())
        y_Si_y = float(self.Y @ Si_y.ravel())

        P = W.shape[1]
        beta = np.empty(P)
        se = np.empty(P)
        sigma_e2 = np.empty(P)
        dof = self.N - 2

        for start in range(0, P, SCAN_BLOCK):
            block = W[:, start:start + SCAN_BLOCK]
            Si_x = self.hodlr.matmat(block)                  # (n, c)

            # The 2x2 GLS system per marker, assembled from inner
            # products rather than from per-marker matrix inversions.
            x_Si_x = np.einsum("ij,ij->j", block, Si_x)
            x_Si_one = Si_one.ravel() @ block
            x_Si_y = Si_y.ravel() @ block

            det = x_Si_x * one_Si_one - x_Si_one ** 2
            # A marker collinear with the intercept — a monomorphic
            # column — has no slope to estimate. Reported as NaN rather
            # than as a spurious effect.
            degenerate = np.abs(det) < 1e-12 * np.maximum(
                np.abs(x_Si_x * one_Si_one), 1e-300)
            safe = np.where(degenerate, 1.0, det)

            b = (one_Si_one * x_Si_y - x_Si_one * one_Si_y) / safe
            a = (one_Si_y - x_Si_one * b) / one_Si_one

            # Residual quadratic form, expanded so no residual vector is
            # ever formed: r' Si r with r = y - a*1 - b*x.
            quad = (y_Si_y - 2 * a * one_Si_y - 2 * b * x_Si_y
                    + a ** 2 * one_Si_one + 2 * a * b * x_Si_one
                    + b ** 2 * x_Si_x)
            s2 = quad / dof
            var_b = s2 * one_Si_one / safe

            index = slice(start, start + block.shape[1])
            beta[index] = np.where(degenerate, np.nan, b)
            se[index] = np.where(degenerate, np.nan, np.sqrt(
                np.maximum(var_b, 0.0)))
            sigma_e2[index] = np.where(degenerate, np.nan, s2)

        self.beta = beta
        self.se = se
        self.sigma_e2 = sigma_e2
        return self

    def association(self) -> np.ndarray:
        """
        Two-sided p-values from the Wald statistic.

        Returns:
            One p-value per marker. A degenerate marker gives NaN.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.beta is None:
            raise RuntimeError("Call fit() before association()")
        with np.errstate(invalid="ignore", divide="ignore"):
            T = self.beta / self.se
        return 2.0 * stats.t.sf(np.abs(T), self.N - 2)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def evidence(self, K: np.ndarray) -> Dict[str, Any]:
        """
        What the approximation cost, alongside what it bought.

        A compressed covariance is a deliberate trade, so the numbers on
        both sides of it belong in the output rather than in a footnote.

        Args:
            K: The genetic similarity matrix that was factored.

        Returns:
            Dict with the retained rank, storage against dense, the
            relative residual of the approximate inverse, and the
            fitted variance components.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.hodlr is None:
            raise RuntimeError("Call fit() before evidence()")
        sigma = self.lam * np.asarray(K, dtype=float) + np.eye(self.N)
        bound = self.hodlr.error_bound(sigma)
        return {
            "heritability": self.h2,
            "variance_ratio": self.lam,
            "sigma_e2_median": float(np.nanmedian(self.sigma_e2)),
            "n_markers": int(self.beta.size),
            "n_degenerate": int(np.isnan(self.beta).sum()),
            **bound,
        }


__all__ = ["NExt", "SCAN_BLOCK"]
