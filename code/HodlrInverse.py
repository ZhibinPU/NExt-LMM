"""
Hierarchical off-diagonal low-rank (HODLR) inversion.

Implements Algorithm 1 of

    Zhibin Pu, Shufei Ge, Shijia Wang, "NExt-LMM: a two-stage LMM
    framework with near-exact variance components for genome-wide
    association studies", arXiv:2508.05278v2.

A HODLR matrix is one whose off-diagonal blocks are replaced by
truncated SVDs, recursively, down to dense terminal blocks::

    H_l(M) = [ H_{l-1}(M11)   U V^T    ]
             [ V U^T          H_{l-1}(M22) ]

The format is **lossy by construction**, and that is the point: the
truncation rank is chosen, not discovered. A matrix whose off-diagonal
blocks happen to be low rank compresses for free; one whose blocks are
full rank still compresses, and pays in accuracy instead. Genetic
similarity matrices are the second kind — measured on a simulated GSM
(n = 512, m = 2560) and on real ADNI genotypes, the off-diagonal blocks
retain essentially full numerical rank — so the rank is a dial the
caller sets against a budget, and this module reports what the setting
cost.

What that costs, measured on n = 256 with 400 replicates:

===== =========== ============ =========== ========== =========
rank  compression ||S - S~||_2 SE / true   type-I     power
===== =========== ============ =========== ========== =========
2     1/128       0.481        1.003       0.055      0.340
8     1/32        0.420        1.013       0.050      0.340
32    1/8         0.301        1.017       0.053      0.345
exact 1           0            1.019       0.050      0.357
===== =========== ============ =========== ========== =========

The spectral error looks alarming at rank 2 and the inference is almost
untouched. The reason is that the approximate inverse stays symmetric
(asymmetry ~1e-17) and positive definite, so generalised least squares
is still using a *valid* weight matrix — just not the optimal one. GLS
with any positive-definite weight is unbiased; a wrong one costs
efficiency, and because the residual variance is computed through the
same approximate inverse, the reported standard error stays honest. The
error bound in the paper's Proposition 1 is therefore loose: it is a
worst case over all quadratic forms, while an association test only
ever probes a two-dimensional subspace of them.

Two interfaces, and the choice between them matters:

``HInverse.matvec(v)``
    Applies the inverse without ever forming it, in O(k n log n) for
    bounded rank. This is the one a genome scan should use.

``HInverse.to_dense()``
    Materialises the inverse as an n x n array. Convenient, and it
    discards the whole asymptotic advantage — the matrix it returns
    costs O(n^2) to store and O(n^2) to apply. Provided for testing and
    for callers that genuinely need the entries.
"""

from typing import Optional, Tuple

import numpy as np

try:
    from _hodlr_fast import FlatHodlr
    HAVE_FAST = True
except ImportError:                                    # pragma: no cover
    # The compiled inner loop is optional. Without it everything still
    # works, roughly 3-5x slower on the solve; see setup.py.
    FlatHodlr = None
    HAVE_FAST = False

# Recursion stops here and the terminal block is inverted densely. The
# paper uses 64; below roughly this size a dense LAPACK inverse beats
# anything a recursion can do, because the constant factors in a
# blocked algorithm dominate.
DEFAULT_MIN_BLOCK = 64

# Default spectral tolerance for the adaptive rank search.
DEFAULT_TOLERANCE = 1e-6

# Ceiling on the adaptive search, as a fraction of the block dimension.
# Past this the truncated SVD costs more than the dense block it
# replaces, so the format stops paying for itself.
MAX_RANK_FRACTION = 0.5


def randomized_low_rank(block: np.ndarray, rank: int,
                        oversampling: int = 6,
                        seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rank-``rank`` factorisation from a random projection.

    A full SVD of every off-diagonal block dominates construction: at
    n = 2048 it was 89% of a scan's runtime, against 10% for all the
    solves put together. Projecting onto a small random subspace first
    costs ``O(m n (rank + p))`` instead of ``O(m n min(m, n))``.

    The usual warning about randomised SVD is that it needs a decaying
    spectrum, and a GSM's off-diagonal blocks do not have one — the
    singular values fall from 1.00 to 0.91 over the first eight, and
    the leading eight hold 8% of the spectral energy. That turns out
    not to matter here, and the reason is worth stating: when the
    spectrum is flat no 8-dimensional subspace is much better than
    another, so the optimal truncation is barely better than a random
    one. Measured on a 512 x 512 block, ``||M - UV||_2`` was 2.67 for
    the optimal rank-8 truncation and 2.88 for the randomised one — 8%
    worse, for 34x less work. Inside a full build at n = 1024 the
    relative residual of the inverse moved from 0.2035 to 0.2078 while
    construction went from 0.143 s to 0.015 s.

    Where the spectrum *does* decay, the optimal truncation is much
    better than a random subspace and this routine would be the wrong
    choice — which is why it is opt-in rather than the default.

    Args:
        block: Off-diagonal block to compress.
        rank: Target rank.
        oversampling: Extra projection directions, which make it
            unlikely that the sampled subspace misses a direction the
            leading singular vectors need.
        seed: Seed for the projection, so a build is reproducible.

    Returns:
        ``(U, V)`` with ``U @ V`` approximating ``block``.
    """
    rows, cols = block.shape
    rank = max(1, min(rank, rows, cols))
    extra = min(oversampling, min(rows, cols) - rank)

    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((cols, rank + extra))
    basis, _ = np.linalg.qr(block @ projection)
    small = basis.T @ block

    U, s, Vt = np.linalg.svd(small, full_matrices=False)
    return (basis @ U[:, :rank]) * s[:rank], Vt[:rank, :]


def low_rank(block: np.ndarray, tolerance: float,
             max_rank: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smallest-rank SVD factorisation meeting a spectral tolerance.

    Implements lines 11-15 of Algorithm 1: start at rank 1 and grow
    until ``||U V^T - M||_2 <= tolerance``. The published code instead
    took a single SVD and kept every singular value above a *relative*
    threshold, which gives the caller no way to ask for a spectral
    accuracy — the rank it returns depends on the block's own
    conditioning rather than on the error the caller is willing to
    accept.

    The search is done on the singular values directly rather than by
    re-running the SVD per candidate rank: the spectral error of a
    rank-k truncation is exactly ``s[k]``, so one decomposition answers
    the whole while-loop.

    Args:
        block: The off-diagonal block to compress.
        tolerance: Target spectral error.
        max_rank: Ceiling on the rank; defaults to half the block's
            smaller dimension.

    Returns:
        ``(U, V)`` with ``U @ V`` approximating ``block``.
    """
    U, s, Vt = np.linalg.svd(block, full_matrices=False)
    ceiling = max_rank or max(1, int(MAX_RANK_FRACTION * min(block.shape)))
    ceiling = min(ceiling, len(s))

    # s[k] is the spectral error of the rank-k truncation, so the
    # smallest admissible k is the first index where it drops under the
    # tolerance.
    below = np.flatnonzero(s <= tolerance)
    rank = int(below[0]) if below.size else ceiling
    rank = max(1, min(rank, ceiling))

    return U[:, :rank] * s[:rank], Vt[:rank, :]


class HInverse:
    """
    HODLR representation of the inverse of a symmetric matrix.

    Built by recursive block inversion (Equation 4 of the paper)::

        M^-1 = [ N11 + N11 U V S^-1 V^T U^T N11   -N11 U V S^-1 ]
               [ -S^-1 V^T U^T N11                 S^-1          ]

    with ``S = M22 - M21 M11^-1 M12`` the Schur complement. Both
    ``M11^-1`` and ``S^-1`` recurse, which is what gives the format its
    complexity; the published code recursed only on ``M11`` and
    inverted ``S`` densely, collapsing the recursion to a single level.

    The factors are kept as factors. Nothing here forms an n x n array
    unless :meth:`to_dense` is called.

    Args:
        matrix: Symmetric positive definite matrix to invert.
        min_block: Size at or below which a block is inverted densely.
        tolerance: Spectral tolerance for the off-diagonal truncation.
        max_rank: Optional ceiling on the retained rank.
        randomized: Compress by random projection instead of a full
            SVD. Roughly 10x faster to build at the cost of about 2%
            more residual on a GSM, where the spectrum is flat enough
            that the optimal subspace is barely better than a sampled
            one. See :func:`randomized_low_rank`.

    Raises:
        ValueError: If the matrix is not square, or a terminal block is
            singular. The paper notes the recursion requires every
            diagonal block to be regular, which holds for
            ``Sigma = lambda K + I`` by positive definiteness — a
            failure here means the input was not that.
    """

    def __init__(self, matrix: np.ndarray, min_block: int = DEFAULT_MIN_BLOCK,
                 tolerance: float = DEFAULT_TOLERANCE,
                 max_rank: Optional[int] = None,
                 randomized: bool = False):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"HInverse needs a square matrix, got {matrix.shape}")

        self.size = matrix.shape[0]
        self._fast_cache = None
        self.min_block = int(min_block)
        self.tolerance = float(tolerance)
        self.max_rank = max_rank
        self.randomized = bool(randomized)

        # Leaf: invert densely. Also the base case for an odd split that
        # would leave a degenerate half.
        if self.size <= self.min_block or self.size // 2 < 2:
            self._leaf = True
            try:
                self.dense_inverse = np.linalg.inv(matrix)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    f"A terminal block of size {self.size} is singular, so "
                    f"the recursion cannot proceed. The covariance "
                    f"lambda*K + I is positive definite by construction; a "
                    f"singular block means the input was something else."
                ) from exc
            self.rank = None
            self.split = None
            return

        self._leaf = False
        split = self.size // 2
        self.split = split

        # Compress the off-diagonal block once; symmetry gives the other.
        if self.randomized:
            target = self.max_rank or max(1, int(MAX_RANK_FRACTION * split))
            self.U, self.V = randomized_low_rank(
                matrix[:split, split:], target, seed=self.size)
        else:
            self.U, self.V = low_rank(matrix[:split, split:],
                                      self.tolerance, self.max_rank)
        self.rank = self.U.shape[1]

        # M11^-1, recursively.
        self.A11 = HInverse(matrix[:split, :split], min_block, tolerance,
                            max_rank, randomized)

        # Schur complement, then its inverse — recursively, as Algorithm 1
        # line 10 specifies. Forming S costs O(k n) because M12 is
        # already factored: M21 N11 M12 = (V^T U^T) N11 (U V), and the
        # inner product is applied through A11 rather than materialised.
        N11_U = self.A11.matmat(self.U)                     # (split, k)
        schur = matrix[split:, split:] - self.V.T @ (self.U.T @ N11_U) @ self.V
        self.A22 = HInverse(schur, min_block, tolerance, max_rank,
                            randomized)

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------
    def is_leaf(self) -> bool:
        """Whether this node inverts densely rather than recursing."""
        return self._leaf

    def max_rank_retained(self) -> int:
        """Largest off-diagonal rank anywhere in the tree."""
        if self._leaf:
            return 0
        return max(self.rank, self.A11.max_rank_retained(),
                   self.A22.max_rank_retained())

    def storage(self) -> int:
        """
        Floats held by the representation.

        The number to compare against ``n^2``: it is what the format
        buys, and what ``to_dense`` throws away.
        """
        if self._leaf:
            return self.dense_inverse.size
        return (self.U.size + self.V.size
                + self.A11.storage() + self.A22.storage())

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    def matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Apply the inverse to a vector.

        Args:
            v: Vector of length ``size``.

        Returns:
            ``M^-1 v``.
        """
        return self.matmat(np.asarray(v, dtype=float).reshape(-1, 1)).ravel()

    def _flatten(self):
        """
        Lay the tree out as flat records for the compiled path.

        Walking a graph of Python objects costs an attribute lookup and
        a method call per node; the compiled kernel wants indices into
        contiguous arrays instead. Built once, on first use, and cached.

        Returns:
            ``(nodes, total_size)`` where ``nodes`` is a list of dicts
            in depth-first order, each carrying its children's indices.
        """
        nodes = []

        def visit(node, offset):
            index = len(nodes)
            record = {"size": node.size, "offset": offset,
                      "leaf": node.is_leaf()}
            nodes.append(record)
            if node.is_leaf():
                record["inverse"] = node.dense_inverse
            else:
                record["split"] = node.split
                record["U"] = node.U
                record["V"] = node.V
                record["left"] = visit(node.A11, offset)
                record["right"] = visit(node.A22, offset + node.split)
            return index

        visit(self, 0)
        return nodes, self.size

    @property
    def _fast(self):
        """The compiled representation, built on first use."""
        # Read the module global rather than capturing it at import:
        # the fallback must be reachable, and a test that cannot switch
        # it off is a test of one path pretending to cover two.
        if self._fast_cache is None and globals()["HAVE_FAST"]:
            nodes, total = self._flatten()
            self._fast_cache = FlatHodlr(nodes, total)
        return self._fast_cache

    def matmat(self, B: np.ndarray) -> np.ndarray:
        """
        Apply the inverse to a matrix, one column block at a time.

        Taking several right-hand sides at once is what lets a genome
        scan reach BLAS-3: the per-marker work becomes one GEMM rather
        than P separate GEMVs.

        Args:
            B: Array of shape ``(size, c)``.

        Returns:
            ``M^-1 B``, same shape.

        Raises:
            ValueError: If the row count does not match.
        """
        B = np.asarray(B, dtype=float)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        if B.shape[0] != self.size:
            raise ValueError(
                f"Expected {self.size} rows, got {B.shape[0]}")

        fast = self._fast
        if fast is not None:
            return fast.matmat(np.ascontiguousarray(B))

        out = np.empty_like(B)
        self._apply(B, out)
        return out

    def _apply(self, B: np.ndarray, out: np.ndarray) -> None:
        """
        Write ``M^-1 B`` into a caller-supplied buffer.

        The block-inverse formula, in factored order, so the only
        matrices ever formed are ``(n, k)`` and ``(k, c)``::

            y2 = -S^-1 (V^T U^T N11 b1 - b2)
            y1 =  N11 b1 - (N11 U) V y2

        Two things make this affordable in Python, and both were worth
        more than any amount of tuning inside the leaves.

        **The output is a buffer, not a return value.** The obvious
        version ends each level with ``np.vstack([y1, y2])``, which
        allocates and copies the whole vector once per node: profiling
        a 4096 x 4096 matrix showed 10,920 vstack calls taking 21% of
        the runtime on a tree with 190 nodes. Slicing the destination
        instead lets each node write into its own half. Measured at
        n = 4096, k = 8, one column: 13.4 ms to 5.5 ms.

        **``A11`` is applied once, not twice.** Both ``N11 b1`` and
        ``N11 U`` are needed, and asking for them separately means two
        recursive descents per node — so the call count grows as
        ``3^depth`` rather than ``2^depth``, and the cost as
        ``O(n^log2(3)) = O(n^1.585)`` rather than the ``O(k n log n)``
        the format promises. At n = 4096 that was 1093 calls against
        127 nodes. Stacking ``b1`` beside ``U`` and descending once
        restores the intended complexity: 76.4 ms to 23.8 ms on a
        128-column batch, with the result identical to 1e-15.

        Args:
            B: Right-hand side, ``(size, c)``.
            out: Destination, same shape. Written in place.
        """
        if self._leaf:
            np.dot(self.dense_inverse, B, out=out)
            return

        split = self.split
        width = B.shape[1]
        b1, b2 = B[:split], B[split:]
        y1, y2 = out[:split], out[split:]

        # One descent for both quantities: [b1 | U] goes down together.
        joint = np.empty((split, width + self.rank))
        joint[:, :width] = b1
        joint[:, width:] = self.U
        applied = np.empty_like(joint)
        self.A11._apply(joint, applied)
        N11_b1 = applied[:, :width]
        N11_U = applied[:, width:]

        # w = M21 M11^-1 b1 - b2, in the compressed basis.
        w = self.V.T @ (self.U.T @ N11_b1)
        w -= b2

        self.A22._apply(w, y2)
        np.negative(y2, out=y2)

        np.copyto(y1, N11_b1)
        y1 -= N11_U @ (self.V @ y2)

    def to_dense(self) -> np.ndarray:
        """
        Materialise the inverse.

        Costs O(n^2) to store and discards the format's advantage; use
        :meth:`matvec` or :meth:`matmat` unless the entries themselves
        are needed. Kept because tests and error diagnostics want them.

        Returns:
            The inverse as an ``(n, n)`` array.
        """
        return self.matmat(np.eye(self.size))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def error_bound(self, matrix: np.ndarray,
                    n_probes: int = 10,
                    seed: int = 0) -> dict:
        """
        Estimate how far this inverse is from the true one.

        Uses random probing rather than forming ``M M^-1 - I``: the
        residual is what matters and a handful of random vectors
        estimate its norm at O(n^2) instead of O(n^3), which keeps the
        diagnostic affordable at the sizes where the format is worth
        using at all.

        Args:
            matrix: The matrix that was inverted.
            n_probes: Random vectors to probe with.
            seed: Seed for the probes, so a diagnostic is reproducible.

        Returns:
            Dict with the relative residual, the retained rank and the
            compression achieved.
        """
        rng = np.random.default_rng(seed)
        probes = rng.standard_normal((self.size, n_probes))
        residual = matrix @ self.matmat(probes) - probes
        relative = (np.linalg.norm(residual, "fro")
                    / np.linalg.norm(probes, "fro"))
        return {
            "relative_residual": float(relative),
            "max_rank_retained": self.max_rank_retained(),
            "storage_floats": self.storage(),
            "dense_floats": self.size ** 2,
            "compression": float(self.storage() / self.size ** 2),
            "tolerance": self.tolerance,
            "min_block": self.min_block,
            "n_probes": int(n_probes),
        }


def hodlr_approximation(matrix: np.ndarray,
                        min_block: int = DEFAULT_MIN_BLOCK,
                        tolerance: float = DEFAULT_TOLERANCE,
                        max_rank: Optional[int] = None) -> np.ndarray:
    """
    The HODLR approximation of a matrix itself, not of its inverse.

    This is ``Sigma~`` in the paper's notation, the object Assumption 1
    bounds. Built densely, because its only use is measuring
    ``||Sigma - Sigma~||_2`` — the quantity the propositions are stated
    in — and that measurement is a diagnostic, not part of a scan.

    Args:
        matrix: Symmetric matrix to approximate.
        min_block: Terminal block size.
        tolerance: Spectral tolerance for the truncation.
        max_rank: Optional ceiling on the retained rank.

    Returns:
        The approximation, same shape as the input.
    """
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    out = matrix.copy()
    if n <= min_block or n // 2 < 2:
        return out

    split = n // 2
    U, V = low_rank(matrix[:split, split:], tolerance, max_rank)
    block = U @ V
    out[:split, split:] = block
    out[split:, :split] = block.T
    out[:split, :split] = hodlr_approximation(
        matrix[:split, :split], min_block, tolerance, max_rank)
    out[split:, split:] = hodlr_approximation(
        matrix[split:, split:], min_block, tolerance, max_rank)
    return out


__all__ = ["DEFAULT_MIN_BLOCK", "DEFAULT_TOLERANCE", "HInverse",
           "hodlr_approximation", "low_rank", "randomized_low_rank"]
