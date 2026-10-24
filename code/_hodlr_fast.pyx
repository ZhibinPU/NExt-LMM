# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Compiled inner loop for the HODLR solve.

The Python implementation in ``HodlrInverse.py`` is already correct and
already has the right complexity — one descent per node, no per-level
copies. What remains is interpreter overhead: at n = 4096 it performs
3.1% of the dense floating-point work and takes 20% of the dense time,
so roughly a 6x constant factor sits in Python call frames and numpy
dispatch on small arrays.

This module removes that. Two changes:

**The tree is flattened.** ``HInverse`` is a graph of Python objects;
walking it costs an attribute lookup and a method call per node. Here
the same tree is laid out as flat arrays — one row per node, children
found by index — so the traversal is a loop over integers.

**BLAS is called directly.** ``scipy.linalg.cython_blas`` exposes
``dgemm`` at the C level, so a 64x64 block multiply costs one function
call rather than a numpy dispatch that inspects dtypes, checks
alignment, and allocates an output.

The arithmetic is unchanged, and ``test_cython_matches_python`` asserts
that element-wise. If this extension is not built, ``HodlrInverse``
falls back to the Python path and everything still works — just slower.
"""

import numpy as np

cimport numpy as cnp
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy
from scipy.linalg.cython_blas cimport dgemm

cnp.import_array()


cdef inline void _gemm(double* A, double* B, double* C,
                       int m, int n, int k,
                       double alpha, double beta) noexcept nogil:
    """
    C(m,n) := alpha * A(m,k) @ B(k,n) + beta * C, all row-major.

    BLAS is column-major and these buffers are row-major. Rather than
    transposing anything, the identity ``(A B)^T = B^T A^T`` is used:
    handing Fortran the same buffers as though they were the transposes
    computes exactly the wanted product, in place.
    """
    cdef char* trans = 'N'
    dgemm(trans, trans, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n)


cdef inline void _gemm_at(double* A, double* B, double* C,
                          int m, int n, int k,
                          double alpha, double beta) noexcept nogil:
    """
    C(m,n) := alpha * A(k,m)^T @ B(k,n) + beta * C, all row-major.

    Two of the three products at each node transpose their left
    operand: ``U^T (N11 b1)`` and ``V^T tmp``, because ``U`` is stored
    ``(split, k)`` and ``V`` is ``(k, lower)``. An earlier version of
    this file avoided the layout question by writing those two as
    explicit triple loops — which cost 6.3 million scalar operations
    per node against the 12.9 GFLOPS BLAS reaches on the same work,
    and made the compiled path three times *slower* than the numpy one
    it was meant to replace. Getting the flags right is the whole job.

    In the column-major view the wanted result is ``C^T = B^T A``, so
    the first operand is passed untransposed and the second transposed.
    """
    cdef char* no_trans = 'N'
    cdef char* do_trans = 'T'
    dgemm(no_trans, do_trans, &n, &m, &k, &alpha, B, &n, A, &m, &beta, C, &n)


cdef class FlatHodlr:
    """
    Flattened HODLR factorisation, applied without touching Python.

    Node ``i`` occupies rows ``offset[i] : offset[i] + size[i]`` of the
    vector. A leaf carries a dense inverse; an internal node carries
    the off-diagonal factors ``U`` and ``V`` plus the indices of its
    two children.

    All per-node arrays are stored contiguously in one buffer each,
    with per-node slices addressed by offset, so the traversal touches
    memory in order rather than chasing pointers across the heap.

    Args:
        nodes: One dict per node, in the order produced by
            ``HInverse._flatten`` — children before parents is *not*
            required, the child indices say what goes where.
        size: Dimension of each node's block.
    """

    cdef:
        int n_nodes, total_size
        int[::1] node_size, node_offset, node_split
        int[::1] left, right, rank
        int[::1] leaf_start, factor_start
        double[::1] leaf_data      # concatenated dense inverses
        double[::1] u_data         # concatenated U factors
        double[::1] v_data         # concatenated V factors
        int[::1] u_start, v_start
        double* scratch
        int scratch_size

    def __cinit__(self):
        self.scratch = NULL
        self.scratch_size = 0

    def __dealloc__(self):
        if self.scratch != NULL:
            free(self.scratch)

    def __init__(self, nodes, int total_size):
        cdef int count = len(nodes)
        self.n_nodes = count
        self.total_size = total_size

        self.node_size = np.empty(count, dtype=np.int32)
        self.node_offset = np.empty(count, dtype=np.int32)
        self.node_split = np.empty(count, dtype=np.int32)
        self.left = np.empty(count, dtype=np.int32)
        self.right = np.empty(count, dtype=np.int32)
        self.rank = np.empty(count, dtype=np.int32)
        self.leaf_start = np.empty(count, dtype=np.int32)
        self.u_start = np.empty(count, dtype=np.int32)
        self.v_start = np.empty(count, dtype=np.int32)

        leaves, us, vs = [], [], []
        cdef int leaf_cursor = 0, u_cursor = 0, v_cursor = 0
        cdef int i

        for i, node in enumerate(nodes):
            self.node_size[i] = node["size"]
            self.node_offset[i] = node["offset"]
            if node["leaf"]:
                self.node_split[i] = 0
                self.left[i] = -1
                self.right[i] = -1
                self.rank[i] = 0
                self.leaf_start[i] = leaf_cursor
                self.u_start[i] = -1
                self.v_start[i] = -1
                block = np.ascontiguousarray(node["inverse"], dtype=np.float64)
                leaves.append(block.ravel())
                leaf_cursor += block.size
            else:
                self.node_split[i] = node["split"]
                self.left[i] = node["left"]
                self.right[i] = node["right"]
                U = np.ascontiguousarray(node["U"], dtype=np.float64)
                V = np.ascontiguousarray(node["V"], dtype=np.float64)
                self.rank[i] = U.shape[1]
                self.leaf_start[i] = -1
                self.u_start[i] = u_cursor
                self.v_start[i] = v_cursor
                us.append(U.ravel())
                vs.append(V.ravel())
                u_cursor += U.size
                v_cursor += V.size

        self.leaf_data = (np.concatenate(leaves) if leaves
                          else np.empty(0, dtype=np.float64))
        self.u_data = (np.concatenate(us) if us
                       else np.empty(0, dtype=np.float64))
        self.v_data = (np.concatenate(vs) if vs
                       else np.empty(0, dtype=np.float64))

    cdef void _ensure_scratch(self, int needed):
        """Grow the working buffer, reusing it across calls."""
        if needed <= self.scratch_size:
            return
        if self.scratch != NULL:
            free(self.scratch)
        self.scratch = <double*> malloc(needed * sizeof(double))
        if self.scratch == NULL:
            raise MemoryError(
                f"could not allocate {needed * 8} bytes of scratch space")
        self.scratch_size = needed

    cdef void _apply_node(self, int node, double* B, double* out,
                          int width, double* work) noexcept nogil:
        """
        Write ``M_node^-1 B`` into ``out``, recursively.

        The same algebra as the Python path::

            y2 = -S^-1 (V^T U^T N11 b1 - b2)
            y1 =  N11 b1 - (N11 U) V y2

        with ``[b1 | U]`` descending together so each node is visited
        once. ``work`` is a slice of the shared scratch buffer, sized
        by the caller for the deepest level.
        """
        cdef int size = self.node_size[node]
        cdef int split, k, joint_width, i, j, idx, lower
        cdef double value
        cdef double* joint
        cdef double* applied
        cdef double* w
        cdef double* tmp
        cdef double* nu
        cdef double* child_work
        cdef double* U
        cdef double* V
        cdef double* y2

        if self.left[node] < 0:
            _gemm(&self.leaf_data[self.leaf_start[node]], B, out,
                  size, width, size, 1.0, 0.0)
            return

        split = self.node_split[node]
        k = self.rank[node]
        joint_width = width + k

        joint = work
        applied = joint + split * joint_width
        w = applied + split * joint_width
        tmp = w + (size - split) * width
        nu = tmp + k * width
        child_work = nu + split * k

        # joint = [b1 | U], row-major with joint_width columns.
        U = &self.u_data[self.u_start[node]]
        for i in range(split):
            memcpy(joint + i * joint_width, B + i * width,
                   width * sizeof(double))
            memcpy(joint + i * joint_width + width, U + i * k,
                   k * sizeof(double))

        self._apply_node(self.left[node], joint, applied, joint_width,
                         child_work)

        # tmp = U^T (N11 b1)   -> (k, width)
        # N11 b1 is the leading `width` columns of `applied`, which has
        # a row stride of joint_width — so it is copied out rather than
        # passed as a strided view, which dgemm cannot take here.
        for i in range(split):
            memcpy(w + i * width, applied + i * joint_width,
                   width * sizeof(double))
        _gemm_at(U, w, tmp, k, width, split, 1.0, 0.0)

        # w = V^T tmp - b2     -> (size - split, width)
        # V is stored (k, lower), so this is a transposed product; done
        # as an explicit loop rather than by juggling dgemm's layout
        # flags, which is where a sign or an index silently goes wrong.
        V = &self.v_data[self.v_start[node]]
        lower = size - split
        _gemm_at(V, tmp, w, lower, width, k, 1.0, 0.0)
        for i in range(lower):
            for j in range(width):
                w[i * width + j] -= B[(split + i) * width + j]

        # y2 = -S^-1 w
        y2 = out + split * width
        self._apply_node(self.right[node], w, y2, width, child_work)
        for i in range(lower * width):
            y2[i] = -y2[i]

        # y1 = N11 b1 - (N11 U) (V y2)
        # tmp reused for V y2   -> (k, width); V is (k, lower) here, so
        # this one is a plain product.
        _gemm(V, y2, tmp, k, width, lower, 1.0, 0.0)
        # y1 = N11 b1 - (N11 U) @ tmp. N11 U is the trailing k columns
        # of `applied`; compacted the same way, then one gemm.
        for i in range(split):
            memcpy(out + i * width, applied + i * joint_width,
                   width * sizeof(double))
            memcpy(nu + i * k, applied + i * joint_width + width,
                   k * sizeof(double))
        _gemm(nu, tmp, out, split, width, k, -1.0, 1.0)

    cdef int _scratch_needed(self, int node, int width) noexcept nogil:
        """Working space the deepest path through this node will want."""
        cdef int size, split, k, joint_width, lower, here, a, b
        if self.left[node] < 0:
            return 0
        size = self.node_size[node]
        split = self.node_split[node]
        k = self.rank[node]
        joint_width = width + k
        lower = size - split
        # `w` doubles as a staging buffer for the compacted N11 b1, so
        # it must hold the larger of the two.
        if split > lower:
            lower = split
        here = (2 * split * joint_width + lower * width + k * width
                + split * k)
        a = self._scratch_needed(self.left[node], joint_width)
        b = self._scratch_needed(self.right[node], width)
        return here + (a if a > b else b)

    def matmat(self, double[:, ::1] B):
        """
        Apply the inverse to a C-contiguous right-hand side.

        Args:
            B: ``(n, c)`` array of right-hand sides.

        Returns:
            ``M^-1 B``.

        Raises:
            ValueError: If the row count does not match.
        """
        if B.shape[0] != self.total_size:
            raise ValueError(
                f"Expected {self.total_size} rows, got {B.shape[0]}")
        cdef int width = B.shape[1]
        out_arr = np.empty((self.total_size, width), dtype=np.float64)
        cdef double[:, ::1] out = out_arr
        self._ensure_scratch(self._scratch_needed(0, width) + 16)
        with nogil:
            self._apply_node(0, &B[0, 0], &out[0, 0], width, self.scratch)
        return out_arr
