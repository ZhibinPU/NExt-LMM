"""
Measure the three claims this implementation makes, and write the
numbers to JSON so the figures can be redrawn without re-running an
hour of benchmarks.

    python benchmarks/run_benchmarks.py --out benchmarks/results.json

Three experiments, one per claim:

1. **Scaling.** Solve cost against n, for the HODLR path and a dense
   one. The format claims O(k n log n) against O(n^2); a log-log plot
   of the two should have visibly different slopes, and the fitted
   exponents are reported alongside.
2. **The rank dial.** Approximation error, storage and statistical
   behaviour against the retained rank. This is the trade the method
   actually offers: accuracy for cost, with inference holding up at
   settings where the matrix error looks alarming.
3. **End-to-end.** A genome scan, against the per-marker dense
   approach the released code used.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from HodlrInverse import HInverse          # noqa: E402
from NExt import NExt                      # noqa: E402


def gsm(n, m_factor=2, seed=0):
    """A genetic similarity matrix, built the way a real one is."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, m_factor * n))
    return Z @ Z.T / (m_factor * n)


def timed(fn, repeats):
    """Best-of-three mean, to keep scheduler noise out of the figure."""
    for _ in range(2):
        fn()
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        best = min(best, (time.perf_counter() - start) / repeats)
    return best * 1000.0


def scaling(sizes, rank, width, seed=0):
    """Solve cost against n, HODLR versus dense."""
    rows = []
    for n in sizes:
        Sigma = gsm(n, seed=seed) + np.eye(n)
        model = HInverse(Sigma, min_block=64, max_rank=rank,
                         randomized=True)
        dense = np.linalg.inv(Sigma)
        B = np.ascontiguousarray(
            np.random.default_rng(n).standard_normal((n, width)))
        repeats = max(2, 40 // max(1, n // 512))
        rows.append({
            "n": n,
            "hodlr_ms": timed(lambda: model.matmat(B), repeats) / width,
            "dense_ms": timed(lambda: dense @ B, repeats) / width,
            "storage_ratio": model.storage() / n ** 2,
            "residual": model.error_bound(Sigma)["relative_residual"],
        })
        print(f"  scaling n={n}: {rows[-1]['hodlr_ms']:.4f} vs "
              f"{rows[-1]['dense_ms']:.4f} ms/column", flush=True)
    return rows


def rank_dial(n, ranks, replicates, seed=0):
    """Error, storage and inference against the retained rank."""
    K = gsm(n, seed=seed)
    Sigma = K + np.eye(n)                      # h2 = 0.5 => lambda = 1
    chol = np.linalg.cholesky(Sigma)
    rng = np.random.default_rng(seed + 1)
    X = rng.binomial(2, 0.3, size=(n, 40)).astype(float)

    rows = []
    for rank in ranks:
        model = HInverse(Sigma, min_block=64, max_rank=rank)
        bound = model.error_bound(Sigma)

        null_p, hits, total = [], 0, 0
        for replicate in range(replicates):
            draw = np.random.default_rng(1000 + replicate)
            y = chol @ draw.standard_normal(n)
            fit = NExt(y, min_block=64, max_rank=rank).fit(X, K, h=0.5)
            null_p.append(fit.association())

            y_signal = y + X[:, :12].sum(axis=1) * 0.22
            powered = NExt(y_signal, min_block=64,
                           max_rank=rank).fit(X, K, h=0.5)
            hits += int((powered.association()[:12] < 0.05).sum())
            total += 12

        null_p = np.concatenate(null_p)
        rows.append({
            "rank": rank,
            "residual": bound["relative_residual"],
            "storage_ratio": bound["compression"],
            "type_one": float((null_p < 0.05).mean()),
            "ks_pvalue": float(stats.kstest(null_p, "uniform").pvalue),
            "power": hits / total,
        })
        print(f"  rank {rank}: residual {rows[-1]['residual']:.3f}, "
              f"type-I {rows[-1]['type_one']:.4f}, "
              f"power {rows[-1]['power']:.3f}", flush=True)
    return rows


def end_to_end(sizes, n_markers, rank, seed=0):
    """A scan, against the per-marker dense approach."""
    rows = []
    for n in sizes:
        K = gsm(n, seed=seed)
        rng = np.random.default_rng(seed + 2)
        y = rng.standard_normal(n)
        X = rng.binomial(2, 0.3, size=(n, n_markers)).astype(float)

        start = time.perf_counter()
        NExt(y, min_block=64, max_rank=rank,
             randomized=True).fit(X, K, h=0.5)
        fast = time.perf_counter() - start

        Sigma_inv = np.linalg.inv(K + np.eye(n))
        start = time.perf_counter()
        for j in range(n_markers):
            design = np.column_stack([X[:, j], np.ones(n)])
            gram = np.linalg.inv(design.T @ Sigma_inv @ design)
            theta = gram @ design.T @ Sigma_inv @ y
            residual = y - design @ theta
            float(residual @ Sigma_inv @ residual / (n - 2))
        slow = time.perf_counter() - start

        rows.append({"n": n, "n_markers": n_markers,
                     "nextlmm_s": fast, "dense_s": slow,
                     "speedup": slow / fast})
        print(f"  scan n={n}: {fast:.2f} s vs {slow:.2f} s "
              f"({slow / fast:.0f}x)", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/results.json")
    parser.add_argument("--quick", action="store_true",
                        help="smaller sizes and fewer replicates")
    args = parser.parse_args()

    if args.quick:
        sizes = [256, 512, 1024]
        scan_sizes = [256, 512]
        replicates, markers = 15, 500
    else:
        sizes = [256, 512, 1024, 2048, 4096]
        scan_sizes = [512, 1024, 2048]
        replicates, markers = 60, 2000

    print("1/3 scaling")
    scale_rows = scaling(sizes, rank=8, width=128)
    print("2/3 rank dial")
    dial_rows = rank_dial(512, [2, 4, 8, 16, 32, 64], replicates)
    print("3/3 end to end")
    scan_rows = end_to_end(scan_sizes, markers, rank=8)

    payload = {"scaling": scale_rows, "rank_dial": dial_rows,
               "end_to_end": scan_rows,
               "settings": {"min_block": 64, "width": 128,
                            "replicates": replicates,
                            "n_markers": markers}}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
