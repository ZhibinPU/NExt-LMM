"""
Render the benchmark figures.

    python benchmarks/make_figures.py

Reads ``benchmarks/results.json`` and writes PDF and PNG to
``figures/``. Nothing is recomputed here: a figure should be
reproducible from a results file, so that regenerating it does not
mean re-running the benchmarks.

Three panels, one per claim the implementation makes. The conventions
they follow are the ones scientific figures usually get wrong:

* **One axis per measure.** Time and storage are not commensurable, so
  they get separate panels rather than two y-scales on one.
* **Reference lines are explicit.** A log-log plot of cost against n is
  unreadable without the slopes it is being compared to, so the O(n)
  and O(n^2) guides are drawn.
* **Identity never rests on colour alone.** Every series carries a
  distinct marker and a direct label.
"""

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results.json"
FIGURES = ROOT / "figures"

# Page geometry, in inches: a 180 mm double column at 7 pt text.
MM = 1 / 25.4
DOUBLE = 183 * MM

# Validated categorical palette; the first three slots clear the
# all-pairs colour-vision-deficiency separation floors in both modes.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8982"
RULE = "#cfcec7"


def style():
    """The rcParams every figure here assumes."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.5,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def tidy(ax, grid=None):
    """Strip the chart junk a default axis carries."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=RULE, linewidth=0.4, alpha=0.8)
        ax.set_axisbelow(True)


def panel_label(ax, letter, dx=-0.17, dy=1.10):
    """A bold lowercase panel letter, in the usual position."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left", color=INK)


def fitted_slope(x, y):
    """Exponent of a power law, from a log-log fit."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def figure_scaling(rows, path):
    """
    Cost against problem size: the asymptotic claim, measured.

    The format promises ``O(k n log n)`` per solve against the dense
    ``O(n^2)``. On a log-log axis that is a difference in slope, which
    is why the guides matter more than the absolute positions — a
    machine twice as fast moves both curves down and changes neither
    slope.
    """
    style()
    n = np.array([r["n"] for r in rows], dtype=float)
    hodlr = np.array([r["hodlr_ms"] for r in rows])
    dense = np.array([r["dense_ms"] for r in rows])
    storage = np.array([r["storage_ratio"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.0))

    # --- a: solve cost -------------------------------------------------
    ax = axes[0]
    ax.plot(n, dense, "o-", color=ORANGE, markersize=3.6,
            markeredgewidth=0)
    ax.plot(n, hodlr, "s-", color=BLUE, markersize=3.4,
            markeredgewidth=0)

    # Slope guides, anchored at the first point so they are comparable.
    for exponent, label in ((2.0, "$n^2$"), (1.0, "$n$")):
        guide = dense[0] * (n / n[0]) ** exponent
        ax.plot(n, guide, color=INK_3, linewidth=0.6,
                linestyle=(0, (3, 2)), zorder=0)
        ax.text(n[-1] * 1.04, guide[-1], label, fontsize=6.5,
                color=INK_3, va="center", ha="left")

    # Direct labels, placed mid-curve where there is clear space: at
    # the left end they collide with the axis, at the right with the
    # slope guides.
    ax.text(n[2] * 0.92, dense[2] * 2.0, "dense", fontsize=7,
            color=ORANGE, fontweight="bold", ha="center")
    ax.text(n[2] * 0.92, hodlr[2] * 0.40, "HODLR", fontsize=7,
            color=BLUE, fontweight="bold", ha="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("Solve cost per column (ms)")
    ax.xaxis.set_major_locator(FixedLocator(n))
    ax.set_xticklabels([f"{int(v)}" for v in n])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(n[0] * 0.8, n[-1] * 1.9)
    tidy(ax)
    panel_label(ax, "a")
    ax.set_title(f"fitted slopes: {fitted_slope(n, hodlr):.2f} vs "
                 f"{fitted_slope(n, dense):.2f}", fontsize=6.5,
                 color=INK_2, pad=4)

    # --- b: speedup ----------------------------------------------------
    ax = axes[1]
    ratio = dense / hodlr
    ax.plot(n, ratio, "s-", color=BLUE, markersize=3.4,
            markeredgewidth=0)
    ax.axhline(1.0, color=INK_3, linewidth=0.7, linestyle=(0, (3, 2)))
    # Parked at the right, where the curve has already climbed away
    # from the line; at the left it sits on the first data point.
    ax.text(n[-1] * 1.02, 1.0, "parity", fontsize=6.5, color=INK_3,
            ha="left", va="center")
    for x, y in zip(n, ratio):
        ax.annotate(f"{y:.1f}x", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6,
                    color=BLUE, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("Speedup over dense")
    ax.xaxis.set_major_locator(FixedLocator(n))
    ax.set_xticklabels([f"{int(v)}" for v in n])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(n[0] * 0.8, n[-1] * 1.75)
    ax.set_ylim(0, max(ratio) * 1.28)
    tidy(ax, grid="y")
    panel_label(ax, "b")

    # --- c: storage ----------------------------------------------------
    ax = axes[2]
    ax.plot(n, storage * 100, "^-", color=AQUA, markersize=3.6,
            markeredgewidth=0)
    for x, value in zip(n, storage * 100):
        ax.annotate(f"{value:.1f}%", (x, value),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=6, color=AQUA,
                    fontweight="bold")
    ax.set_xscale("log")
    # Linear on y: a log axis here ends up labelling one decade, which
    # tells the reader nothing about the magnitudes involved.
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("Storage, % of dense $n^2$")
    ax.set_ylim(0, (storage * 100).max() * 1.3)
    ax.xaxis.set_major_locator(FixedLocator(n))
    ax.set_xticklabels([f"{int(v)}" for v in n])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(n[0] * 0.8, n[-1] * 1.25)
    tidy(ax)
    panel_label(ax, "c")
    ax.set_title("what the format buys first", fontsize=6.5,
                 color=INK_2, pad=4)

    fig.subplots_adjust(wspace=0.42)
    return save(fig, path)


def figure_rank_dial(rows, path):
    """
    The trade the method actually offers.

    A compressed covariance is deliberately inexact, so the panel that
    matters is not the matrix error but what that error does to the
    inference. The two are plotted side by side because the surprise
    is how little they move together: the spectral residual at rank 2
    is three times the residual at rank 64, and the type-I error is
    the same.
    """
    style()
    rank = np.array([r["rank"] for r in rows], dtype=float)
    residual = np.array([r["residual"] for r in rows])
    storage = np.array([r["storage_ratio"] for r in rows])
    type_one = np.array([r["type_one"] for r in rows])
    power = np.array([r["power"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.0))

    # --- a: cost of compression ---------------------------------------
    ax = axes[0]
    ax.plot(rank, residual, "o-", color=ORANGE, markersize=3.6,
            markeredgewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Retained off-diagonal rank $k$")
    ax.set_ylabel(r"Relative residual of $\Sigma^{-1}$")
    ax.xaxis.set_major_locator(FixedLocator(rank))
    ax.set_xticklabels([f"{int(v)}" for v in rank])
    ax.xaxis.set_minor_formatter(NullFormatter())
    tidy(ax, grid="y")
    panel_label(ax, "a")
    ax.set_title("the matrix gets worse", fontsize=6.5, color=INK_2,
                 pad=4)

    # --- b: and the inference does not --------------------------------
    # Both series are probabilities, so one axis is honest here.
    ax = axes[1]
    ax.plot(rank, type_one, "s-", color=BLUE, markersize=3.4,
            markeredgewidth=0)
    ax.plot(rank, power, "^-", color=AQUA, markersize=3.6,
            markeredgewidth=0)
    ax.axhline(0.05, color=INK_3, linewidth=0.7, linestyle=(0, (3, 2)))
    # Inside the axes rather than past the right spine: outside, the
    # two-line label runs into the panel border.
    ax.text(rank[0] * 0.95, 0.125, "nominal 0.05", fontsize=6,
            color=INK_3, va="bottom", ha="left")
    ax.text(rank[1], power[1] + 0.055, "power", fontsize=7, color=AQUA,
            fontweight="bold", ha="left")
    ax.text(rank[1], type_one[1] + 0.055, "type-I error", fontsize=7,
            color=BLUE, fontweight="bold", ha="left")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Retained off-diagonal rank $k$")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, max(power.max(), 0.4) * 1.45)
    ax.xaxis.set_major_locator(FixedLocator(rank))
    ax.set_xticklabels([f"{int(v)}" for v in rank])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(rank[0] * 0.85, rank[-1] * 2.1)
    tidy(ax, grid="y")
    panel_label(ax, "b")
    ax.set_title("the inference does not", fontsize=6.5, color=INK_2,
                 pad=4)

    # --- c: what it costs to store ------------------------------------
    ax = axes[2]
    ax.plot(rank, storage * 100, "D-", color=AQUA, markersize=3.2,
            markeredgewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Retained off-diagonal rank $k$")
    ax.set_ylabel("Storage, % of dense $n^2$")
    ax.xaxis.set_major_locator(FixedLocator(rank))
    ax.set_xticklabels([f"{int(v)}" for v in rank])
    ax.xaxis.set_minor_formatter(NullFormatter())
    tidy(ax, grid="y")
    panel_label(ax, "c")
    ax.set_title("and this is the price", fontsize=6.5, color=INK_2,
                 pad=4)

    fig.subplots_adjust(wspace=0.42)
    return save(fig, path)


def figure_scan(rows, path):
    """End-to-end: the number a user actually waits for."""
    style()
    n = np.array([r["n"] for r in rows], dtype=float)
    fast = np.array([r["nextlmm_s"] for r in rows])
    slow = np.array([r["dense_s"] for r in rows])
    markers = rows[0]["n_markers"]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.68, 2.0))

    ax = axes[0]
    positions = np.arange(len(n))
    width = 0.36
    ax.bar(positions - width / 2, slow, width, color=ORANGE,
           linewidth=0, label="per-marker dense")
    ax.bar(positions + width / 2, fast, width, color=BLUE,
           linewidth=0, label="NExt-LMM")
    for x, value in zip(positions - width / 2, slow):
        ax.text(x, value * 1.12, f"{value:.1f}", ha="center",
                fontsize=6, color=ORANGE, fontweight="bold")
    for x, value in zip(positions + width / 2, fast):
        ax.text(x, value * 1.12, f"{value:.2f}", ha="center",
                fontsize=6, color=BLUE, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(v)}" for v in n])
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel(f"Time to scan {markers} markers (s)")
    ax.legend(loc="upper left", handlelength=1.1, borderpad=0.1)
    tidy(ax, grid="y")
    panel_label(ax, "a", dx=-0.22)

    ax = axes[1]
    speedup = slow / fast
    ax.plot(n, speedup, "s-", color=BLUE, markersize=3.6,
            markeredgewidth=0)
    for x, y in zip(n, speedup):
        ax.annotate(f"{y:.0f}x", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=6.5,
                    color=BLUE, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("Speedup")
    ax.xaxis.set_major_locator(FixedLocator(n))
    ax.set_xticklabels([f"{int(v)}" for v in n])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(n[0] * 0.85, n[-1] * 1.2)
    ax.set_ylim(0, speedup.max() * 1.3)
    tidy(ax, grid="y")
    panel_label(ax, "b", dx=-0.22)

    fig.subplots_adjust(wspace=0.46)
    return save(fig, path)


def save(fig, path, formats=("pdf", "png")):
    """Write a figure in every requested format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in formats:
        target = path.with_suffix(f".{suffix}")
        fig.savefig(target, format=suffix)
        written.append(str(target))
    plt.close(fig)
    return written


def main():
    if not RESULTS.exists():
        sys.exit(f"No results at {RESULTS}.\n"
                 f"Run: python benchmarks/run_benchmarks.py")
    payload = json.loads(RESULTS.read_text())

    written = []
    written += figure_scaling(payload["scaling"], FIGURES / "fig1_scaling")
    written += figure_rank_dial(payload["rank_dial"],
                                FIGURES / "fig2_rank_dial")
    written += figure_scan(payload["end_to_end"], FIGURES / "fig3_scan")

    print("Written:")
    for item in written:
        print(f"  {item}")


if __name__ == "__main__":
    main()
