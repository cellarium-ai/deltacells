# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Make the figures of the "How the encoding works" page from a real h5ad shard (integer counts, up to about 10,000 cells).

    python docs/make_figures.py data/extract_0.h5ad                  # writes docs/source/_static/*.png
    python docs/make_figures.py shard.h5ad --out some/other/folder

The figures are committed, so building the documentation needs no data. Run this script again only to change them. It prints
the numbers quoted on that page. The shard used is stored in increasing order of expression, which compresses almost as well as the
sorted order deltacells writes, so a random gene order is the contrast shown.
"""

from __future__ import annotations

import argparse
import os

import anndata
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from deltacells import encode_tile, permute_columns  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9992"  # sorted (deltacells), original order, random order
MAX_CELLS = 10_000  # one tile


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, out: str, name: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    path = os.path.join(out, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def gene_gaps(m: sp.csr_matrix) -> np.ndarray:
    """The delta-coded gene indices of every cell, as stored: ``g0, g1 - g0, g2 - g1, ...``."""
    idx = m.indices.astype(np.int64)
    gaps = np.empty_like(idx)
    gaps[0] = idx[0]
    gaps[1:] = idx[1:] - idx[:-1]
    starts = m.indptr[:-1][np.diff(m.indptr) > 0]
    gaps[starts] = idx[starts]
    return gaps


def pattern_figure(
    shuffled: sp.csr_matrix, ordered: sp.csr_matrix, out: str, n_cells: int = 80, n_bins: int = 400
) -> None:
    n_genes = shuffled.shape[1]
    edges = np.linspace(0, n_genes, n_bins + 1).astype(int)
    cmap = LinearSegmentedColormap.from_list("blue", [SURFACE, "#9ec3f0", BLUE, "#123f78"])
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    for ax, m, title in (
        (axes[0], shuffled, "Genes in a random order"),
        (axes[1], ordered, "Genes sorted by decreasing total counts"),
    ):
        dense = m[:n_cells].toarray() > 0
        binned = np.add.reduceat(dense, edges[:-1], axis=1) / np.diff(edges)
        ax.imshow(
            binned, aspect="auto", cmap=cmap, vmin=0, vmax=0.5, interpolation="nearest", extent=(0, n_genes, n_cells, 0)
        )
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_ylabel(f"{n_cells} cells", fontsize=9)
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[1].set_xlabel("gene column index", fontsize=9, color=INK)
    fig.tight_layout(h_pad=1.2)
    save(fig, out, "pattern.png")


def gap_figure(gaps: dict[str, np.ndarray], out: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for label, color, g in (
        ("random gene order", ORANGE, gaps["random"]),
        ("sorted by total counts", BLUE, gaps["sorted"]),
    ):
        x = np.sort(g)
        y = np.arange(1, len(x) + 1) / len(x)
        keep = np.unique(np.searchsorted(x, np.unique(x), side="right") - 1)
        ax.plot(x[keep], y[keep], color=color, linewidth=2, label=f"{label} (median gap {int(np.median(g))})")
    ax.set_xscale("log")
    ax.set_xlim(1, 4e4)
    ax.set_ylim(0, 1.0)
    ax.axvline(255, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(285, 0.5, "255: a gap this small\nfits in one byte", color=MUTED, fontsize=8.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("gap between a cell's consecutive detected genes (columns)")
    ax.set_ylabel("fraction of gaps at most this large")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    save(fig, out, "gap-cdf.png")


def stream_figure(gaps: np.ndarray, counts: np.ndarray, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    cap = 10
    for ax, values, title, xlabel in (
        (axes[0], gaps, "Gene gaps (sorted order)", "gap, in columns"),
        (axes[1], counts, "Counts", "count"),
    ):
        style(ax)
        clipped = np.minimum(values, cap)
        frac = np.bincount(clipped, minlength=cap + 1)[1:] / len(values)
        bars = ax.bar(np.arange(1, cap + 1), frac, color=BLUE, width=0.72)
        for i in (0, 1, 2):
            ax.text(i + 1, frac[i] + 0.012, f"{100 * frac[i]:.0f}%", ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(np.arange(1, cap + 1), [str(i) for i in range(1, cap)] + [f"{cap}+"])
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlabel(xlabel)
        del bars
    axes[0].set_ylabel("fraction of values")
    fig.tight_layout(w_pad=2)
    save(fig, out, "streams.png")


def size_figure(sizes: dict[str, float], out: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    names = ["random gene order", "sorted by decreasing\ntotal counts"]
    keys = ["random", "sorted"]
    colors = [GRAY, BLUE]
    y = np.arange(len(keys))[::-1]
    ax.barh(y, [sizes[k] for k in keys], color=colors, height=0.6)
    for yi, k in zip(y, keys):
        ax.text(sizes[k] + 0.01, yi, f"{sizes[k]:.2f} bytes", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", length=0, labelsize=10, labelcolor=INK)
    ax.set_xlim(0, max(sizes.values()) * 1.25)
    ax.set_xlabel("compressed size per nonzero value (zstd level 19)")
    save(fig, out, "size-by-order.png")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5ad")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "source", "_static"))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    original = sp.csr_matrix(anndata.read_h5ad(args.h5ad).X)[:MAX_CELLS]
    original.sort_indices()
    n_cells, n_genes = original.shape
    totals = np.asarray(original.sum(axis=0)).ravel()
    sort_order = np.argsort(-totals, kind="stable")  # what `deltacells convert` does
    random_order = np.random.default_rng(0).permutation(n_genes)
    ordered = permute_columns(original, sort_order)
    shuffled = permute_columns(original, random_order)
    print(f"{n_cells} cells x {n_genes} genes, {original.nnz / n_cells:.0f} nonzeros per cell")

    gaps = {"original": gene_gaps(original), "random": gene_gaps(shuffled), "sorted": gene_gaps(ordered)}
    counts = ordered.data.astype(np.int64)
    for k, g in gaps.items():
        print(
            f"gaps, {k}: median {np.median(g):.0f}, mean {g.mean():.1f}, <=255: {100 * (g <= 255).mean():.1f}%, ==1: {100 * (g == 1).mean():.1f}%"
        )
    print(
        f"counts: ==1 {100 * (counts == 1).mean():.1f}%, ==2 {100 * (counts == 2).mean():.1f}%, <=3 {100 * (counts <= 3).mean():.1f}%, >65535: {(counts > 65535).sum()}"
    )
    depth = np.diff(ordered.indptr)
    cell = int(np.argsort(np.abs(depth - np.median(depth)))[0])  # a cell of typical depth
    row = slice(int(ordered.indptr[cell]), int(ordered.indptr[cell + 1]))
    start = depth[cell] // 3
    print(
        f"cell {cell} ({depth[cell]} nonzeros), 24 consecutive detected genes of its sorted row (from the {start}th):"
    )
    print("  gaps  ", gaps["sorted"][row][start : start + 24].tolist())
    print("  counts", ordered.data[row][start : start + 24].astype(int).tolist())

    sizes = {}
    for key, m in (("original", original), ("sorted", ordered), ("random", shuffled)):
        sizes[key] = len(encode_tile(m, level=19, n_chunks=8)) / m.nnz
        print(f"size, {key}: {sizes[key]:.3f} bytes per nonzero")

    pattern_figure(shuffled, ordered, args.out)
    gap_figure(gaps, args.out)
    stream_figure(gaps["sorted"], counts, args.out)
    size_figure(sizes, args.out)


if __name__ == "__main__":
    main()
