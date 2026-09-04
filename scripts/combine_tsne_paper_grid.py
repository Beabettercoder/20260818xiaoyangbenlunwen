#!/usr/bin/env python3
"""Build a paper-style 2xN t-SNE grid with equal square panels."""

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_tsne_paper_nwpu_source import (
    COLOR_PALETTE,
    compute_axis_limits,
    configure_matplotlib_times,
    draw_paper_panel,
    make_paper_legend_handles,
    target_title_name,
)

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for the combined square t-SNE grid") from exc


DEFAULT_TARGET_ORDER = ("EuroSAT", "AID", "UCM")


def parse_targets(targets_arg):
    targets = [item.strip() for item in targets_arg.split(",") if item.strip()]
    return targets or list(DEFAULT_TARGET_ORDER)


def find_coord_files(input_dir, source, tag, seed, target_order):
    input_dir = Path(input_dir)
    items = []
    for target in target_order:
        path = input_dir / f"tsne_paper_{source}_to_{target}_{tag}_seed{seed}_coords.npz"
        if path.is_file():
            items.append((target, path))

    known = {target for target, _path in items}
    for path in sorted(input_dir.glob(f"tsne_paper_{source}_to_*_{tag}_seed{seed}_coords.npz")):
        name = path.name
        prefix = f"tsne_paper_{source}_to_"
        suffix = f"_{tag}_seed{seed}_coords.npz"
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        target = name[len(prefix):-len(suffix)]
        if target not in known:
            items.append((target, path))
    return items


def load_item(target, path):
    data = np.load(path, allow_pickle=True)
    return {
        "target": target,
        "class_names": [str(item) for item in data["class_names"]],
        "labels": data["labels"],
        "emb_base": data["emb_base"],
        "emb_sga": data["emb_sga"],
    }


def draw_compact_legend(ax, class_names):
    ax.set_axis_off()
    handles = make_paper_legend_handles(class_names)
    ncol = 2 if len(class_names) > 6 else 1
    ax.legend(
        handles=handles,
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize=6.7,
        columnspacing=0.75,
        handlelength=0.8,
        handletextpad=0.35,
        borderaxespad=0.0,
        markerscale=0.85,
    )


def plot_square_grid(source, items, output_stem, dpi, point_size, axis_mode):
    if not items:
        raise RuntimeError("No per-target t-SNE coordinate files were found.")

    configure_matplotlib_times()
    loaded = [load_item(target, path) for target, path in items]
    ncols = len(loaded)

    fig_w = 3.35 * ncols + 1.2
    fig_h = 7.35
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#FFFFFF")
    grid = fig.add_gridspec(
        3,
        ncols,
        height_ratios=[1.0, 1.0, 0.18],
        left=0.145,
        right=0.985,
        top=0.91,
        bottom=0.07,
        wspace=0.13,
        hspace=0.08,
    )

    axes = []
    for row in range(2):
        row_axes = []
        for col, item in enumerate(loaded):
            ax = fig.add_subplot(grid[row, col])
            row_axes.append(ax)
            if axis_mode == "shared":
                limits = compute_axis_limits(np.vstack([item["emb_base"], item["emb_sga"]]))
            else:
                emb = item["emb_base"] if row == 0 else item["emb_sga"]
                limits = compute_axis_limits(emb)
            emb = item["emb_base"] if row == 0 else item["emb_sga"]
            draw_paper_panel(
                ax,
                emb,
                item["labels"],
                item["class_names"],
                limits[0],
                limits[1],
                point_size,
            )
            if row == 0:
                ax.set_title(
                    f"{source}\u2192{target_title_name(item['target'])}",
                    fontsize=16,
                    fontweight="bold",
                    pad=12,
                )
        axes.append(row_axes)

    for col, item in enumerate(loaded):
        draw_compact_legend(fig.add_subplot(grid[2, col]), item["class_names"])

    for row, label in enumerate(("Baseline", "SGA-Net\n(ours)")):
        first = axes[row][0].get_position()
        fig.text(
            0.105,
            (first.y0 + first.y1) / 2,
            label,
            ha="right",
            va="center",
            fontsize=16,
            fontweight="bold",
            linespacing=0.88,
        )

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi)
    try:
        fig.savefig(pdf_path)
        saved_pdf_path = pdf_path
    except PermissionError:
        saved_pdf_path = output_stem.with_name(output_stem.name + "_new").with_suffix(".pdf")
        fig.savefig(saved_pdf_path)
        print(f"[warn] PDF is locked, saved alternate: {saved_pdf_path}")
    plt.close(fig)
    print(f"[saved] {png_path}")
    print(f"[saved] {saved_pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Build an equal-square-panel t-SNE grid.")
    parser.add_argument("--input_dir", type=str, default="output/tsne_paper_nwpu_source_20260501_v1/main")
    parser.add_argument("--output_dir", type=str, default="output/tsne_paper_nwpu_source_20260501_v1/main")
    parser.add_argument("--source_dataset", type=str, default="NWPU")
    parser.add_argument("--figure_tag", type=str, default="main")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--point_size", type=float, default=7.0)
    parser.add_argument("--axis_mode", type=str, default="panel", choices=["panel", "shared"])
    parser.add_argument("--targets", type=str, default="EuroSAT,AID,UCM",
                        help="Comma-separated target order for columns.")
    parser.add_argument("--require_targets", action="store_true",
                        help="Fail if any target listed by --targets has no coordinate file.")
    parser.add_argument("--gap", type=int, default=80, help="Accepted for compatibility; not used in grid mode.")
    args = parser.parse_args()

    target_order = parse_targets(args.targets)
    items = find_coord_files(args.input_dir, args.source_dataset, args.figure_tag, args.seed, target_order)
    if args.require_targets:
        found = {target for target, _path in items}
        missing = [target for target in target_order if target not in found]
        if missing:
            raise RuntimeError(
                "Missing required t-SNE coordinate files for "
                f"{args.source_dataset}: {', '.join(missing)}"
            )
    output_stem = Path(args.output_dir) / f"tsne_paper_{args.source_dataset}_source_all_targets_seed{args.seed}"
    plot_square_grid(args.source_dataset, items, output_stem, args.dpi, args.point_size, args.axis_mode)


if __name__ == "__main__":
    main()
