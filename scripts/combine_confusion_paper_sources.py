#!/usr/bin/env python3
"""Combine multiple source-domain confusion grids with one shared colorbar."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_confusion_paper_nwpu_source import (
    find_normalized_csv,
    read_confusion_csv,
    save_figure,
    style_matrix_axis,
)


DEFAULT_TARGETS = {
    "NWPU": ["EuroSAT", "AID", "UCM"],
    "AID": ["NWPU", "EuroSAT", "UCM"],
    "UCM": ["NWPU", "EuroSAT", "AID"],
}


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_targets_by_source(value):
    mapping = {}
    if not value:
        return mapping
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid source target mapping: {chunk}")
        source, targets = chunk.split("=", 1)
        mapping[source.strip()] = parse_csv(targets)
    return mapping


def load_items(sources, input_dirs, targets_by_source, shots, model_suffix, require_complete):
    items = []
    missing = []
    if len(input_dirs) == 1 and len(sources) > 1:
        input_dirs = input_dirs * len(sources)
    if len(input_dirs) != len(sources):
        raise ValueError("--input_dirs must contain one directory or one directory per source")

    for source, input_dir in zip(sources, input_dirs):
        targets = targets_by_source.get(source, DEFAULT_TARGETS.get(source, []))
        if not targets:
            raise ValueError(f"No targets configured for source={source}")
        for shot in shots:
            row = []
            for target in targets:
                csv_path = find_normalized_csv(input_dir, source, target, shot, model_suffix)
                if csv_path is None:
                    missing.append(f"{source}->{target} {shot}-shot")
                    continue
                matrix, class_names = read_confusion_csv(csv_path)
                row.append(
                    {
                        "matrix": matrix,
                        "class_names": class_names,
                        "title": f"{source}->{target} | {shot}-shot",
                        "caption": f"{source}->{target} {shot}-shot",
                    }
                )
            if row:
                items.append(row)

    if missing:
        print("[missing] These matrices were not found:")
        for item in missing:
            print(f"  - {item}")
        if require_complete:
            raise RuntimeError("Missing required confusion matrices.")
    return items


def panel_label(index):
    letters = "abcdefghijklmnopqrstuvwxyz"
    if index < len(letters):
        return f"({letters[index]})"
    return f"({index + 1})"


def plot_combined_grid(rows, output_path, dpi):
    if not rows:
        raise RuntimeError("No confusion matrices were loaded.")

    nrows = len(rows)
    ncols = max(len(row) for row in rows)
    fig_h = 4.25 * nrows + 0.45
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.8, fig_h), constrained_layout=False)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    fig.subplots_adjust(
        left=0.075,
        right=0.895,
        top=0.975,
        bottom=0.045,
        wspace=0.320,
        hspace=0.700,
    )

    image = None
    label_index = 0
    for row_idx, row_items in enumerate(rows):
        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(row_items):
                ax.set_axis_off()
                continue
            item = row_items[col_idx]
            image = style_matrix_axis(
                ax,
                item["matrix"],
                item["class_names"],
                item["title"],
                show_ylabel=(col_idx == 0),
                xlabel_pad=5,
            )
            ax.text(
                0.5,
                -0.345,
                f"{panel_label(label_index)} {item['caption']}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=12,
            )
            label_index += 1

    cbar_ax = fig.add_axes([0.915, 0.055, 0.018, 0.890])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=9, length=2)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label("Normalized Probability", fontsize=10, labelpad=8)

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    output_stem = Path(output_path).with_suffix("")
    print(f"[saved] {output_stem.with_suffix('.png')}")
    print(f"[saved] {output_stem.with_suffix('.pdf')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Combine source confusion grids with one shared colorbar.")
    parser.add_argument("--sources", type=parse_csv, required=True,
                        help="Comma-separated source datasets in row-block order.")
    parser.add_argument("--input_dirs", type=parse_csv, required=True,
                        help="One raw confusion directory, or one directory per source.")
    parser.add_argument("--targets_by_source", type=parse_targets_by_source, default={},
                        help="Semicolon-separated mapping, e.g. NWPU=EuroSAT,AID,UCM;AID=NWPU,EuroSAT,UCM")
    parser.add_argument("--shots", type=lambda value: [int(item.strip()) for item in value.split(",") if item.strip()],
                        default=[1, 5])
    parser.add_argument("--model_suffix", type=str, default="gnn")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--require_complete", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_items(
        args.sources,
        [Path(item) for item in args.input_dirs],
        args.targets_by_source,
        args.shots,
        args.model_suffix,
        args.require_complete,
    )
    plot_combined_grid(rows, args.output, args.dpi)


if __name__ == "__main__":
    main()
