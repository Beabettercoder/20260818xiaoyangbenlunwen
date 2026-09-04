#!/usr/bin/env python3
"""Redraw source-domain confusion matrices in a paper-friendly style.

This script does not recompute predictions and does not modify confusion
values. It only reads existing normalized CSV files and exports compact
Matplotlib figures suitable for thesis/paper insertion.
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


REPO_ROOT = Path(__file__).resolve().parents[1]


def configure_times_font():
    candidates = [
        os.environ.get("CONFUSION_PAPER_FONT", ""),
        os.environ.get("TSNE_PAPER_FONT", ""),
        str(REPO_ROOT / "fonts" / "times.ttf"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            if family != "Times New Roman":
                raise RuntimeError(f"The configured font is '{family}', not Times New Roman: {path}")
            break
    else:
        searched = "\n  - ".join(path for path in candidates if path)
        raise RuntimeError(
            "Times New Roman font was not found. Set CONFUSION_PAPER_FONT "
            f"or TSNE_PAPER_FONT to a real Times New Roman .ttf file. Searched:\n  - {searched}"
        )

    plt.rcParams.update(
        {
            "font.family": family,
            "font.serif": [family, "Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return family


TIMES_FAMILY = configure_times_font()


TARGETS = ("EuroSAT", "AID", "UCM")
SHOTS = (1, 5)


def read_confusion_csv(csv_path):
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {csv_path}")

    class_names = [item.strip() for item in rows[0][1:]]
    matrix_rows = []
    row_names = []
    for row in rows[1:]:
        if not row:
            continue
        row_names.append(row[0].strip())
        matrix_rows.append([float(value) for value in row[1:]])

    matrix = np.asarray(matrix_rows, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square: {csv_path}, shape={matrix.shape}")
    if len(class_names) != matrix.shape[1]:
        raise ValueError(f"header/class count mismatch: {csv_path}")
    if row_names and len(row_names) == len(class_names):
        class_names = row_names
    return matrix, class_names


def format_class_name(name):
    words = str(name).replace("_", " ").split()
    if not words:
        return str(name)
    return " ".join(word[:1].upper() + word[1:] for word in words)


def save_figure(fig, save_path, dpi=600):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    stem = save_path.with_suffix("")
    for suffix, kwargs in (
        (".png", {"dpi": dpi}),
        (".pdf", {}),
    ):
        out_path = stem.with_suffix(suffix)
        try:
            fig.savefig(
                out_path,
                bbox_inches="tight",
                pad_inches=0.08,
                facecolor="white",
                **kwargs,
            )
        except PermissionError:
            fallback = stem.with_name(f"{stem.name}_updated").with_suffix(suffix)
            fig.savefig(
                fallback,
                bbox_inches="tight",
                pad_inches=0.08,
                facecolor="white",
                **kwargs,
            )
            print(f"[locked] {out_path}")
            print(f"[saved] {fallback}")


def annotate_cells(ax, matrix, fontsize=9):
    n_rows, n_cols = matrix.shape
    for row in range(n_rows):
        for col in range(n_cols):
            value = float(matrix[row, col])
            color = "white" if value > 0.5 else "#4A4A4A"
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=fontsize,
                fontfamily=TIMES_FAMILY,
            )


def style_matrix_axis(ax, matrix, class_names, title, show_ylabel=True, xlabel_pad=8):
    display_names = [format_class_name(name) for name in class_names]
    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=xlabel_pad)
    ax.set_ylabel("True Label" if show_ylabel else "", fontsize=12, labelpad=8)

    ax.set_xticks(np.arange(len(display_names)))
    ax.set_yticks(np.arange(len(display_names)))
    ax.set_xticklabels(display_names, rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    ax.set_yticklabels(display_names, fontsize=10)
    ax.tick_params(axis="both", length=0)

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#D9D9D9", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#5F6B7A")
        spine.set_linewidth(0.8)

    annotate_cells(ax, matrix)
    return im


def add_colorbar(fig, image, axes):
    cbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=9, length=2)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label("Normalized Probability", fontsize=10, labelpad=8)
    return cbar


def title_from_info(title_info, model_name):
    if isinstance(title_info, dict):
        source = title_info.get("source", "NWPU")
        target = title_info.get("target", "Target")
        shot = title_info.get("shot", "")
        shot_text = f" | {shot}-shot" if shot else ""
        return f"{model_name} | {source}->{target}{shot_text}"
    return f"{model_name} | {title_info}"


def plot_confusion_matrix_pair(cm_baseline, cm_sganet, class_names, save_path, title_info):
    """Plot Baseline vs SGA-Net normalized confusion matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.8), constrained_layout=True)
    image = style_matrix_axis(
        axes[0],
        cm_baseline,
        class_names,
        title_from_info(title_info, "Baseline"),
        show_ylabel=True,
    )
    style_matrix_axis(
        axes[1],
        cm_sganet,
        class_names,
        title_from_info(title_info, "SGA-Net"),
        show_ylabel=False,
    )
    add_colorbar(fig, image, axes.ravel().tolist())
    axes[0].text(0.5, -0.30, "(a) Baseline", transform=axes[0].transAxes, ha="center", va="top", fontsize=14)
    axes[1].text(0.5, -0.30, "(b) SGA-Net", transform=axes[1].transAxes, ha="center", va="top", fontsize=14)
    save_figure(fig, save_path)
    plt.close(fig)


def plot_confusion_matrix_single(matrix, class_names, save_path, title, panel_label=None):
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.8), constrained_layout=True)
    image = style_matrix_axis(ax, matrix, class_names, title, show_ylabel=True)
    add_colorbar(fig, image, [ax])
    if panel_label:
        ax.text(0.5, -0.30, panel_label, transform=ax.transAxes, ha="center", va="top", fontsize=14)
    save_figure(fig, save_path)
    plt.close(fig)


def plot_confusion_grid(items, save_path):
    if not items:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 9.4), constrained_layout=False)
    fig.subplots_adjust(
        left=0.075,
        right=0.900,
        top=0.935,
        bottom=0.120,
        wspace=0.320,
        hspace=0.760,
    )
    image = None
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for idx, (ax, item) in enumerate(zip(axes.ravel(), items)):
        image = style_matrix_axis(
            ax,
            item["matrix"],
            item["class_names"],
            item["title"],
            show_ylabel=(idx % 3 == 0),
            xlabel_pad=5,
        )
        ax.text(
            0.5,
            -0.345,
            f"{labels[idx]} {item['caption']}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )
    for ax in axes.ravel()[len(items):]:
        ax.set_axis_off()

    cbar_ax = fig.add_axes([0.915, 0.160, 0.018, 0.735])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=9, length=2)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label("Normalized Probability", fontsize=10, labelpad=8)
    save_figure(fig, save_path)
    plt.close(fig)


def find_normalized_csv(input_dir, source, target, shot, suffix):
    folder = Path(input_dir) / f"{source}_to_{target}"
    direct = folder / f"confusion_{source}_to_{target}_{shot}shot_{suffix}_normalized.csv"
    if direct.is_file():
        return direct
    matches = sorted(folder.glob(f"*{shot}shot*{suffix}*normalized.csv"))
    return matches[0] if matches else None


def batch_redraw(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    grid_items = []
    missing = []

    for shot in args.shots:
        for target in args.targets:
            csv_path = find_normalized_csv(input_dir, args.source_dataset, target, shot, args.model_suffix)
            if csv_path is None:
                missing.append(f"{args.source_dataset}->{target} {shot}-shot")
                continue

            matrix, class_names = read_confusion_csv(csv_path)
            out_subdir = output_dir / f"{args.source_dataset}_to_{target}"
            out_stem = out_subdir / f"confusion_paper_{args.source_dataset}_to_{target}_{shot}shot_{args.model_suffix}"
            title = f"SGA-Net | {args.source_dataset}->{target} | {shot}-shot"
            caption = f"{args.source_dataset}->{target} {shot}-shot"
            plot_confusion_matrix_single(matrix, class_names, out_stem, title)

            grid_items.append(
                {
                    "matrix": matrix,
                    "class_names": class_names,
                    "title": f"{args.source_dataset}->{target} | {shot}-shot",
                    "caption": caption,
                }
            )
            print(f"[saved] {out_stem.with_suffix('.png')}")
            print(f"[saved] {out_stem.with_suffix('.pdf')}")

    if missing:
        print("[missing] These matrices were not found:")
        for item in missing:
            print(f"  - {item}")
        if args.require_complete:
            raise RuntimeError("Missing required confusion matrices.")

    if args.make_grid and grid_items:
        # Keep the thesis grid ordered by shot first, then target.
        grid_stem = output_dir / f"confusion_paper_{args.source_dataset}_source_all_targets_{args.model_suffix}"
        plot_confusion_grid(grid_items, grid_stem)
        print(f"[saved] {grid_stem.with_suffix('.png')}")
        print(f"[saved] {grid_stem.with_suffix('.pdf')}")


def pair_mode(args):
    cm_base, names_base = read_confusion_csv(args.baseline_csv)
    cm_sga, names_sga = read_confusion_csv(args.sganet_csv)
    if names_base != names_sga:
        raise ValueError("baseline and SGA-Net class names do not match")
    if cm_base.shape != cm_sga.shape:
        raise ValueError("baseline and SGA-Net matrix shapes do not match")
    title_info = {"source": args.source_dataset, "target": args.target, "shot": args.shot}
    plot_confusion_matrix_pair(cm_base, cm_sga, names_base, args.pair_output, title_info)
    print(f"[saved] {Path(args.pair_output).with_suffix('.png')}")
    print(f"[saved] {Path(args.pair_output).with_suffix('.pdf')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot paper-style confusion matrices from normalized CSV files.")
    parser.add_argument("--input_dir", type=str, default="output/confusion_nwpu_source_20260410_v1")
    parser.add_argument("--output_dir", type=str, default="output/confusion_nwpu_source_20260410_v1_paper")
    parser.add_argument("--source_dataset", type=str, default="NWPU")
    parser.add_argument("--targets", type=lambda value: [item.strip() for item in value.split(",") if item.strip()], default=list(TARGETS))
    parser.add_argument("--shots", type=lambda value: [int(item.strip()) for item in value.split(",") if item.strip()], default=list(SHOTS))
    parser.add_argument("--model_suffix", type=str, default="gnn")
    parser.add_argument("--make_grid", action="store_true", default=True)
    parser.add_argument("--require_complete", action="store_true",
                        help="Fail if any requested source/target/shot matrix is missing.")

    parser.add_argument("--baseline_csv", type=str, default="")
    parser.add_argument("--sganet_csv", type=str, default="")
    parser.add_argument("--pair_output", type=str, default="output/confusion_pair_paper")
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--shot", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.baseline_csv and args.sganet_csv:
        pair_mode(args)
    else:
        batch_redraw(args)


if __name__ == "__main__":
    main()
