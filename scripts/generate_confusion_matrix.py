#!/usr/bin/env python3
"""Generate a normalized confusion matrix for a fixed few-shot class set.

The script repeatedly samples episodes from the same target-domain class set,
accumulates query predictions, and saves:

- PNG heatmap
- normalized CSV
- raw count CSV
- summary TXT
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.datamgr import TransformLoader, resolve_dataset_split_file
from methods.StyleAdv_RN_GNN import StyleAdvGNN
from methods.backbone_multiblock import model_dict
from methods.text_prompt import canonicalize_class_name, load_class_names


def resolve_checkpoint_path(path):
    if os.path.isdir(path):
        return os.path.join(path, "best_model.tar")
    return path


def build_model(checkpoint_path, source_dataset, data_dir, n_way, n_shot, device):
    model = StyleAdvGNN(
        model_dict["ResNet10"],
        n_way=n_way,
        n_support=n_shot,
        dataset_name=source_dataset,
        data_root=data_dir,
        text_guide_epsilon=1,
        text_guide_gradient=1,
        use_prompt_ensemble=True,
        text_calibration_weight=0.0,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("state", ckpt.get("model_state", ckpt))
    current = model.state_dict()
    filtered = {key: value for key, value in state.items() if key in current and current[key].shape == value.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


def load_split_meta(target_dataset, data_dir, split):
    split_file = resolve_dataset_split_file(target_dataset, split=split, data_root=data_dir)
    if split_file is None or not os.path.isfile(split_file):
        raise FileNotFoundError(f"split json not found for {target_dataset}:{split}")
    with open(split_file, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return meta, split_file


def build_label_index(meta):
    by_label = defaultdict(list)
    for idx, label in enumerate(meta["image_labels"]):
        by_label[int(label)].append(idx)
    return by_label


def resolve_class_ids(target_dataset, data_dir, by_label, n_way, classes_arg, seed):
    class_names = load_class_names(target_dataset, data_root=data_dir)
    label_to_name = {
        label: class_names[label] if label < len(class_names) else f"class_{label}"
        for label in by_label.keys()
    }

    if classes_arg:
        requested = [canonicalize_class_name(item.strip()) for item in classes_arg.split(",") if item.strip()]
        name_to_label = {canonicalize_class_name(name): label for label, name in label_to_name.items()}
        selected = []
        for item in requested:
            if item not in name_to_label:
                raise ValueError(f"class not found in split: {item}")
            selected.append(name_to_label[item])
    else:
        candidates = sorted(by_label.keys())
        if len(candidates) < n_way:
            raise ValueError(f"split only has {len(candidates)} classes, but n_way={n_way}")
        rng = random.Random(seed)
        selected = rng.sample(candidates, n_way)

    if len(selected) != n_way:
        raise ValueError(f"need exactly {n_way} classes, got {len(selected)}")

    selected_names = [label_to_name[label] for label in selected]
    return selected, selected_names


def normalize_image_path(path):
    path = os.path.join(path)
    return path.replace("DATACENTER/4", "share/test")


def sample_episode(meta, by_label, selected_labels, transform, n_shot, n_query, rng):
    per_class_tensors = []
    for label in selected_labels:
        indices = by_label[label]
        need = n_shot + n_query
        if len(indices) >= need:
            chosen = rng.sample(indices, need)
        else:
            chosen = [rng.choice(indices) for _ in range(need)]

        tensors = []
        for idx in chosen:
            image_path = normalize_image_path(meta["image_names"][idx])
            image = Image.open(image_path).convert("RGB")
            tensors.append(transform(image))
        per_class_tensors.append(torch.stack(tensors, dim=0))

    return torch.stack(per_class_tensors, dim=0)


def update_confusion(confusion, pred, n_way, n_query):
    truth = np.repeat(np.arange(n_way), n_query)
    for true_idx, pred_idx in zip(truth, pred):
        confusion[true_idx, pred_idx] += 1


def normalize_confusion(confusion):
    normalized = confusion.astype(np.float32)
    row_sums = normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return normalized / row_sums


def heatmap_color(value):
    value = float(np.clip(value, 0.0, 1.0))
    low = np.array([245, 247, 252], dtype=np.float32)
    high = np.array([49, 130, 189], dtype=np.float32)
    color = low * (1.0 - value) + high * value
    return tuple(color.astype(np.uint8).tolist())


def render_confusion_png(matrix, class_names, title, output_path):
    n = matrix.shape[0]
    cell = 88
    left_margin = 160
    top_margin = 110
    right_margin = 30
    bottom_margin = 90
    width = left_margin + n * cell + right_margin
    height = top_margin + n * cell + bottom_margin

    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((left_margin, 20), title, fill=(20, 20, 20), font=font)
    draw.text((left_margin, 42), "Normalized Confusion Matrix", fill=(60, 60, 60), font=font)

    for i, name in enumerate(class_names):
        x = left_margin + i * cell + 16
        draw.text((x, top_margin - 28), name[:12], fill=(20, 20, 20), font=font)
        y = top_margin + i * cell + 34
        draw.text((16, y), name[:16], fill=(20, 20, 20), font=font)

    draw.text((left_margin + (n * cell) // 2 - 45, height - 28), "Predicted Label", fill=(20, 20, 20), font=font)
    draw.text((16, top_margin - 60), "True Label", fill=(20, 20, 20), font=font)

    for row in range(n):
        for col in range(n):
            value = float(matrix[row, col])
            x0 = left_margin + col * cell
            y0 = top_margin + row * cell
            x1 = x0 + cell
            y1 = y0 + cell
            fill = heatmap_color(value)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(210, 210, 210))
            text = f"{value:.2f}"
            text_fill = (255, 255, 255) if value >= 0.55 else (15, 15, 15)
            draw.text((x0 + 24, y0 + 34), text, fill=text_fill, font=font)

    canvas.save(output_path)


def save_csv(matrix, class_names, path):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred"] + class_names)
        for idx, name in enumerate(class_names):
            writer.writerow([name] + [f"{float(value):.6f}" for value in matrix[idx]])


def main():
    parser = argparse.ArgumentParser(description="Generate confusion matrix for a fixed few-shot class set")
    parser.add_argument("--source_dataset", type=str, required=True)
    parser.add_argument("--target_dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="novel")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_shot", type=int, default=1)
    parser.add_argument("--n_query", type=int, default=15)
    parser.add_argument("--n_episodes", type=int, default=600)
    parser.add_argument("--classes", type=str, default="")
    parser.add_argument("--use_gnn", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument(
        "--no_png",
        action="store_true",
        help="Skip the simple raw PNG; CSV outputs can be redrawn with the paper-style plotting script.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    meta, split_file = load_split_meta(args.target_dataset, args.data_dir, args.split)
    by_label = build_label_index(meta)
    selected_labels, selected_names = resolve_class_ids(
        args.target_dataset,
        args.data_dir,
        by_label,
        args.n_way,
        args.classes,
        args.seed,
    )

    model = build_model(
        checkpoint_path,
        args.source_dataset,
        args.data_dir,
        args.n_way,
        args.n_shot,
        device,
    )
    model.n_query = args.n_query

    transform = TransformLoader(224).get_composed_transform(aug=False)
    confusion = np.zeros((args.n_way, args.n_way), dtype=np.int64)
    rng = random.Random(args.seed)

    print(f"[classes] {selected_names}")

    for episode_idx in range(args.n_episodes):
        episode = sample_episode(
            meta,
            by_label,
            selected_labels,
            transform,
            args.n_shot,
            args.n_query,
            rng,
        )
        with torch.no_grad():
            scores = model.set_forward(
                episode,
                is_feature=False,
                class_names=selected_names,
                use_gnn=bool(args.use_gnn),
            )
        pred = scores.argmax(dim=1).cpu().numpy()
        update_confusion(confusion, pred, args.n_way, args.n_query)

        if (episode_idx + 1) % 100 == 0:
            print(f"[{episode_idx + 1}/{args.n_episodes}] running...")

    normalized = normalize_confusion(confusion)
    mode_name = "gnn" if args.use_gnn else "proto"
    stem = f"confusion_{args.source_dataset}_to_{args.target_dataset}_{args.n_shot}shot_{mode_name}"
    png_path = os.path.join(args.output_dir, f"{stem}.png")
    raw_csv_path = os.path.join(args.output_dir, f"{stem}_counts.csv")
    norm_csv_path = os.path.join(args.output_dir, f"{stem}_normalized.csv")
    summary_path = os.path.join(args.output_dir, f"{stem}_summary.txt")

    title = args.title.strip() or f"{args.source_dataset}->{args.target_dataset} | {args.n_shot}-shot | {mode_name.upper()}"
    if not args.no_png:
        render_confusion_png(normalized, selected_names, title, png_path)
    save_csv(confusion, selected_names, raw_csv_path)
    save_csv(normalized, selected_names, norm_csv_path)

    diag = np.diag(normalized)
    overall = float(diag.mean()) * 100.0
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Confusion matrix summary\n")
        handle.write(f"source_dataset={args.source_dataset}\n")
        handle.write(f"target_dataset={args.target_dataset}\n")
        handle.write(f"split={args.split}\n")
        handle.write(f"checkpoint={checkpoint_path}\n")
        handle.write(f"n_way={args.n_way}\n")
        handle.write(f"n_shot={args.n_shot}\n")
        handle.write(f"n_query={args.n_query}\n")
        handle.write(f"n_episodes={args.n_episodes}\n")
        handle.write(f"use_gnn={int(bool(args.use_gnn))}\n")
        handle.write(f"classes={','.join(selected_names)}\n")
        handle.write(f"mean_diagonal_accuracy={overall:.4f}\n")
        handle.write(f"split_file={split_file}\n")

    if args.no_png:
        print(f"[skip raw png] {png_path}")
    else:
        print(f"[saved] {png_path}")
    print(f"[saved] {raw_csv_path}")
    print(f"[saved] {norm_csv_path}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
