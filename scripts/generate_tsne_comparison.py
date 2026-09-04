#!/usr/bin/env python3
"""Generate Baseline vs SGA t-SNE scatter plots for cross-domain scenarios.

Each scenario outputs a 1x2 plot:
- left: Baseline
- right: SGA-net

This script now supports:
- all target classes
- shared t-SNE fitting across baseline and SGA features
- selectable feature spaces (default: backbone representation)
"""

import argparse
import os
import random
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from methods.backbone_multiblock import model_dict
from methods.StyleAdv_RN_GNN import StyleAdvGNN


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PALETTE = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
    (255, 127, 0), (255, 255, 51), (166, 86, 40), (247, 129, 191),
    (153, 153, 153), (0, 191, 255), (60, 179, 113), (238, 130, 238),
    (205, 92, 92), (70, 130, 180), (154, 205, 50), (218, 112, 214),
]


def build_model(checkpoint_path: str, source_dataset: str, data_dir: str, device: torch.device):
    model = StyleAdvGNN(
        model_dict["ResNet10"],
        n_way=5,
        n_support=1,
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
    filtered = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


def scan_dataset_images(dataset_root):
    by_class = defaultdict(list)
    if not os.path.isdir(dataset_root):
        return by_class

    for cls in sorted(os.listdir(dataset_root)):
        cls_dir = os.path.join(dataset_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fn in os.listdir(cls_dir):
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                by_class[cls].append(os.path.join(cls_dir, fn))
    return by_class


@torch.no_grad()
def extract_features(model, image_paths, device, feature_space: str = "backbone", batch_size: int = 32):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    feats = []
    batch_tensors = []

    def flush_batch():
        nonlocal batch_tensors, feats
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors, dim=0).to(device)
        padded = False
        valid_count = x.size(0)
        if valid_count == 1:
            # model.fc contains BatchNorm1d(track_running_stats=False),
            # so a single-sample batch is invalid. Duplicate once and discard later.
            x = torch.cat([x, x.clone()], dim=0)
            padded = True
        raw = model.feature(x)
        if isinstance(raw, (tuple, list)):
            raw = raw[-1]

        if feature_space == "metric_fc":
            z = model.fc(raw)
        else:
            z = raw

        z = F.normalize(z, dim=-1)
        z = z[:valid_count] if padded else z
        feats.extend(z.cpu().numpy())
        batch_tensors = []

    for fp in image_paths:
        img = Image.open(fp).convert("RGB")
        batch_tensors.append(tfm(img))
        if len(batch_tensors) >= batch_size:
            flush_batch()
    flush_batch()

    return np.asarray(feats, dtype=np.float32)


def sample_images(by_class, n_classes, samples_per_class, seed):
    rng = random.Random(seed)
    all_classes = [c for c in by_class.keys() if len(by_class[c]) > 0]
    if len(all_classes) == 0:
        return [], [], []

    if n_classes <= 0 or len(all_classes) <= n_classes:
        pick_classes = all_classes
    else:
        pick_classes = rng.sample(all_classes, n_classes)

    image_paths = []
    labels = []
    class_names = []
    for idx, cls in enumerate(pick_classes):
        imgs = by_class[cls]
        if samples_per_class <= 0 or len(imgs) <= samples_per_class:
            pick = imgs
        else:
            pick = rng.sample(imgs, samples_per_class)
        image_paths.extend(pick)
        labels.extend([idx] * len(pick))
        class_names.append(cls)

    return image_paths, np.asarray(labels), class_names


def normalize_embedding(emb: np.ndarray, width: int, height: int, margin: int = 24) -> np.ndarray:
    emb = emb.astype(np.float32)
    mins = emb.min(axis=0)
    maxs = emb.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    norm = (emb - mins) / spans
    x = margin + norm[:, 0] * max(width - margin * 2, 1)
    y = margin + norm[:, 1] * max(height - margin * 2, 1)
    return np.stack([x, y], axis=1)


def draw_panel(draw: ImageDraw.ImageDraw, emb: np.ndarray, labels: np.ndarray, class_names, title: str,
               origin_x: int, origin_y: int, panel_w: int, panel_h: int, point_r: int = 3):
    font = ImageFont.load_default()
    draw.rectangle([origin_x, origin_y, origin_x + panel_w, origin_y + panel_h], outline=(180, 180, 180), width=1)
    draw.text((origin_x + 8, origin_y + 6), title, fill=(20, 20, 20), font=font)

    plot_y = origin_y + 24
    plot_h = panel_h - 32
    coords = normalize_embedding(emb, panel_w, plot_h)

    for i, cls in enumerate(class_names):
      mask = labels == i
      color = PALETTE[i % len(PALETTE)]
      for px, py in coords[mask]:
        x = int(origin_x + px)
        y = int(plot_y + py)
        draw.ellipse([x - point_r, y - point_r, x + point_r, y + point_r], fill=color, outline=color)


def draw_legend(draw: ImageDraw.ImageDraw, class_names, origin_x: int, origin_y: int, max_height: int):
    font = ImageFont.load_default()
    draw.text((origin_x, origin_y), "Classes", fill=(20, 20, 20), font=font)
    row_h = 18
    rows_per_col = max(1, (max_height - origin_y - 20) // row_h)
    for i, cls in enumerate(class_names):
        col_idx = i // rows_per_col
        row_idx = i % rows_per_col
        x = origin_x + col_idx * 190
        y = origin_y + 18 + row_idx * row_h
        color = PALETTE[i % len(PALETTE)]
        draw.rectangle([x, y + 2, x + 10, y + 12], fill=color, outline=color)
        draw.text((x + 16, y), cls[:26], fill=(20, 20, 20), font=font)


def fit_shared_tsne(feat_base: np.ndarray, feat_sga: np.ndarray, seed: int, perplexity: float):
    all_feat = np.concatenate([feat_base, feat_sga], axis=0).astype(np.float32)
    if all_feat.shape[0] >= 3:
        max_perplexity = max(2.0, min(perplexity, float(all_feat.shape[0] - 1) / 3.0))
    else:
        max_perplexity = 2.0

    if all_feat.shape[1] > 50 and all_feat.shape[0] > 50:
        pca_dim = min(50, all_feat.shape[0], all_feat.shape[1])
        all_feat = PCA(n_components=pca_dim, random_state=seed).fit_transform(all_feat)

    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=max_perplexity,
        init="pca",
        learning_rate="auto",
    )
    emb_all = tsne.fit_transform(all_feat)
    split = feat_base.shape[0]
    return emb_all[:split], emb_all[split:]


def run_scenario(data_dir, source, target, baseline_ckpt, sga_ckpt, output_dir,
                 n_classes, samples_per_class, seed, perplexity, feature_space):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_root = os.path.join(data_dir, target)
    by_class = scan_dataset_images(target_root)
    img_paths, labels, class_names = sample_images(by_class, n_classes, samples_per_class, seed)
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found for target dataset: {target_root}")

    baseline = build_model(baseline_ckpt, source, data_dir, device)
    sga = build_model(sga_ckpt, source, data_dir, device)

    feat_base = extract_features(baseline, img_paths, device, feature_space=feature_space)
    feat_sga = extract_features(sga, img_paths, device, feature_space=feature_space)
    emb_base, emb_sga = fit_shared_tsne(feat_base, feat_sga, seed, perplexity)

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"tsne_{source}_to_{target}_seed{seed}.png")
    panel_w = 640
    panel_h = 640
    rows_per_col = max(1, (panel_h - 20) // 18)
    legend_cols = max(1, int(np.ceil(len(class_names) / rows_per_col)))
    legend_w = max(240, legend_cols * 190 + 20)
    canvas = Image.new("RGB", (panel_w * 2 + legend_w + 40, panel_h + 20), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw_panel(draw, emb_base, labels, class_names, f"Baseline | {source}->{target}", 10, 10, panel_w, panel_h)
    draw_panel(draw, emb_sga, labels, class_names, f"SGA-net | {source}->{target}", 20 + panel_w, 10, panel_w, panel_h)
    draw_legend(draw, class_names, 30 + panel_w * 2, 20, panel_h)
    canvas.save(out_file)

    print(f"[DONE] {source}->{target} seed={seed} saved: {out_file}")


def parse_scenario_item(item: str):
    # format: source->target:baseline_ckpt:sga_ckpt
    parts = item.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid scenario: {item}")
    st, baseline_ckpt, sga_ckpt = parts
    if "->" not in st:
        raise ValueError(f"Invalid source->target in scenario: {item}")
    source, target = st.split("->", 1)
    return source.strip(), target.strip(), baseline_ckpt.strip(), sga_ckpt.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate t-SNE comparison for multiple cross-domain scenarios")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--scenarios", type=str, required=True,
                        help="Semicolon-separated items: source->target:baseline_ckpt:sga_ckpt")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_classes", type=int, default=0,
                        help="Number of target classes to visualize. <=0 means all classes.")
    parser.add_argument("--samples_per_class", type=int, default=40)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--perplexity", type=float, default=35.0)
    parser.add_argument("--feature_space", type=str, default="backbone", choices=["backbone", "metric_fc"])
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    scenarios = [x.strip() for x in args.scenarios.split(";") if x.strip()]

    for sc in scenarios:
        source, target, baseline_ckpt, sga_ckpt = parse_scenario_item(sc)
        for seed in seeds:
            run_scenario(
                args.data_dir,
                source,
                target,
                baseline_ckpt,
                sga_ckpt,
                args.output_dir,
                args.n_classes,
                args.samples_per_class,
                seed,
                args.perplexity,
                args.feature_space,
            )


if __name__ == "__main__":
    main()
