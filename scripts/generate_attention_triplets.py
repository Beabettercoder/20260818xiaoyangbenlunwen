#!/usr/bin/env python3
"""Generate baseline vs SGA attention triplets for cross-domain qualitative comparison.

Output per sample: [Original | Baseline attention | SGA attention]
"""

import argparse
import os
import random
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
from torchvision import transforms

from methods.backbone_multiblock import model_dict
from methods.StyleAdv_RN_GNN import StyleAdvGNN


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_class_images(dataset_root: str, class_name: str) -> List[str]:
    class_dir = os.path.join(dataset_root, class_name)
    if not os.path.isdir(class_dir):
        return []
    files = []
    for fn in os.listdir(class_dir):
        ext = os.path.splitext(fn)[1].lower()
        if ext in IMG_EXTS:
            files.append(os.path.join(class_dir, fn))
    files.sort()
    return files


def resolve_class_dir(dataset_root: str, class_query: str) -> str:
    if os.path.isdir(os.path.join(dataset_root, class_query)):
        return class_query
    # case-insensitive fallback
    mapping = {x.lower(): x for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))}
    return mapping.get(class_query.lower(), "")


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


def activation_attention(model, x: torch.Tensor) -> np.ndarray:
    # x: [1,3,224,224]
    with torch.no_grad():
        b1 = model.feature.forward_block1(x)
        b2 = model.feature.forward_block2(b1)
        b3 = model.feature.forward_block3(b2)
        b4 = model.feature.forward_block4(b3)
        # channel-mean activation as attention proxy
        att = b4.mean(dim=1, keepdim=True)
        att = F.relu(att)
        att = F.interpolate(att, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
        att = att[0, 0].cpu().numpy()

    att = att - att.min()
    if att.max() > 1e-8:
        att = att / att.max()
    return att


def jet_colormap(att: np.ndarray) -> np.ndarray:
    x = np.clip(att, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def overlay_heatmap(img_np: np.ndarray, att: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = jet_colormap(att)
    out = np.clip((1 - alpha) * img_np + alpha * heat, 0.0, 1.0)
    return out


def to_pil_image(img_np: np.ndarray) -> Image.Image:
    arr = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_triplet_images(img_np, base_map, sga_map, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    original = to_pil_image(img_np)
    baseline = to_pil_image(overlay_heatmap(img_np, base_map))
    sga = to_pil_image(overlay_heatmap(img_np, sga_map))

    original.save(os.path.join(out_dir, "original.png"))
    baseline.save(os.path.join(out_dir, "baseline.png"))
    sga.save(os.path.join(out_dir, "sga_net.png"))


def main():
    parser = argparse.ArgumentParser(description="Generate attention triplets (Original/Baseline/SGA)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--target_dataset", type=str, required=True)
    parser.add_argument("--source_dataset", type=str, required=True)
    parser.add_argument("--baseline_ckpt", type=str, required=True)
    parser.add_argument("--sga_ckpt", type=str, required=True)
    parser.add_argument(
        "--classes",
        type=str,
        required=True,
        help="Comma-separated class names, or __ALL__ to traverse all target classes",
    )
    parser.add_argument("--samples_per_class", type=int, default=4)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tgt_root = os.path.join(args.data_dir, args.target_dataset)
    if not os.path.isdir(tgt_root):
        raise FileNotFoundError(f"target dataset dir not found: {tgt_root}")

    baseline_model = build_model(args.baseline_ckpt, args.source_dataset, args.data_dir, device)
    sga_model = build_model(args.sga_ckpt, args.source_dataset, args.data_dir, device)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if args.classes.strip() == "__ALL__":
        classes = [
            cls for cls in sorted(os.listdir(tgt_root))
            if os.path.isdir(os.path.join(tgt_root, cls))
        ]
    else:
        classes = [x.strip() for x in args.classes.split(",") if x.strip()]
    exported = 0

    for cls_query in classes:
        cls = resolve_class_dir(tgt_root, cls_query)
        if not cls:
            print(f"[WARN] class not found in {args.target_dataset}: {cls_query}")
            continue

        imgs = list_class_images(tgt_root, cls)
        if not imgs:
            print(f"[WARN] no images for class: {cls}")
            continue

        pick = imgs if len(imgs) <= args.samples_per_class else random.sample(imgs, args.samples_per_class)

        for idx, fp in enumerate(pick):
            img = Image.open(fp).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            img_np = np.asarray(img.resize((224, 224)), dtype=np.float32) / 255.0

            base_map = activation_attention(baseline_model, x)
            sga_map = activation_attention(sga_model, x)

            stem = os.path.splitext(os.path.basename(fp))[0]
            sample_dir_name = stem
            out_dir = os.path.join(args.output_dir, sample_dir_name)
            if os.path.exists(out_dir):
                sample_dir_name = f"{cls}__{stem}"
                out_dir = os.path.join(args.output_dir, sample_dir_name)
            save_triplet_images(img_np, base_map, sga_map, out_dir)
            exported += 1

    print(f"[DONE] exported triplets: {exported}")
    print(f"[OUT] {args.output_dir}")


if __name__ == "__main__":
    main()
