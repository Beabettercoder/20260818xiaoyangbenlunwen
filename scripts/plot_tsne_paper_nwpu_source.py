#!/usr/bin/env python3
"""Paper-friendly shared t-SNE plots for NWPU-source visualization.

This script recomputes features and shared t-SNE coordinates from the real
models/data. It does not redraw from existing PNG files.
"""

import argparse
import csv
import os
import random
import re
import sys
from collections import defaultdict

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from torchvision import transforms


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from methods.StyleAdv_RN_GNN import StyleAdvGNN
from methods.backbone_multiblock import model_dict


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

COLOR_PALETTE = [
    "#00D7D7",  # cyan
    "#000000",  # black
    "#0000CC",  # blue
    "#7A3CFF",  # violet
    "#A63A35",  # brown red
    "#6BA6A6",  # muted teal
    "#66EE00",  # bright green
    "#FF7F50",  # coral
    "#6FA8DC",  # light blue
    "#E6193C",  # crimson
    "#C49A6C",
    "#3D96A8",
    "#B66595",
    "#79A85E",
    "#D45D5D",
    "#5B84B1",
    "#BDA84E",
    "#7E70C9",
    "#8E8E8E",
    "#2FA376",
    "#E28F8F",
    "#4E6E7D",
    "#A679D2",
    "#A9B72F",
    "#3589C4",
    "#CD7F45",
    "#4DAA57",
    "#9661A5",
    "#D69B2E",
    "#268EA8",
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p", "*", "8", "d"]
DOT_MARKER = "o"

TIMES_FONT_ENV = "TSNE_PAPER_FONT"
TIMES_FONT_BOLD_ENV = "TSNE_PAPER_FONT_BOLD"

KNOWN_CLASS_ORDER = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
    "Airport", "BareLand", "BaseballField", "Beach", "Bridge", "Center",
    "Church", "Commercial", "DenseResidential", "Desert", "Farmland",
    "Meadow", "MediumResidential", "Mountain", "Park", "Parking",
    "Playground", "Pond", "Port", "RailwayStation", "Resort", "School",
    "SparseResidential", "Square", "Stadium", "StorageTanks", "Viaduct",
    "agricultural", "airplane", "baseballdiamond", "buildings", "chaparral",
    "denseresidential", "freeway", "golfcourse", "harbor", "intersection",
    "mediumresidential", "mobilehomepark", "overpass", "parkinglot",
    "runway", "sparseresidential", "storagetanks", "tenniscourt",
]


def class_key(name):
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def times_font_candidates(bold=False):
    candidates = []
    env_name = TIMES_FONT_BOLD_ENV if bold else TIMES_FONT_ENV
    if os.environ.get(env_name):
        candidates.append(os.environ[env_name])
    if os.environ.get(TIMES_FONT_ENV):
        candidates.append(os.environ[TIMES_FONT_ENV])
    candidates.append(
        os.path.join(_REPO_ROOT, "fonts", "timesbd.ttf" if bold else "times.ttf")
    )
    return [item for item in candidates if item]


def resolve_times_font_path(bold=False):
    for path in times_font_candidates(bold=bold):
        if os.path.isfile(path):
            return path
    kind = "bold " if bold else ""
    searched = "\n  - ".join(times_font_candidates(bold=bold))
    raise RuntimeError(
        f"Times New Roman {kind}font was not found. Set {TIMES_FONT_ENV} "
        f"to a real Times New Roman .ttf file. Searched:\n  - {searched}"
    )


def configure_matplotlib_times():
    if not HAS_MATPLOTLIB:
        return
    regular_path = resolve_times_font_path(bold=False)
    bold_path = resolve_times_font_path(bold=True)
    font_manager.fontManager.addfont(regular_path)
    if bold_path != regular_path:
        font_manager.fontManager.addfont(bold_path)
    family_name = font_manager.FontProperties(fname=regular_path).get_name()
    if family_name != "Times New Roman":
        raise RuntimeError(
            f"The configured font is '{family_name}', not Times New Roman: {regular_path}"
        )
    plt.rcParams.update(
        {
            "font.family": family_name,
            "font.serif": [family_name],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_for_class(name):
    known_keys = [class_key(item) for item in KNOWN_CLASS_ORDER]
    key = class_key(name)
    if key in known_keys:
        idx = known_keys.index(key)
    else:
        idx = sum(ord(ch) for ch in key)
    return COLOR_PALETTE[idx % len(COLOR_PALETTE)], MARKERS[idx % len(MARKERS)]


def resolve_checkpoint_path(path):
    if os.path.isdir(path):
        return os.path.join(path, "best_model.tar")
    return path


def build_model(checkpoint_path, source_dataset, data_dir, device):
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
    filtered = {
        key: value for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


def scan_dataset_images(dataset_root):
    by_class = defaultdict(list)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"target dataset directory not found: {dataset_root}")

    for cls in sorted(os.listdir(dataset_root)):
        cls_dir = os.path.join(dataset_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fn in sorted(os.listdir(cls_dir)):
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                by_class[cls].append(os.path.join(cls_dir, fn))
    return by_class


def parse_classes(classes_arg):
    return [item.strip() for item in classes_arg.split(",") if item.strip()]


def select_class_names(by_class, classes_arg):
    available = sorted([name for name, paths in by_class.items() if paths])
    if not available:
        raise RuntimeError("no image classes found in target dataset")
    if not classes_arg.strip():
        return available

    key_to_name = {class_key(name): name for name in available}
    selected = []
    missing = []
    for item in parse_classes(classes_arg):
        key = class_key(item)
        if key in key_to_name:
            selected.append(key_to_name[key])
        else:
            missing.append(item)
    if missing:
        raise ValueError(f"requested classes not found: {', '.join(missing)}")
    return selected


def sample_images(by_class, class_names, samples_per_class, seed):
    rng = random.Random(seed)
    image_paths = []
    labels = []
    for label, cls in enumerate(class_names):
        paths = by_class[cls]
        if len(paths) <= samples_per_class:
            picked = list(paths)
        else:
            picked = rng.sample(paths, samples_per_class)
        image_paths.extend(picked)
        labels.extend([label] * len(picked))
    return image_paths, np.asarray(labels, dtype=np.int64)


@torch.no_grad()
def extract_features(model, image_paths, device, feature_space="backbone", batch_size=32):
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
        valid_count = x.size(0)
        if valid_count == 1:
            x = torch.cat([x, x.clone()], dim=0)

        raw = model.feature(x)
        if isinstance(raw, (tuple, list)):
            raw = raw[-1]
        z = model.fc(raw) if feature_space == "metric_fc" else raw
        z = F.normalize(z, dim=-1)[:valid_count]
        feats.extend(z.cpu().numpy())
        batch_tensors = []

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        batch_tensors.append(tfm(image))
        if len(batch_tensors) >= batch_size:
            flush_batch()
    flush_batch()

    return np.asarray(feats, dtype=np.float32)


def fit_shared_tsne(feat_base, feat_sga, seed, perplexity):
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


def compute_metrics(embedding, labels):
    unique = np.unique(labels)
    if len(unique) < 2 or len(labels) <= len(unique):
        return {"silhouette": None, "db": None, "intra_inter": None}

    try:
        silhouette = float(silhouette_score(embedding, labels))
    except Exception:
        silhouette = None
    try:
        db = float(davies_bouldin_score(embedding, labels))
    except Exception:
        db = None

    centroids = []
    intra_values = []
    for label in unique:
        points = embedding[labels == label]
        if len(points) == 0:
            continue
        centroid = points.mean(axis=0)
        centroids.append(centroid)
        intra_values.extend(np.linalg.norm(points - centroid, axis=1).tolist())

    inter_values = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            inter_values.append(float(np.linalg.norm(centroids[i] - centroids[j])))

    if intra_values and inter_values and np.mean(inter_values) > 1e-8:
        intra_inter = float(np.mean(intra_values) / np.mean(inter_values))
    else:
        intra_inter = None

    return {"silhouette": silhouette, "db": db, "intra_inter": intra_inter}


def fmt_metric(value, digits=3):
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def hex_to_rgba(hex_color, alpha=255):
    value = hex_color.lstrip("#")
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
        int(alpha),
    )


def load_serif_font(size, bold=False):
    return ImageFont.truetype(resolve_times_font_path(bold=bold), size=size)


def text_bbox(draw, xy, text, font):
    try:
        return draw.textbbox(xy, text, font=font)
    except Exception:
        width, height = draw.textsize(text, font=font)
        x, y = xy
        return (x, y, x + width, y + height)


def draw_centered_text(draw, center_x, y, text, font, fill):
    bbox = text_bbox(draw, (0, 0), text, font)
    draw.text((center_x - (bbox[2] - bbox[0]) / 2, y), text, font=font, fill=fill)


def data_to_canvas(points, xlim, ylim, box):
    x0, y0, width, height = box
    points = np.asarray(points, dtype=np.float32)
    x = (points[:, 0] - xlim[0]) / max(xlim[1] - xlim[0], 1e-8)
    y = (points[:, 1] - ylim[0]) / max(ylim[1] - ylim[0], 1e-8)
    canvas_x = x0 + x * width
    canvas_y = y0 + (1.0 - y) * height
    return np.stack([canvas_x, canvas_y], axis=1)


def marker_polygon(x, y, radius, marker):
    if marker == "s":
        return [(x - radius, y - radius), (x + radius, y - radius), (x + radius, y + radius), (x - radius, y + radius)]
    if marker == "^":
        return [(x, y - radius), (x + radius, y + radius), (x - radius, y + radius)]
    if marker == "v":
        return [(x - radius, y - radius), (x + radius, y - radius), (x, y + radius)]
    if marker == "D" or marker == "d":
        return [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
    if marker == "<":
        return [(x - radius, y), (x + radius, y - radius), (x + radius, y + radius)]
    if marker == ">":
        return [(x + radius, y), (x - radius, y - radius), (x - radius, y + radius)]
    if marker == "h":
        angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
        return [(x + radius * np.cos(a), y + radius * np.sin(a)) for a in angles]
    if marker == "p":
        angles = np.linspace(0, 2 * np.pi, 6)[:-1] - np.pi / 2
        return [(x + radius * np.cos(a), y + radius * np.sin(a)) for a in angles]
    if marker == "8":
        angles = np.linspace(0, 2 * np.pi, 9)[:-1] + np.pi / 8
        return [(x + radius * np.cos(a), y + radius * np.sin(a)) for a in angles]
    if marker == "*":
        pts = []
        for idx in range(10):
            angle = -np.pi / 2 + idx * np.pi / 5
            rr = radius if idx % 2 == 0 else radius * 0.45
            pts.append((x + rr * np.cos(angle), y + rr * np.sin(angle)))
        return pts
    return None


def draw_marker(draw, x, y, marker, color, radius=7, alpha=210, outline=(255, 255, 255, 230), hollow=False, width=2):
    fill = None if hollow else (color[0], color[1], color[2], alpha)
    stroke = (color[0], color[1], color[2], 255) if hollow else outline
    x = float(x)
    y = float(y)
    radius = float(radius)

    if marker == "o":
        box = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(box, fill=fill, outline=stroke, width=width)
        return
    if marker in {"P", "+"}:
        draw.line([(x - radius, y), (x + radius, y)], fill=(color[0], color[1], color[2], alpha), width=max(2, width + 1))
        draw.line([(x, y - radius), (x, y + radius)], fill=(color[0], color[1], color[2], alpha), width=max(2, width + 1))
        return
    if marker == "X":
        draw.line([(x - radius, y - radius), (x + radius, y + radius)], fill=(color[0], color[1], color[2], alpha), width=max(2, width + 1))
        draw.line([(x - radius, y + radius), (x + radius, y - radius)], fill=(color[0], color[1], color[2], alpha), width=max(2, width + 1))
        return

    pts = marker_polygon(x, y, radius, marker)
    if pts is None:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill, outline=stroke, width=width)
        return
    draw.polygon(pts, fill=fill)
    draw.line(pts + [pts[0]], fill=stroke, width=width)


def ellipse_polygon_points(points, steps=96):
    if len(points) < 3:
        return None
    try:
        cov = np.cov(points, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        if np.any(vals <= 0) or not np.all(np.isfinite(vals)):
            return None
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angles = np.linspace(0, 2 * np.pi, steps, endpoint=False)
        unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        scaled = unit * np.sqrt(vals)
        center = points.mean(axis=0)
        return center + scaled @ vecs.T
    except Exception:
        return None


def draw_metric_box(draw, box, metrics, fonts):
    x0, y0, x1, y1 = box
    text = (
        f"Silhouette higher {fmt_metric(metrics['silhouette'])}\n"
        f"Davies-Bouldin lower {fmt_metric(metrics['db'])}\n"
        f"Intra/Inter lower {fmt_metric(metrics['intra_inter'])}"
    )
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=(255, 255, 255, 232), outline=(214, 220, 229, 255), width=2)
    else:
        draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255, 232), outline=(214, 220, 229, 255), width=2)
    line_y = y0 + 14
    for line in text.splitlines():
        draw.text((x0 + 14, line_y), line, font=fonts["metric"], fill=(55, 65, 81, 255))
        line_y += 28


def draw_tsne_panel_pil(draw, embedding, labels, class_names, title, metrics, xlim, ylim, box, fonts):
    x0, y0, width, height = box
    draw.rectangle([x0, y0, x0 + width, y0 + height], fill=(255, 255, 255, 255), outline=(216, 222, 232, 255), width=2)

    for frac in np.linspace(0.2, 0.8, 4):
        x = x0 + width * frac
        y = y0 + height * frac
        draw.line([(x, y0), (x, y0 + height)], fill=(231, 235, 240, 255), width=1)
        draw.line([(x0, y), (x0 + width, y)], fill=(231, 235, 240, 255), width=1)

    draw_centered_text(draw, x0 + width / 2, y0 - 58, title, fonts["subtitle"], (24, 34, 48, 255))

    for label, cls in enumerate(class_names):
        points = embedding[labels == label]
        if len(points) == 0:
            continue
        color_hex, marker = style_for_class(cls)
        color = hex_to_rgba(color_hex, 255)

        poly = ellipse_polygon_points(points)
        if poly is not None:
            poly_canvas = data_to_canvas(poly, xlim, ylim, box)
            poly_pts = [(float(x), float(y)) for x, y in poly_canvas]
            draw.polygon(poly_pts, fill=(color[0], color[1], color[2], 28))
            draw.line(poly_pts + [poly_pts[0]], fill=(color[0], color[1], color[2], 78), width=2)

    for label, cls in enumerate(class_names):
        points = embedding[labels == label]
        if len(points) == 0:
            continue
        color_hex, marker = style_for_class(cls)
        color = hex_to_rgba(color_hex, 255)
        coords = data_to_canvas(points, xlim, ylim, box)
        for px, py in coords:
            draw_marker(draw, px, py, marker, color, radius=7, alpha=200, outline=(255, 255, 255, 235), width=2)
        centroid = data_to_canvas(points.mean(axis=0, keepdims=True), xlim, ylim, box)[0]
        draw_marker(draw, centroid[0], centroid[1], marker, color, radius=16, alpha=255, hollow=True, width=4)

    draw_metric_box(draw, (x0 + width - 360, y0 + 22, x0 + width - 24, y0 + 122), metrics, fonts)


def draw_legend_pil(draw, class_names, canvas_w, y0, fonts):
    ncol = min(5, max(2, int(np.ceil(len(class_names) / 2.0))))
    rows = int(np.ceil(len(class_names) / ncol))
    left = 145
    right = canvas_w - 145
    col_w = (right - left) / ncol
    row_h = 34
    for idx, cls in enumerate(class_names):
        col = idx % ncol
        row = idx // ncol
        x = left + col * col_w
        y = y0 + row * row_h
        color_hex, marker = style_for_class(cls)
        color = hex_to_rgba(color_hex, 255)
        draw_marker(draw, x + 12, y + 13, marker, color, radius=8, alpha=230, outline=(255, 255, 255, 240), width=2)
        draw.text((x + 30, y), cls, font=fonts["legend"], fill=(31, 41, 55, 255))


def save_pil_outputs(canvas, png_path, pdf_path, dpi):
    rgb = Image.new("RGB", canvas.size, (255, 255, 255))
    rgb.paste(canvas, mask=canvas.split()[-1])
    rgb.save(png_path, dpi=(dpi, dpi))
    rgb.save(pdf_path, "PDF", resolution=dpi)


def add_sigma_ellipse(ax, points, color):
    if len(points) < 3:
        return
    try:
        cov = np.cov(points, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        if np.any(vals <= 0) or not np.all(np.isfinite(vals)):
            return
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2.0 * np.sqrt(vals)
        center = points.mean(axis=0)
        ellipse = Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            linewidth=1.0,
            alpha=0.11,
            zorder=1,
        )
        ax.add_patch(ellipse)
    except Exception:
        return


def plot_panel(ax, embedding, labels, class_names, title, metrics, xlim, ylim):
    ax.set_title(title, fontsize=12, fontweight="semibold", pad=8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#E7EBF0", linewidth=0.7)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    for spine in ax.spines.values():
        spine.set_color("#D8DEE8")
        spine.set_linewidth(0.8)

    for label, cls in enumerate(class_names):
        points = embedding[labels == label]
        if len(points) == 0:
            continue
        color, marker = style_for_class(cls)
        add_sigma_ellipse(ax, points, color)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=22,
            c=color,
            marker=marker,
            alpha=0.78,
            edgecolors="#FFFFFF",
            linewidths=0.45,
            zorder=2,
        )
        centroid = points.mean(axis=0)
        ax.scatter(
            [centroid[0]],
            [centroid[1]],
            s=115,
            facecolors="none",
            edgecolors=color,
            marker=marker,
            linewidths=1.6,
            zorder=3,
        )

    metric_text = (
        f"Silhouette ↑ {fmt_metric(metrics['silhouette'])}\n"
        f"Davies-Bouldin ↓ {fmt_metric(metrics['db'])}\n"
        f"Intra/Inter ↓ {fmt_metric(metrics['intra_inter'])}"
    )
    ax.text(
        0.985,
        0.985,
        metric_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFFFFF", edgecolor="#D6DCE5", alpha=0.88),
    )


def make_legend_handles(class_names):
    handles = []
    for cls in class_names:
        color, marker = style_for_class(cls)
        handles.append(Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            label=cls,
            markerfacecolor=color,
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.6,
            markersize=7,
            alpha=0.95,
        ))
    return handles


def plot_tsne_figure_matplotlib(source, target, class_names, labels, emb_base, emb_sga, seed, output_dir, figure_tag, dpi):
    all_emb = np.vstack([emb_base, emb_sga])
    x_pad = max((all_emb[:, 0].max() - all_emb[:, 0].min()) * 0.06, 1e-3)
    y_pad = max((all_emb[:, 1].max() - all_emb[:, 1].min()) * 0.06, 1e-3)
    xlim = (all_emb[:, 0].min() - x_pad, all_emb[:, 0].max() + x_pad)
    ylim = (all_emb[:, 1].min() - y_pad, all_emb[:, 1].max() + y_pad)

    metrics_base = compute_metrics(emb_base, labels)
    metrics_sga = compute_metrics(emb_sga, labels)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.2), dpi=dpi)
    fig.patch.set_facecolor("#FFFFFF")
    plot_panel(axes[0], emb_base, labels, class_names, "Baseline", metrics_base, xlim, ylim)
    plot_panel(axes[1], emb_sga, labels, class_names, "SGA-Net", metrics_sga, xlim, ylim)
    fig.suptitle(f"Cross-domain t-SNE: {source} -> {target}", fontsize=14, fontweight="semibold", y=0.965)

    handles = make_legend_handles(class_names)
    ncol = min(5, max(2, int(np.ceil(len(class_names) / 2.0))))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=ncol,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.2,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.045, right=0.985, top=0.90, bottom=0.20, wspace=0.08)

    os.makedirs(output_dir, exist_ok=True)
    stem = f"tsne_paper_{source}_to_{target}_{figure_tag}_seed{seed}"
    png_path = os.path.join(output_dir, f"{stem}.png")
    pdf_path = os.path.join(output_dir, f"{stem}.pdf")
    metrics_path = os.path.join(output_dir, f"{stem}_metrics.csv")
    coord_path = os.path.join(output_dir, f"{stem}_coords.npz")

    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    with open(metrics_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "silhouette", "davies_bouldin", "intra_inter"])
        writer.writerow(["Baseline", metrics_base["silhouette"], metrics_base["db"], metrics_base["intra_inter"]])
        writer.writerow(["SGA-Net", metrics_sga["silhouette"], metrics_sga["db"], metrics_sga["intra_inter"]])

    np.savez(
        coord_path,
        emb_base=emb_base,
        emb_sga=emb_sga,
        labels=labels,
        class_names=np.asarray(class_names, dtype=object),
    )

    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")
    print(f"[saved] {metrics_path}")
    print(f"[saved] {coord_path}")


def compute_axis_limits(embedding, pad_ratio=0.025):
    x_pad = max((embedding[:, 0].max() - embedding[:, 0].min()) * pad_ratio, 1e-3)
    y_pad = max((embedding[:, 1].max() - embedding[:, 1].min()) * pad_ratio, 1e-3)
    return (
        (embedding[:, 0].min() - x_pad, embedding[:, 0].max() + x_pad),
        (embedding[:, 1].min() - y_pad, embedding[:, 1].max() + y_pad),
    )


def target_title_name(target):
    return {"EuroSAT": "ES"}.get(target, target)


def draw_paper_panel(ax, embedding, labels, class_names, xlim, ylim, point_size):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("auto", adjustable="box")
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)
    ax.grid(False)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    for spine in ax.spines.values():
        spine.set_color("#303030")
        spine.set_linewidth(0.65)

    for label, cls in enumerate(class_names):
        points = embedding[labels == label]
        if len(points) == 0:
            continue
        color = COLOR_PALETTE[label % len(COLOR_PALETTE)]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=point_size,
            c=color,
            marker=DOT_MARKER,
            alpha=0.92,
            edgecolors="none",
            linewidths=0,
            zorder=2,
            rasterized=True,
        )


def make_paper_legend_handles(class_names):
    handles = []
    for idx, cls in enumerate(class_names):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        handles.append(Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            label=cls,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0,
            markersize=5.2,
            alpha=0.95,
        ))
    return handles


def plot_tsne_figure_matplotlib(source, target, class_names, labels, emb_base, emb_sga,
                                seed, output_dir, figure_tag, dpi, axis_mode="panel",
                                point_size=7.0):
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib backend requested, but matplotlib is not installed")
    configure_matplotlib_times()

    metrics_base = compute_metrics(emb_base, labels)
    metrics_sga = compute_metrics(emb_sga, labels)

    if axis_mode == "shared":
        all_emb = np.vstack([emb_base, emb_sga])
        shared_limits = compute_axis_limits(all_emb)
        base_limits = shared_limits
        sga_limits = shared_limits
    else:
        base_limits = compute_axis_limits(emb_base)
        sga_limits = compute_axis_limits(emb_sga)

    legend_cols = 1 if len(class_names) <= 14 else 2
    fig_w = 7.2 if legend_cols == 1 else 8.7
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 9.0), dpi=dpi)
    fig.patch.set_facecolor("#FFFFFF")

    draw_paper_panel(axes[0], emb_base, labels, class_names, base_limits[0], base_limits[1], point_size)
    draw_paper_panel(axes[1], emb_sga, labels, class_names, sga_limits[0], sga_limits[1], point_size)

    fig.suptitle(f"{source}\u2192{target_title_name(target)}", fontsize=18, fontweight="bold", y=0.955)
    axes[0].text(
        -0.055,
        0.50,
        "Baseline",
        transform=axes[0].transAxes,
        ha="right",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    axes[1].text(
        -0.055,
        0.50,
        "SGA-Net\n(ours)",
        transform=axes[1].transAxes,
        ha="right",
        va="center",
        fontsize=15,
        fontweight="bold",
        linespacing=0.9,
    )

    handles = make_paper_legend_handles(class_names)
    for ax in axes:
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.045, 1.0),
            ncol=legend_cols,
            frameon=True,
            fancybox=False,
            edgecolor="#D0D0D0",
            facecolor="#FFFFFF",
            fontsize=7.0 if len(class_names) <= 14 else 6.0,
            columnspacing=0.85,
            handlelength=0.9,
            handletextpad=0.45,
            borderpad=0.35,
        )
    fig.subplots_adjust(
        left=0.19,
        right=0.78 if legend_cols == 1 else 0.72,
        top=0.915,
        bottom=0.055,
        hspace=0.015,
    )

    os.makedirs(output_dir, exist_ok=True)
    stem = f"tsne_paper_{source}_to_{target}_{figure_tag}_seed{seed}"
    png_path = os.path.join(output_dir, f"{stem}.png")
    pdf_path = os.path.join(output_dir, f"{stem}.pdf")
    metrics_path = os.path.join(output_dir, f"{stem}_metrics.csv")
    coord_path = os.path.join(output_dir, f"{stem}_coords.npz")

    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    with open(metrics_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "silhouette", "davies_bouldin", "intra_inter"])
        writer.writerow(["Baseline", metrics_base["silhouette"], metrics_base["db"], metrics_base["intra_inter"]])
        writer.writerow(["SGA-Net", metrics_sga["silhouette"], metrics_sga["db"], metrics_sga["intra_inter"]])

    np.savez(
        coord_path,
        emb_base=emb_base,
        emb_sga=emb_sga,
        labels=labels,
        class_names=np.asarray(class_names, dtype=object),
    )

    print(f"[backend] matplotlib-paper")
    print(f"[font] Times New Roman")
    print(f"[axis_mode] {axis_mode}")
    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")
    print(f"[saved] {metrics_path}")
    print(f"[saved] {coord_path}")


def plot_tsne_figure_pil(source, target, class_names, labels, emb_base, emb_sga, seed, output_dir, figure_tag, dpi):
    all_emb = np.vstack([emb_base, emb_sga])
    x_pad = max((all_emb[:, 0].max() - all_emb[:, 0].min()) * 0.06, 1e-3)
    y_pad = max((all_emb[:, 1].max() - all_emb[:, 1].min()) * 0.06, 1e-3)
    xlim = (all_emb[:, 0].min() - x_pad, all_emb[:, 0].max() + x_pad)
    ylim = (all_emb[:, 1].min() - y_pad, all_emb[:, 1].max() + y_pad)

    metrics_base = compute_metrics(emb_base, labels)
    metrics_sga = compute_metrics(emb_sga, labels)

    os.makedirs(output_dir, exist_ok=True)
    stem = f"tsne_paper_{source}_to_{target}_{figure_tag}_seed{seed}"
    png_path = os.path.join(output_dir, f"{stem}.png")
    pdf_path = os.path.join(output_dir, f"{stem}.pdf")
    metrics_path = os.path.join(output_dir, f"{stem}_metrics.csv")
    coord_path = os.path.join(output_dir, f"{stem}_coords.npz")

    canvas_w = 3600
    canvas_h = 2120 if len(class_names) > 16 else 1980
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")

    fonts = {
        "title": load_serif_font(54, bold=True),
        "subtitle": load_serif_font(40, bold=True),
        "metric": load_serif_font(24, bold=False),
        "legend": load_serif_font(25, bold=False),
    }

    draw_centered_text(
        draw,
        canvas_w / 2,
        54,
        f"Cross-domain t-SNE: {source} -> {target}",
        fonts["title"],
        (17, 24, 39, 255),
    )

    panel_top = 190
    panel_w = 1480
    panel_h = 1240
    left_x = 150
    gap = 140
    right_x = left_x + panel_w + gap
    draw_tsne_panel_pil(
        draw,
        emb_base,
        labels,
        class_names,
        "Baseline",
        metrics_base,
        xlim,
        ylim,
        (left_x, panel_top, panel_w, panel_h),
        fonts,
    )
    draw_tsne_panel_pil(
        draw,
        emb_sga,
        labels,
        class_names,
        "SGA-Net",
        metrics_sga,
        xlim,
        ylim,
        (right_x, panel_top, panel_w, panel_h),
        fonts,
    )

    legend_y = panel_top + panel_h + 145
    draw_legend_pil(draw, class_names, canvas_w, legend_y, fonts)

    save_pil_outputs(canvas, png_path, pdf_path, dpi)

    with open(metrics_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "silhouette", "davies_bouldin", "intra_inter"])
        writer.writerow(["Baseline", metrics_base["silhouette"], metrics_base["db"], metrics_base["intra_inter"]])
        writer.writerow(["SGA-Net", metrics_sga["silhouette"], metrics_sga["db"], metrics_sga["intra_inter"]])

    np.savez(
        coord_path,
        emb_base=emb_base,
        emb_sga=emb_sga,
        labels=labels,
        class_names=np.asarray(class_names, dtype=object),
    )

    print(f"[backend] PIL")
    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")
    print(f"[saved] {metrics_path}")
    print(f"[saved] {coord_path}")


def plot_tsne_figure(source, target, class_names, labels, emb_base, emb_sga, seed,
                     output_dir, figure_tag, dpi, backend="matplotlib", axis_mode="panel",
                     point_size=7.0):
    if backend in {"matplotlib", "auto"} and not HAS_MATPLOTLIB:
        raise RuntimeError(
            "matplotlib is required for the paper t-SNE layout. Install matplotlib "
            "or explicitly pass --plot_backend pil for the legacy fallback."
        )
    if backend == "pil":
        return plot_tsne_figure_pil(source, target, class_names, labels, emb_base, emb_sga, seed, output_dir, figure_tag, dpi)
    return plot_tsne_figure_matplotlib(
        source, target, class_names, labels, emb_base, emb_sga, seed, output_dir,
        figure_tag, dpi, axis_mode=axis_mode, point_size=point_size,
    )


def run_once(args, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_root = os.path.join(args.data_dir, args.target_dataset)
    by_class = scan_dataset_images(target_root)
    class_names = select_class_names(by_class, args.classes)
    image_paths, labels = sample_images(by_class, class_names, args.samples_per_class, seed)
    if len(image_paths) == 0:
        raise RuntimeError(f"no sampled images for {args.target_dataset}")

    baseline_ckpt = resolve_checkpoint_path(args.baseline_ckpt)
    sga_ckpt = resolve_checkpoint_path(args.sga_ckpt)
    if not os.path.isfile(baseline_ckpt):
        raise FileNotFoundError(f"baseline checkpoint not found: {baseline_ckpt}")
    if not os.path.isfile(sga_ckpt):
        raise FileNotFoundError(f"SGA checkpoint not found: {sga_ckpt}")

    print(f"[info] target={args.target_dataset}, seed={seed}, classes={len(class_names)}, images={len(image_paths)}")
    print(f"[info] device={device}, feature_space={args.feature_space}")

    baseline = build_model(baseline_ckpt, args.source_dataset, args.data_dir, device)
    sga = build_model(sga_ckpt, args.source_dataset, args.data_dir, device)

    feat_base = extract_features(
        baseline,
        image_paths,
        device,
        feature_space=args.feature_space,
        batch_size=args.batch_size,
    )
    feat_sga = extract_features(
        sga,
        image_paths,
        device,
        feature_space=args.feature_space,
        batch_size=args.batch_size,
    )
    emb_base, emb_sga = fit_shared_tsne(feat_base, feat_sga, seed, args.perplexity)
    plot_tsne_figure(
        args.source_dataset,
        args.target_dataset,
        class_names,
        labels,
        emb_base,
        emb_sga,
        seed,
        args.output_dir,
        args.figure_tag,
        args.dpi,
        args.plot_backend,
        args.axis_mode,
        args.point_size,
    )


def main():
    parser = argparse.ArgumentParser(description="Draw paper-friendly shared t-SNE for NWPU-source scenarios")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--source_dataset", type=str, default="NWPU")
    parser.add_argument("--target_dataset", type=str, required=True)
    parser.add_argument("--baseline_ckpt", type=str, required=True)
    parser.add_argument("--sga_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--classes", type=str, default="")
    parser.add_argument("--samples_per_class", type=int, default=80)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--feature_space", type=str, default="backbone", choices=["backbone", "metric_fc"])
    parser.add_argument("--figure_tag", type=str, default="main")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--plot_backend", type=str, default="matplotlib", choices=["auto", "matplotlib", "pil"])
    parser.add_argument("--axis_mode", type=str, default="panel", choices=["panel", "shared"],
                        help="panel fills each subplot independently; shared uses one common axis range")
    parser.add_argument("--point_size", type=float, default=7.0)
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    for seed in seeds:
        run_once(args, seed)


if __name__ == "__main__":
    main()
