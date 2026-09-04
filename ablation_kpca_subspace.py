#!/usr/bin/env python3
"""Subspace comparison for ProtoNet + L2 and GNN calibrator.

This script evaluates a trained StyleAdv/GNN checkpoint under several
feature-space transforms and produces thesis-friendly outputs:

1. No Subspace
2. Linear
3. Gaussian
4. Polynomial
5. Nonlinear

The ProtoNet branch operates directly in the transformed space with L2
distance by default. The GNN branch reconstructs transformed features back to
the original 128-d metric space before feeding them to the trained GNN.
"""

import argparse
import csv
import json
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import KernelPCA, PCA


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data.datamgr import SetDataManager, resolve_dataset_split_file
from methods.StyleAdv_RN_GNN import StyleAdvGNN
from methods.backbone_multiblock import model_dict
from utils.path_utils import absolute_path, validate_dataset_root


SUBSPACE_DIM = 64
KPCA_GAMMA = 1.0 / 128.0

SUBSPACE_SPECS = [
    ("no_subspace", "No Subspace"),
    ("pca_linear", "Linear"),
    ("kpca_rbf", "Gaussian"),
    ("kpca_poly", "Polynomial"),
    ("kpca_sigmoid", "Nonlinear"),
]
SUBSPACE_KEYS = [key for key, _ in SUBSPACE_SPECS]
SUBSPACE_LABELS = {key: label for key, label in SUBSPACE_SPECS}


def build_model(args, device):
    model_func = model_dict[args.model]
    model = StyleAdvGNN(
        model_func,
        n_way=args.n_way,
        n_support=args.n_shot,
        text_guide_epsilon=1,
        text_guide_gradient=1,
        dataset_name=args.source_dataset,
        data_root=args.data_dir,
        clip_model_name="ViT-B/32",
        use_prompt_ensemble=True,
        text_calibration_weight=0.0,
    ).to(device)

    ckpt_path = args.checkpoint_dir
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "best_model.tar")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state", ckpt.get("model_state", ckpt))
    current = model.state_dict()
    filtered = {
        key: value
        for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    dropped = len(state) - len(filtered)
    if dropped:
        print(f"[load] dropped {dropped} incompatible keys")
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


@torch.no_grad()
def extract_fc_features(model, x, device):
    x = x.to(device)
    n_way, total, channels, height, width = x.shape
    n_support = model.n_support

    x_flat = x.view(-1, channels, height, width)
    raw = model.feature(x_flat)
    if isinstance(raw, (tuple, list)):
        raw = raw[-1]

    z = model.fc(raw)
    z = z.view(n_way, -1, z.size(-1))

    support = z[:, :n_support].reshape(-1, z.size(-1)).cpu().numpy()
    query = z[:, n_support:].reshape(-1, z.size(-1)).cpu().numpy()
    return support, query


def build_transformer_candidates(name, input_dim=SUBSPACE_DIM):
    if name == "no_subspace":
        return [("identity", None)]
    if name == "pca_linear":
        return [("pca_linear", PCA(n_components=input_dim))]
    if name == "kpca_rbf":
        return [
            (
                "kpca_rbf_default",
                KernelPCA(
                    n_components=input_dim,
                    kernel="rbf",
                    gamma=KPCA_GAMMA,
                    fit_inverse_transform=True,
                ),
            ),
            (
                "kpca_rbf_arpack",
                KernelPCA(
                    n_components=input_dim,
                    kernel="rbf",
                    gamma=KPCA_GAMMA * 0.5,
                    fit_inverse_transform=True,
                    eigen_solver="arpack",
                    remove_zero_eig=True,
                ),
            ),
        ]
    if name == "kpca_poly":
        return [
            (
                "kpca_poly_default",
                KernelPCA(
                    n_components=input_dim,
                    kernel="poly",
                    degree=3,
                    gamma=KPCA_GAMMA,
                    coef0=1.0,
                    fit_inverse_transform=True,
                ),
            ),
            (
                "kpca_poly_arpack",
                KernelPCA(
                    n_components=input_dim,
                    kernel="poly",
                    degree=3,
                    gamma=KPCA_GAMMA * 0.5,
                    coef0=1.0,
                    fit_inverse_transform=True,
                    eigen_solver="arpack",
                    remove_zero_eig=True,
                ),
            ),
            (
                "kpca_poly_stable",
                KernelPCA(
                    n_components=input_dim,
                    kernel="poly",
                    degree=2,
                    gamma=KPCA_GAMMA * 0.25,
                    coef0=1.0,
                    fit_inverse_transform=True,
                    eigen_solver="arpack",
                    remove_zero_eig=True,
                ),
            ),
        ]
    if name == "kpca_sigmoid":
        return [
            (
                "kpca_sigmoid_default",
                KernelPCA(
                    n_components=input_dim,
                    kernel="sigmoid",
                    gamma=KPCA_GAMMA,
                    coef0=1.0,
                    fit_inverse_transform=True,
                ),
            ),
            (
                "kpca_sigmoid_stable",
                KernelPCA(
                    n_components=input_dim,
                    kernel="sigmoid",
                    gamma=KPCA_GAMMA * 0.25,
                    coef0=0.5,
                    fit_inverse_transform=True,
                    eigen_solver="arpack",
                    remove_zero_eig=True,
                ),
            ),
        ]
    raise ValueError(f"unknown subspace config: {name}")


def fit_and_transform(name, support_np, query_np):
    if name == "no_subspace":
        return support_np.copy(), query_np.copy(), None

    all_feats = np.concatenate([support_np, query_np], axis=0)
    n_samples = all_feats.shape[0]
    candidates = build_transformer_candidates(name)
    last_exc = None

    for candidate_name, transformer in candidates:
        if transformer is None:
            return support_np.copy(), query_np.copy(), None

        if hasattr(transformer, "n_components") and transformer.n_components >= n_samples:
            new_dim = max(1, n_samples - 1)
            warnings.warn(
                f"[{name}] n_components={transformer.n_components} >= n_samples={n_samples}, "
                f"reduce to {new_dim}"
            )
            transformer.set_params(n_components=new_dim)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                transformer.fit(all_feats)
            support_proj = transformer.transform(support_np)
            query_proj = transformer.transform(query_np)
            if candidate_name != candidates[0][0]:
                print(f"[subspace] {name} fallback -> {candidate_name}")
            return support_proj, query_proj, transformer
        except ValueError as exc:
            last_exc = exc
            if "negative eigenvalues" not in str(exc).lower():
                raise
            print(f"[subspace] {name} failed with {candidate_name}: {exc}")
            continue

    raise RuntimeError(f"all transformer candidates failed for {name}") from last_exc


def reconstruct_128(name, projected, transformer, original_128):
    if name == "no_subspace" or transformer is None:
        return original_128.astype(np.float32)
    return transformer.inverse_transform(projected).astype(np.float32)


def protonet_eval(support_proj, query_proj, n_way, n_support, n_query, metric="l2"):
    support = torch.from_numpy(support_proj).float()
    query = torch.from_numpy(query_proj).float()

    prototypes = support.view(n_way, n_support, -1).mean(dim=1)
    labels = np.repeat(np.arange(n_way), n_query)

    if metric == "cosine":
        proto_norm = F.normalize(prototypes, dim=-1)
        query_norm = F.normalize(query, dim=-1)
        scores = query_norm @ proto_norm.t()
        pred = scores.argmax(dim=1).numpy()
    else:
        dists = ((query.unsqueeze(1) - prototypes.unsqueeze(0)) ** 2).sum(dim=-1)
        pred = (-dists).argmax(dim=1).numpy()
    return float((pred == labels).mean() * 100.0)


def gnn_eval(model, support_128, query_128, n_way, n_support, n_query, device):
    support = torch.from_numpy(support_128).float().view(n_way, n_support, 128)
    query = torch.from_numpy(query_128).float().view(n_way, n_query, 128)
    z = torch.cat([support, query], dim=1).to(device)

    z_stack = [
        torch.cat([z[:, :n_support], z[:, n_support + i : n_support + i + 1]], dim=1).view(1, -1, 128)
        for i in range(n_query)
    ]

    with torch.no_grad():
        model.n_query = n_query
        scores = model.forward_gnn(z_stack).float().cpu()

    pred = scores.argmax(dim=1).numpy()
    labels = np.repeat(np.arange(n_way), n_query)
    return float((pred == labels).mean() * 100.0)


def evaluate(model, loader, n_way, n_shot, n_query, device, n_iter=600, proto_metric="l2"):
    results = {name: {"protonet": [], "gnn": []} for name in SUBSPACE_KEYS}
    model.n_support = n_shot
    completed_tasks = 0

    for task_idx, (x, _) in enumerate(loader):
        if task_idx >= n_iter:
            break
        completed_tasks += 1

        support_128, query_128 = extract_fc_features(model, x, device)

        for name in SUBSPACE_KEYS:
            support_proj, query_proj, transformer = fit_and_transform(name, support_128, query_128)
            proto_acc = protonet_eval(
                support_proj,
                query_proj,
                n_way,
                n_shot,
                n_query,
                metric=proto_metric,
            )

            support_rec = reconstruct_128(name, support_proj, transformer, support_128)
            query_rec = reconstruct_128(name, query_proj, transformer, query_128)
            support_rec = (
                support_rec
                / (np.linalg.norm(support_rec, axis=1, keepdims=True) + 1e-8)
            ).astype(np.float32)
            query_rec = (
                query_rec
                / (np.linalg.norm(query_rec, axis=1, keepdims=True) + 1e-8)
            ).astype(np.float32)

            gnn_acc = gnn_eval(model, support_rec, query_rec, n_way, n_shot, n_query, device)
            results[name]["protonet"].append(proto_acc)
            results[name]["gnn"].append(gnn_acc)

        if (task_idx + 1) % 100 == 0:
            print(f"  [{task_idx + 1}/{n_iter}] running...")

    if completed_tasks == 0:
        raise RuntimeError("no evaluation episodes were executed; check target split and episodic loader")

    summary = {}
    for name in SUBSPACE_KEYS:
        for head in ("protonet", "gnn"):
            accs = np.asarray(results[name][head], dtype=np.float32)
            mean = float(accs.mean())
            if len(accs) > 1:
                ci = float(1.96 * accs.std(ddof=1) / np.sqrt(len(accs)))
            else:
                ci = 0.0
            summary[f"{name}_{head}"] = (mean, ci)
    return summary


def build_loader(eval_dataset, data_dir, n_way, n_shot, n_query, n_episode, image_size=224):
    split_file = resolve_dataset_split_file(eval_dataset, split="novel", data_root=data_dir)
    if split_file is None or not os.path.isfile(split_file):
        raise FileNotFoundError(f"novel split not found for evaluation dataset: {eval_dataset}")

    with open(split_file, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    labels = meta.get("image_labels", [])
    n_classes = len(set(int(label) for label in labels))
    if n_classes < n_way:
        raise ValueError(
            f"{eval_dataset} novel split only has {n_classes} classes, but n_way={n_way}. "
            f"split_file={split_file}"
        )

    datamgr = SetDataManager(
        image_size,
        n_query=n_query,
        n_way=n_way,
        n_support=n_shot,
        n_eposide=n_episode,
    )
    loader = datamgr.get_data_loader(split_file, aug=False)
    return loader, split_file, n_classes


def parse_args():
    parser = argparse.ArgumentParser(description="Subspace ablation for ProtoNet and GNN")
    parser.add_argument("--source_dataset", type=str, required=True)
    parser.add_argument("--target_dataset", type=str, default=None)
    parser.add_argument("--n_shot", type=int, default=1)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=15)
    parser.add_argument("--n_iter", type=int, default=600)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing the dataset folders")
    parser.add_argument("--model", type=str, default="ResNet10")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proto_metric", type=str, default="l2", choices=["l2", "cosine"])
    args = parser.parse_args()
    args.data_dir = absolute_path(args.data_dir)
    args.checkpoint_dir = absolute_path(args.checkpoint_dir)
    if args.output_dir is not None:
        args.output_dir = absolute_path(args.output_dir)
    return args


def save_outputs(summary, args):
    eval_dataset = args.target_dataset or args.source_dataset
    txt_path = os.path.join(args.output_dir, f"kpca_{eval_dataset}_{args.n_shot}shot.txt")
    csv_path = os.path.join(args.output_dir, f"kpca_{eval_dataset}_{args.n_shot}shot.csv")

    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("Subspace comparison results\n")
        handle.write(
            f"source_dataset={args.source_dataset}, target_dataset={eval_dataset}, "
            f"n_shot={args.n_shot}, n_iter={args.n_iter}, "
            f"proto_metric={args.proto_metric}\n\n"
        )
        handle.write(f"{'Subspace':<16} {'ProtoNet+L2':>18} {'GNN Calibrator':>18}\n")
        handle.write("-" * 56 + "\n")
        for key in SUBSPACE_KEYS:
            proto_mean, proto_ci = summary[f"{key}_protonet"]
            gnn_mean, gnn_ci = summary[f"{key}_gnn"]
            handle.write(
                f"{SUBSPACE_LABELS[key]:<16} "
                f"{proto_mean:7.2f} +/- {proto_ci:4.2f}   "
                f"{gnn_mean:7.2f} +/- {gnn_ci:4.2f}\n"
            )

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "n_shot",
                "subspace_key",
                "subspace_label",
                "proto_metric",
                "protonet_mean",
                "protonet_ci",
                "gnn_mean",
                "gnn_ci",
            ],
        )
        writer.writeheader()
        for key in SUBSPACE_KEYS:
            proto_mean, proto_ci = summary[f"{key}_protonet"]
            gnn_mean, gnn_ci = summary[f"{key}_gnn"]
            writer.writerow(
                {
                    "dataset": eval_dataset,
                    "n_shot": args.n_shot,
                    "subspace_key": key,
                    "subspace_label": SUBSPACE_LABELS[key],
                    "proto_metric": args.proto_metric,
                    "protonet_mean": f"{proto_mean:.6f}",
                    "protonet_ci": f"{proto_ci:.6f}",
                    "gnn_mean": f"{gnn_mean:.6f}",
                    "gnn_ci": f"{gnn_ci:.6f}",
                }
            )

    return txt_path, csv_path


def main():
    args = parse_args()
    eval_dataset = args.target_dataset or args.source_dataset

    missing_paths = validate_dataset_root(
        args.data_dir,
        [eval_dataset],
        required_splits=("novel",),
    )
    if missing_paths:
        raise FileNotFoundError(
            "Dataset files were not found under --data_dir:\n" + missing_paths
        )

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.checkpoint_dir), "kpca_ablation")
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Subspace comparison")
    print(f"source:       {args.source_dataset}")
    print(f"target:       {eval_dataset}")
    print(f"n_shot:       {args.n_shot}")
    print(f"n_way:        {args.n_way}")
    print(f"n_iter:       {args.n_iter}")
    print(f"proto_metric: {args.proto_metric}")
    print(f"checkpoint:   {args.checkpoint_dir}")
    print("=" * 60)

    print("\n[1/3] loading model...")
    model = build_model(args, device)

    print("[2/3] building episodic loader...")
    loader, split_file, n_classes = build_loader(
        eval_dataset,
        args.data_dir,
        args.n_way,
        args.n_shot,
        args.n_query,
        args.n_iter,
    )
    print(f"      split_file: {split_file}")
    print(f"      novel_classes: {n_classes}")

    print(f"[3/3] evaluating {args.n_iter} episodes...")
    summary = evaluate(
        model,
        loader,
        args.n_way,
        args.n_shot,
        args.n_query,
        device,
        n_iter=args.n_iter,
        proto_metric=args.proto_metric,
    )

    print(f"\n{'=' * 60}")
    print(f"{'Subspace':<16} {'ProtoNet+L2':>18} {'GNN Calibrator':>18}")
    print("-" * 56)
    for key in SUBSPACE_KEYS:
        proto_mean, proto_ci = summary[f"{key}_protonet"]
        gnn_mean, gnn_ci = summary[f"{key}_gnn"]
        print(
            f"{SUBSPACE_LABELS[key]:<16} "
            f"{proto_mean:7.2f} +/- {proto_ci:4.2f}   "
            f"{gnn_mean:7.2f} +/- {gnn_ci:4.2f}"
        )
    print("=" * 60)

    txt_path, csv_path = save_outputs(summary, args)
    print(f"[saved] {txt_path}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
