#!/usr/bin/env python3
"""Aggregate per-dataset subspace comparison results."""

import argparse
import csv
import os
import re

import numpy as np


DEFAULT_DATASETS = ["NWPU", "EuroSAT", "AID", "UCM"]
SUBSPACE_ORDER = [
    "No Subspace",
    "Linear",
    "Gaussian",
    "Polynomial",
    "Nonlinear",
]

OLD_LINE_RE = re.compile(
    r"^(PCA|KPCA-Linear|KPCA-RBF|KPCA-Poly).*?([0-9]+\.[0-9]+)\s*[\+\-\/ ]+.*?([0-9]+\.[0-9]+)"
)
OLD_METHOD_MAP = {
    "PCA": "Linear",
    "KPCA-Linear": "Linear",
    "KPCA-RBF": "Gaussian",
    "KPCA-Poly": "Polynomial",
}


def parse_csv_file(path):
    parsed = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row["subspace_label"]
            parsed[label] = {
                "protonet": float(row["protonet_mean"]),
                "gnn": float(row["gnn_mean"]),
            }
    return parsed


def parse_old_txt_file(path):
    parsed = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            matched = OLD_LINE_RE.match(line)
            if not matched:
                continue
            label = OLD_METHOD_MAP[matched.group(1)]
            parsed[label] = {
                "protonet": float(matched.group(2)),
                "gnn": float(matched.group(3)),
            }
    return parsed


def load_dataset_result(input_dir, dataset, n_shot):
    csv_path = os.path.join(input_dir, f"kpca_{dataset}_{n_shot}shot.csv")
    txt_path = os.path.join(input_dir, f"kpca_{dataset}_{n_shot}shot.txt")

    if os.path.isfile(csv_path):
        return parse_csv_file(csv_path)
    if os.path.isfile(txt_path):
        return parse_old_txt_file(txt_path)
    raise FileNotFoundError(f"missing result file for {dataset}, {n_shot}-shot")


def main():
    parser = argparse.ArgumentParser(description="Aggregate subspace comparison results")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--n_shot", type=int, required=True, choices=[1, 5])
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if len(datasets) < 2:
        raise ValueError("need at least two datasets for aggregation")

    if args.output_file is None:
        args.output_file = os.path.join(args.input_dir, f"kpca_average_{args.n_shot}shot.txt")

    by_dataset = {dataset: load_dataset_result(args.input_dir, dataset, args.n_shot) for dataset in datasets}

    summary = {}
    for label in SUBSPACE_ORDER:
        proto_vals = [by_dataset[dataset][label]["protonet"] for dataset in datasets if label in by_dataset[dataset]]
        gnn_vals = [by_dataset[dataset][label]["gnn"] for dataset in datasets if label in by_dataset[dataset]]
        if not proto_vals or not gnn_vals:
            continue
        summary[label] = {
            "protonet_mean": float(np.mean(proto_vals)),
            "protonet_std": float(np.std(proto_vals, ddof=1)) if len(proto_vals) > 1 else 0.0,
            "gnn_mean": float(np.mean(gnn_vals)),
            "gnn_std": float(np.std(gnn_vals, ddof=1)) if len(gnn_vals) > 1 else 0.0,
        }

    lines = []
    lines.append(f"Subspace comparison average ({args.n_shot}-shot)")
    lines.append("Datasets: " + ", ".join(datasets))
    lines.append("")
    lines.append(f"{'Subspace':<16} {'ProtoNet+L2(avg)':>20} {'GNN Calibrator(avg)':>22}")
    lines.append("-" * 62)
    for label in SUBSPACE_ORDER:
        if label not in summary:
            continue
        row = summary[label]
        lines.append(
            f"{label:<16} "
            f"{row['protonet_mean']:7.2f} +/- {row['protonet_std']:4.2f}   "
            f"{row['gnn_mean']:7.2f} +/- {row['gnn_std']:4.2f}"
        )
    text = "\n".join(lines) + "\n"
    print(text)

    with open(args.output_file, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"saved average summary to: {args.output_file}")


if __name__ == "__main__":
    main()
