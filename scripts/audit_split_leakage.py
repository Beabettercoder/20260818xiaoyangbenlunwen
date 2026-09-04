#!/usr/bin/env python3
"""Audit split protocol and leakage for cross-domain few-shot fairness.

Protocol:
1. Source domain: base/val must each have at least n_way classes.
    novel must have at least n_way classes unless --allow_source_empty_novel is set.
2. Target domains: base/val must be empty, novel must have at least n_way classes.
3. Image overlap across base/val/novel must be zero.
4. Label overlap across base/val/novel should be zero.
"""

import argparse
import json
import os
import sys
from itertools import combinations

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SPLIT_SPEC_FILENAME = "split_spec.json"
SOURCE_IMAGE_DISJOINT_MODE = "image_disjoint_shared_classes"
SOURCE_CLASS_DISJOINT_MODE = "class_disjoint"
PROTOCOL_LEGACY = "legacy"
PROTOCOL_STRICT_PAIRWISE = "strict_pairwise"
PROTOCOL_DMPC = "dmpc"
PAIRWISE_PROTOCOL_NAMES = {PROTOCOL_STRICT_PAIRWISE, PROTOCOL_DMPC}
SOURCE_STRICT_PAIRWISE_MODE = "strict_pairwise_source_image_disjoint_all_classes"
TARGET_STRICT_PAIRWISE_MODE = "strict_pairwise_target_unlabeled_eval"
SOURCE_DMPC_MODE = "dmpc_source_image_disjoint_all_classes"
TARGET_DMPC_MODE = "dmpc_target_unlabeled_eval"
SOURCE_PAIRWISE_MODE_NAMES = {SOURCE_STRICT_PAIRWISE_MODE, SOURCE_DMPC_MODE}
TARGET_PAIRWISE_MODE_NAMES = {TARGET_STRICT_PAIRWISE_MODE, TARGET_DMPC_MODE}


def load_split(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = data.get("image_names", [])
    labels = data.get("image_labels", [])
    return set(names), set(labels), len(names), len(set(labels))


def load_split_spec(ds_dir):
    spec_path = os.path.join(ds_dir, SPLIT_SPEC_FILENAME)
    if not os.path.isfile(spec_path):
        return {}
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def audit_one_dataset(data_dir, dataset, n_way, is_source, allow_source_empty_novel=False):
    ds_dir = os.path.join(data_dir, dataset)
    split_spec = load_split_spec(ds_dir)
    protocol = split_spec.get("protocol", PROTOCOL_LEGACY)
    source_mode = split_spec.get("source_split_mode")
    target_mode = split_spec.get("target_split_mode")
    files = {
        "base": os.path.join(ds_dir, "base.json"),
        "val": os.path.join(ds_dir, "val.json"),
        "novel": os.path.join(ds_dir, "novel.json"),
    }
    extra_files = {}
    if protocol in PAIRWISE_PROTOCOL_NAMES and not is_source:
        extra_files = {
            "unlabeled": os.path.join(ds_dir, "unlabeled.json"),
            "eval": os.path.join(ds_dir, "eval.json"),
        }

    report = {
        "dataset": dataset,
        "ok": True,
        "messages": [],
    }

    for split, fp in {**files, **extra_files}.items():
        if not os.path.isfile(fp):
            report["ok"] = False
            report["messages"].append(f"[ERROR] missing file: {fp}")

    if not report["ok"]:
        return report

    split_data = {}
    for split, fp in {**files, **extra_files}.items():
        names, labels, n_img, n_cls = load_split(fp)
        split_data[split] = {
            "names": names,
            "labels": labels,
            "n_img": n_img,
            "n_cls": n_cls,
        }
        report["messages"].append(f"[INFO] {split}: images={n_img}, classes={n_cls}")
    report["messages"].append(f"[INFO] protocol={protocol}")
    if is_source and source_mode:
        report["messages"].append(f"[INFO] source_split_mode={source_mode}")
    if (not is_source) and target_mode:
        report["messages"].append(f"[INFO] target_split_mode={target_mode}")

    # protocol sanity
    if is_source:
        for split in ("base", "val"):
            if split_data[split]["n_cls"] < n_way:
                report["ok"] = False
                report["messages"].append(
                    f"[ERROR] source {split}.json has only {split_data[split]['n_cls']} classes (< n_way={n_way})"
                )

        source_novel_cls = split_data["novel"]["n_cls"]
        require_source_novel = source_mode in (SOURCE_CLASS_DISJOINT_MODE, SOURCE_IMAGE_DISJOINT_MODE)
        if protocol in PAIRWISE_PROTOCOL_NAMES or source_mode in SOURCE_PAIRWISE_MODE_NAMES:
            require_source_novel = False
        if require_source_novel or source_novel_cls > 0:
            if source_novel_cls < n_way:
                if allow_source_empty_novel and source_novel_cls == 0:
                    report["messages"].append(
                        "[OK] source novel.json is empty and allowed by --allow_source_empty_novel"
                    )
                else:
                    report["ok"] = False
                    report["messages"].append(
                        f"[ERROR] source novel.json has only {source_novel_cls} classes (< n_way={n_way})"
                    )
    else:
        if split_data["base"]["n_cls"] != 0:
            report["ok"] = False
            report["messages"].append(
                f"[ERROR] target base.json must be empty, got {split_data['base']['n_cls']} classes"
            )
        else:
            report["messages"].append("[OK] target base.json is empty")

        if split_data["val"]["n_cls"] != 0:
            report["ok"] = False
            report["messages"].append(
                f"[ERROR] target val.json must be empty, got {split_data['val']['n_cls']} classes"
            )
        else:
            report["messages"].append("[OK] target val.json is empty")

        if split_data["novel"]["n_cls"] < n_way:
            report["ok"] = False
            report["messages"].append(
                f"[ERROR] target novel.json has only {split_data['novel']['n_cls']} classes (< n_way={n_way})"
            )
        if protocol in PAIRWISE_PROTOCOL_NAMES or target_mode in TARGET_PAIRWISE_MODE_NAMES:
            if split_data["unlabeled"]["n_cls"] < n_way:
                report["ok"] = False
                report["messages"].append(
                    f"[ERROR] target unlabeled.json has only {split_data['unlabeled']['n_cls']} classes (< n_way={n_way})"
                )
            if split_data["eval"]["n_cls"] < n_way:
                report["ok"] = False
                report["messages"].append(
                    f"[ERROR] target eval.json has only {split_data['eval']['n_cls']} classes (< n_way={n_way})"
                )

    # overlap checks
    for a, b in combinations(("base", "val", "novel"), 2):
        img_overlap = split_data[a]["names"] & split_data[b]["names"]
        lbl_overlap = split_data[a]["labels"] & split_data[b]["labels"]

        if img_overlap:
            report["ok"] = False
            report["messages"].append(
                f"[ERROR] image leakage {a}<->{b}: {len(img_overlap)} overlapped images"
            )
        else:
            report["messages"].append(f"[OK] image overlap {a}<->{b}: 0")

        if lbl_overlap and is_source and source_mode in (SOURCE_IMAGE_DISJOINT_MODE, *SOURCE_PAIRWISE_MODE_NAMES):
            report["messages"].append(
                f"[OK] label overlap {a}<->{b}: {len(lbl_overlap)} (allowed by source_split_mode={source_mode})"
            )
        elif lbl_overlap:
            report["ok"] = False
            report["messages"].append(
                f"[ERROR] label overlap {a}<->{b}: {len(lbl_overlap)} overlapped class ids"
            )
        else:
            report["messages"].append(f"[OK] label overlap {a}<->{b}: 0")

    if protocol in PAIRWISE_PROTOCOL_NAMES and not is_source:
        dmpc_pairs = [("unlabeled", "novel"), ("unlabeled", "eval")]
        for a, b in dmpc_pairs:
            img_overlap = split_data[a]["names"] & split_data[b]["names"]
            lbl_overlap = split_data[a]["labels"] & split_data[b]["labels"]
            if img_overlap:
                report["ok"] = False
                report["messages"].append(
                    f"[ERROR] image leakage {a}<->{b}: {len(img_overlap)} overlapped images"
                )
            else:
                report["messages"].append(f"[OK] image overlap {a}<->{b}: 0")

            if lbl_overlap:
                report["messages"].append(
                    f"[OK] label overlap {a}<->{b}: {len(lbl_overlap)} (expected under strict pairwise target protocol)"
                )
            else:
                report["messages"].append(f"[OK] label overlap {a}<->{b}: 0")

    return report


def main():
    parser = argparse.ArgumentParser(description="Audit split leakage for multiple datasets")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="NWPU,EuroSAT,AID,UCM")
    parser.add_argument("--source", type=str, default="NWPU", help="source domain name")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument(
        "--allow_source_empty_novel",
        action="store_true",
        help="Allow source novel.json to have 0 classes (used when source novel testing is skipped)",
    )
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    all_ok = True
    print("=" * 72)
    print("Split Leakage Audit")
    print(f"data_dir={args.data_dir}")
    print(f"datasets={datasets}")
    print("=" * 72)

    for ds in datasets:
        is_source = (ds == args.source)
        rep = audit_one_dataset(
            args.data_dir,
            ds,
            args.n_way,
            is_source=is_source,
            allow_source_empty_novel=args.allow_source_empty_novel,
        )
        all_ok = all_ok and rep["ok"]
        print("\n" + "-" * 72)
        print(f"Dataset: {ds} ({'SOURCE' if is_source else 'TARGET'}) | {'PASS' if rep['ok'] else 'FAIL'}")
        print("-" * 72)
        for msg in rep["messages"]:
            print(msg)

    print("\n" + "=" * 72)
    print("FINAL:", "PASS" if all_ok else "FAIL")
    print("=" * 72)

    raise SystemExit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
