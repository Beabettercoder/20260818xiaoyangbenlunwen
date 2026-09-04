#!/usr/bin/env python3
"""
Generate DKD-style unlabeled/eval splits (20%/80% by default) for registered datasets.

The script scans the chosen dataset(s), builds a flat list of samples from the provided split(s),
then writes a JSON under <root>/splits_dkd/<dataset>_dkd_split.json:
{
  "unlabeled": [...indices...],
  "eval": [...]
}

Usage:
  python tools/generate_dkd_splits.py --datasets auto --splits novel --unlabeled_ratio 0.2
  python tools/generate_dkd_splits.py --datasets NWPU,AID --data_root ./datasets --splits all --seed 1
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure project root on sys.path when run from tools/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from config_datasets import DATASET_REGISTRY, get_dataset_config, update_dataset_root
from data.datamgr import load_dataset_meta
from utils.dataset_utils import dkd_split_indices, save_dkd_split, get_dkd_split_path


def _parse_datasets(arg: str):
  if arg.lower() == "auto":
    return [k for k, v in DATASET_REGISTRY.items() if v.get("enable_bscdfsl", False)]
  return [x.strip() for x in arg.split(",") if x.strip()]


def _parse_splits(arg: str):
  if arg.lower() == "all":
    return ("base", "val", "novel")
  return tuple([x.strip() for x in arg.split(",") if x.strip()])


def main():
  parser = argparse.ArgumentParser(description="Generate DKD splits (unlabeled/eval index lists).")
  parser.add_argument("--datasets", default="auto", help="auto = all enable_bscdfsl datasets; or comma list")
  parser.add_argument("--data_root", default=None, help="Override dataset root (expects <data_root>/<name>/...)")
  parser.add_argument("--splits", default="novel", help="Splits to pool samples from: base,val,novel,all")
  parser.add_argument("--unlabeled_ratio", type=float, default=0.2, help="Portion of samples to mark as unlabeled")
  parser.add_argument("--seed", type=int, default=0, help="Random seed for splitting")
  args = parser.parse_args()

  targets = _parse_datasets(args.datasets)
  splits = _parse_splits(args.splits)

  if not targets:
    raise SystemExit("No target datasets resolved. Check --datasets.")

  for name in targets:
    try:
      if args.data_root:
        update_dataset_root(name, os.path.join(args.data_root, name))
      cfg = get_dataset_config(name)
      meta, found = load_dataset_meta(name, splits=splits, data_root=args.data_root)
      all_indices = list(range(len(meta["image_names"])))
      unlabeled, eval_set = dkd_split_indices(all_indices, unlabeled_ratio=args.unlabeled_ratio, seed=args.seed)
      split_path = get_dkd_split_path(name, cfg["root"])
      save_dkd_split(split_path, unlabeled, eval_set)
      print(
        f"[DKD SPLIT] {name}: total={len(all_indices)}, unlabeled={len(unlabeled)}, eval={len(eval_set)}, "
        f"found_splits={found}, saved={split_path}"
      )
    except Exception as exc:
      print(f"[WARN] Failed to build DKD split for {name}: {exc}")


if __name__ == "__main__":
  main()
