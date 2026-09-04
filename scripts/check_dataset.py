#!/usr/bin/env python3
"""
Quick sanity check: verify a registered dataset can be loaded (no training).

Usage:
  python scripts/check_dataset.py --dataset NWPU
  python scripts/check_dataset.py --dataset AID --n_way 5 --n_support 1 --n_query 1
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path so config_datasets/data modules resolve when run from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from config_datasets import get_all_datasets
from data.datamgr import get_few_shot_datamgr


def parse_args():
  parser = argparse.ArgumentParser(description="Dataset loading sanity check")
  parser.add_argument(
    "--dataset",
    required=True,
    choices=get_all_datasets(),
    help="Dataset name registered in config_datasets.py",
  )
  parser.add_argument("--image_size", type=int, default=224, help="Input image size")
  parser.add_argument("--batch_size", type=int, default=4, help="Batch size for simple loader")
  parser.add_argument("--n_way", type=int, default=5, help="Way for episodic loader")
  parser.add_argument("--n_support", type=int, default=1, help="Support per class")
  parser.add_argument("--n_query", type=int, default=1, help="Query per class")
  parser.add_argument("--n_episode", type=int, default=2, help="Episodes to sample")
  parser.add_argument(
    "--data_root",
    type=str,
    default=None,
    help="Root directory containing datasets/<name>; overrides default registry path",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  print(f"[check] dataset={args.dataset}")

  # Simple loader sanity check
  try:
    simple = get_few_shot_datamgr(
      args.dataset,
      episodic=False,
      image_size=args.image_size,
      batch_size=args.batch_size,
      data_root=args.data_root,
    )
    simple_loader = simple.get_data_loader(aug=False)
    imgs, labels = next(iter(simple_loader))
    print(f"[simple] ok: imgs={tuple(imgs.shape)}, labels={tuple(labels.shape)}")
  except Exception as exc:
    print(f"[simple] failed: {exc}")
    sys.exit(1)

  # Episodic loader sanity check
  try:
    episodic = get_few_shot_datamgr(
      args.dataset,
      episodic=True,
      image_size=args.image_size,
      n_way=args.n_way,
      n_support=args.n_support,
      n_query=args.n_query,
      n_eposide=args.n_episode,
      data_root=args.data_root,
    )
    epi_loader = episodic.get_data_loader(aug=False)
    for i, (x, y) in enumerate(epi_loader):
      print(f"[episode {i}] x={tuple(x.shape)}, y={tuple(y.shape)}")
      if i + 1 >= args.n_episode:
        break
  except Exception as exc:
    print(f"[episodic] failed: {exc}")
    sys.exit(1)

  print("[check] success: dataset can be loaded.")


if __name__ == "__main__":
  main()
