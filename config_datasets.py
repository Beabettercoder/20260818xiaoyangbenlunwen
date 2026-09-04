"""
Central dataset registry used across training and cross-domain benchmarks.

Register each dataset once here to expose:
  - dataset root and image directory
  - loader module name under data/
  - whether it participates in BSCDFSL benchmarks
"""

import os
from typing import Dict, List

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")


def _resolve_image_root(root: str) -> str:
    candidates = [
        os.path.join(root, "images"),
        os.path.join(root, "JPEGImages"),
        os.path.join(root, "2750"),
        root,
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return root


def _build_entry(root: str, loader: str, dtype: str = "remote_sensing", enable_bscdfsl: bool = True, dkd_split_mode: str = None) -> Dict:
    return {
        "root": root,
        "images": _resolve_image_root(root),
        "splits": {split: os.path.join(root, f"{split}.json") for split in ("base", "val", "novel")},
        "type": dtype,
        "loader": loader,
        "enable_bscdfsl": enable_bscdfsl,
        "dkd_split_mode": dkd_split_mode,
    }


DATASET_REGISTRY: Dict[str, Dict] = {
    "NWPU": _build_entry(os.path.join(DEFAULT_DATASETS_DIR, "NWPU"), "NWPU_few_shot"),
    "AID": _build_entry(os.path.join(DEFAULT_DATASETS_DIR, "AID"), "AID_few_shot"),
    "UCM": _build_entry(os.path.join(DEFAULT_DATASETS_DIR, "UCM"), "UCM_few_shot"),
    "EuroSAT": _build_entry(os.path.join(DEFAULT_DATASETS_DIR, "EuroSAT"), "EuroSAT_few_shot"),
    # SAR dataset placeholder with DKD angle split (17° unlabeled / 15° eval)
    "MSTAR": _build_entry(os.path.join(DEFAULT_DATASETS_DIR, "MSTAR"), "MSTAR_few_shot", dtype="sar", enable_bscdfsl=True, dkd_split_mode="angle"),
    # Example (disabled) legacy datasets can be re-enabled by adding entries here:
    # "miniImageNet": _build_entry("/path/to/miniImageNet", "MiniImageNet_few_shot", dtype="natural", enable_bscdfsl=False),
}


def get_all_datasets() -> List[str]:
    return list(DATASET_REGISTRY.keys())


def get_bscdfsl_candidates() -> List[str]:
    return [k for k, v in DATASET_REGISTRY.items() if v.get("enable_bscdfsl", False)]


def get_dataset_config(name: str) -> Dict:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' is not registered. Please add it to DATASET_REGISTRY.")
    return DATASET_REGISTRY[name]


def update_dataset_root(name: str, root: str):
    """Override dataset root (e.g., when passing --data_dir)."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' is not registered.")
    cfg = DATASET_REGISTRY[name]
    DATASET_REGISTRY[name] = _build_entry(root, cfg["loader"], dtype=cfg.get("type", "remote_sensing"), enable_bscdfsl=cfg.get("enable_bscdfsl", False))


__all__ = [
    "DATASET_REGISTRY",
    "get_all_datasets",
    "get_bscdfsl_candidates",
    "get_dataset_config",
    "update_dataset_root",
    "DEFAULT_DATASETS_DIR",
]
