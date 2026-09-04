import os

# ================================================================
# Base paths
# ================================================================
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")

# Canonical dataset roots (edit these to match your local paths)
DATASET_ROOT = {
    "NWPU": os.path.join(DEFAULT_DATASETS_DIR, "NWPU"),
    "AID": os.path.join(DEFAULT_DATASETS_DIR, "AID"),
    "UCM": os.path.join(DEFAULT_DATASETS_DIR, "UCM"),
    "EuroSAT": os.path.join(DEFAULT_DATASETS_DIR, "EuroSAT"),
}

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

for path in (CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR):
    os.makedirs(path, exist_ok=True)


# ================================================================
# Utilities
# ================================================================
def _first_existing(paths, *, expect_file=False):
    """Return the first existing path (file or directory) in ``paths``."""
    for candidate in paths:
        if not candidate:
            continue
        if expect_file and os.path.isfile(candidate):
            return os.path.abspath(candidate)
        if not expect_file and os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return None


def get_num_classes(path):
    """Return the number of immediate subdirectories in ``path``."""
    if not os.path.isdir(path):
        return 0
    return sum(
        os.path.isdir(os.path.join(path, name)) for name in os.listdir(path)
    )


# ================================================================
# Dataset paths (only the four remote-sensing datasets)
# ================================================================
def _resolve_image_root(root):
    """Resolve the directory that actually holds the class folders."""
    candidates = [
        os.path.join(root, "images"),
        os.path.join(root, "JPEGImages"),
        os.path.join(root, "2750"),
        root,
    ]
    return _first_existing(candidates) or root


def _build_dataset_entry(name, root):
    images = _resolve_image_root(root)
    splits = {
        split: os.path.join(root, f"{split}.json") for split in ("base", "val", "novel")
    }
    return {
        "root": root,
        "images": images,
        "splits": splits,
        "num_classes": get_num_classes(images),
    }


DATASETS = {name: _build_dataset_entry(name, root) for name, root in DATASET_ROOT.items()}


__all__ = [
    "ROOT_DIR",
    "DEFAULT_DATASETS_DIR",
    "DATASET_ROOT",
    "CHECKPOINT_DIR",
    "LOG_DIR",
    "RESULTS_DIR",
    "DATASETS",
    "get_num_classes",
]
