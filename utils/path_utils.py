"""Portable path helpers used by dataset loaders and command-line entrypoints.

Dataset split JSON files in older experiments may contain absolute paths from
the machine that created the split.  The training code should not depend on
those paths: when ``data_root`` is supplied, this module remaps such entries
to ``<data_root>/<dataset>/<relative path>`` whenever the original file is
not available.
"""

import os
import re
from pathlib import Path


def repository_root():
  """Return the repository root, independent of the current working directory."""
  return Path(__file__).resolve().parents[1]


def absolute_path(value, base_dir=None):
  """Expand a user path and return an absolute normalized string."""
  if value is None or str(value).strip() == "":
    return ""
  path = Path(os.path.expandvars(os.path.expanduser(str(value))))
  if not path.is_absolute():
    path = (Path(base_dir) if base_dir is not None else Path.cwd()) / path
  return str(path.resolve())


def _path_parts(value):
  """Split both Windows and POSIX path spellings, even on Linux."""
  return [part for part in re.split(r"[/\\]+", str(value)) if part not in ("", ".")]


def _candidate_from_dataset_suffix(raw_path, data_root, dataset_name):
  if not data_root or not dataset_name:
    return None
  parts = _path_parts(raw_path)
  dataset_lower = str(dataset_name).lower()
  for index, part in enumerate(parts):
    if part.lower() == dataset_lower:
      suffix = parts[index + 1:]
      candidate = Path(data_root) / str(dataset_name)
      if suffix:
        candidate = candidate.joinpath(*suffix)
      return candidate
  return None


def resolve_image_path(image_path, data_root=None, dataset_name=None):
  """Resolve an image path from a split JSON file.

  The original path is preferred when it exists.  Otherwise, a supplied data
  root is used to reconstruct paths from either a dataset-relative entry or an
  old absolute entry that contains the dataset directory name.
  """
  raw = os.path.expandvars(os.path.expanduser(str(image_path))).strip()
  if not raw:
    raise FileNotFoundError("Empty image path in dataset metadata")

  candidates = [Path(raw)]
  if data_root:
    root = Path(data_root)
    suffix_candidate = _candidate_from_dataset_suffix(raw, root, dataset_name)
    if suffix_candidate is not None:
      candidates.append(suffix_candidate)

    raw_parts = _path_parts(raw)
    if not Path(raw).is_absolute():
      candidates.append(root / raw)
      if dataset_name:
        candidates.append(root / str(dataset_name) / raw)

    # Historical metadata used DATACENTER/4 as a mount-point placeholder.
    # Keep compatibility, but still anchor the result at the caller's root.
    marker_parts = [part.lower() for part in raw_parts]
    if "datacenter" in marker_parts and dataset_name:
      candidates.append(root / str(dataset_name) / Path(*raw_parts[marker_parts.index("datacenter") + 2:]))

  seen = set()
  for candidate in candidates:
    normalized = Path(str(candidate)).expanduser()
    key = os.path.normcase(os.path.normpath(str(normalized)))
    if key in seen:
      continue
    seen.add(key)
    if normalized.is_file():
      return str(normalized.resolve())

  attempted = "\n  ".join(str(candidate) for candidate in candidates)
  raise FileNotFoundError(
    "Image referenced by dataset metadata was not found. "
    "Check --data_dir and the dataset layout. Tried:\n  " + attempted
  )


def dataset_context(data_file, data_root=None, dataset_name=None):
  """Infer portable dataset context from a JSON file path when possible."""
  inferred_root = data_root
  inferred_name = dataset_name
  if isinstance(data_file, (str, os.PathLike)):
    file_path = Path(data_file).resolve()
    if inferred_name is None:
      inferred_name = file_path.parent.name
    if inferred_root is None and file_path.parent.parent != file_path.parent:
      inferred_root = str(file_path.parent.parent)
  return inferred_root, inferred_name


def validate_dataset_root(data_root, dataset_names, required_splits=()):
  """Return a clear error string for missing dataset folders or split files."""
  missing = []
  root = Path(data_root)
  for dataset_name in dataset_names:
    if not dataset_name:
      continue
    dataset_dir = root / str(dataset_name)
    if not dataset_dir.is_dir():
      missing.append(str(dataset_dir))
      continue
    for split in required_splits:
      split_path = dataset_dir / f"{split}.json"
      if not split_path.is_file():
        missing.append(str(split_path))
  if missing:
    return "\n".join(missing)
  return ""
