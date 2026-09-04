from typing import List, Optional
from config_datasets import get_bscdfsl_candidates
import json
import os
import random


def parse_target_datasets(source: str, target_str: str) -> List[str]:
  """
  Parse target dataset selection string.

  - 'none' -> []
  - 'auto' -> all BSCDFSL-enabled datasets except source
  - 'A,B,C' -> explicit list (whitespace allowed)
  """
  if target_str is None:
    return []
  target_str = target_str.strip()
  if target_str.lower() == "none":
    return []
  if target_str.lower() == "auto":
    candidates = get_bscdfsl_candidates()
    return [d for d in candidates if d != source]
  return [x.strip() for x in target_str.split(",") if x.strip()]


def dkd_split_indices(all_indices, unlabeled_ratio: float = 0.2, seed: int = 0):
  """Split indices into unlabeled/eval parts following DKD (20%/80% by default)."""
  rng = random.Random(seed)
  indices = list(all_indices)
  rng.shuffle(indices)
  n_total = len(indices)
  n_unl = int(round(unlabeled_ratio * n_total))
  unlabeled = indices[:n_unl]
  eval_set = indices[n_unl:]
  return unlabeled, eval_set


def save_dkd_split(json_path: str, unlabeled, eval_set):
  os.makedirs(os.path.dirname(json_path), exist_ok=True)
  payload = {"unlabeled": unlabeled, "eval": eval_set}
  with open(json_path, "w") as handle:
    json.dump(payload, handle, indent=2)


def load_dkd_split(json_path: str):
  with open(json_path, "r") as handle:
    obj = json.load(handle)
  return obj["unlabeled"], obj["eval"]


def get_dkd_split_path(dataset_name: str, root_dir: str) -> str:
  # Store alongside the dataset root to avoid adding extra class-like folders that break ImageFolder.
  return os.path.join(root_dir, f"{dataset_name}_dkd_split.json")

def get_class_text_prompts(dataset_name: str, class_ids, data_root: Optional[str] = None, category_path: Optional[str] = None) -> List[str]:
  """
  Return a list of text prompts (one per class id) for the given dataset.
  Falls back to synthetic placeholders if class names are unavailable.
  """
  try:
    from methods.text_prompt import load_class_names, default_template  # lazy import to avoid cycles
  except Exception:
    load_class_names = None
    default_template = None

  class_ids = list(class_ids)
  names: List[str] = []
  if load_class_names is not None:
    names = load_class_names(dataset_name, data_root=data_root, category_path=category_path)

  def _fallback_template():
    if default_template is not None:
      return default_template(dataset_name)
    return "a photo of a {name}"

  tmpl = default_template(dataset_name) if default_template is not None else _fallback_template()
  if names:
    prompts = []
    for cid in class_ids:
      idx = int(cid) if int(cid) < len(names) else int(cid) % len(names)
      cname = names[idx]
      prompts.append(tmpl.format(name=cname))
    return prompts

  # fallback prompts when no names are known
  return [tmpl.format(name=f"class {cid}") for cid in class_ids]


__all__ = [
  "parse_target_datasets",
  "dkd_split_indices",
  "save_dkd_split",
  "load_dkd_split",
  "get_dkd_split_path",
  "get_class_text_prompts",
]
