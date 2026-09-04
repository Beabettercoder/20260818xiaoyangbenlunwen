import json
import os
import re
from pathlib import Path
from typing import List, Optional


def canonicalize_class_name(name: str) -> str:
  text = str(name).strip()
  if not text:
    return text
  text = text.replace("_", " ").replace("-", " ")
  text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
  text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
  text = re.sub(r"(?<=\D)(\d)", r" \1", text)
  text = re.sub(r"(\d)(?=\D)", r"\1 ", text)
  text = re.sub(r"\s+", " ", text).strip().lower()
  return text


def _candidate_dataset_dirs(dataset_name: Optional[str], data_root: Optional[str]) -> List[Path]:
  candidates: List[Path] = []
  if data_root and dataset_name:
    candidates.append(Path(data_root) / dataset_name)
  if dataset_name:
    repo_data_dir = Path(__file__).resolve().parents[1] / "data" / dataset_name
    candidates.append(repo_data_dir)
  deduped: List[Path] = []
  seen = set()
  for path in candidates:
    key = str(path.resolve()) if path.exists() else str(path)
    if key in seen:
      continue
    seen.add(key)
    deduped.append(path)
  return deduped


def _infer_class_names_from_dirs(dataset_dirs: List[Path]) -> List[str]:
  image_root_candidates = ("images", "JPEGImages", "2750", "")
  for dataset_dir in dataset_dirs:
    for suffix in image_root_candidates:
      image_root = dataset_dir / suffix if suffix else dataset_dir
      if not image_root.is_dir():
        continue
      class_dirs = sorted([p for p in image_root.iterdir() if p.is_dir()])
      if class_dirs:
        return [canonicalize_class_name(p.name) for p in class_dirs]
  return []


def load_class_names(dataset_name: Optional[str] = None, data_root: Optional[str] = None, category_path: Optional[str] = None) -> List[str]:
  """Load class names from category files if present."""
  candidates: List[Path] = []
  if category_path:
    candidates.append(Path(category_path))
  dataset_dirs = _candidate_dataset_dirs(dataset_name, data_root)
  for base in dataset_dirs:
    candidates.extend([
      base / "category.txt",
      base / "categories.txt",
      base / "category.json",
      base / "categories.json",
    ])
  for path in candidates:
    if not path.exists():
      continue
    if path.suffix.lower() == ".json":
      with path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
      for key in ["class_names", "label_names", "categories", "label2name"]:
        if key not in meta:
          continue
        values = meta[key]
        if isinstance(values, dict):
          values = [v for _, v in sorted(values.items(), key=lambda kv: int(kv[0]))]
        return [canonicalize_class_name(str(v)) for v in values]
      continue
    names: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
      for line in handle:
        line = line.strip()
        if not line:
          continue
        parts = line.split()
        if parts[0].isdigit():
          names.append(canonicalize_class_name(" ".join(parts[1:])))
        else:
          names.append(canonicalize_class_name(line))
    if names:
      return names
  inferred = _infer_class_names_from_dirs(dataset_dirs)
  if inferred:
    return inferred
  return []


def default_template(dataset_name: Optional[str]) -> str:
  """Dataset-aware text template focused on AID/NWPU/EuroSAT/UCM with a generic fallback."""
  name = (dataset_name or "").lower()
  if "euro" in name or "sat" in name:
    return "a satellite image of {name} land cover"
  if "aid" in name:
    return "a high-resolution aerial image of {name} scene"
  if "nwpu" in name or "remote" in name:
    return "a remote sensing image of {name} scene"
  if "ucm" in name:
    return "an aerial photograph of a {name} region"
  return "a photo of a {name}"


# ============ Step1: PromptEnsemble - 多模板集成 ============
# 参考 CLIP 零样本分类和 DP-RSCap 的多模板策略

ENSEMBLE_TEMPLATES = {
  "nwpu": [
    "a satellite image of {name}",
    "a remote sensing photo of {name}",
    "an aerial view of {name}",
    "a satellite photograph of {name}",
    "overhead imagery showing {name}",
    "a {name} scene captured from satellite",
    "remote sensing data of {name} area",
    "an orthophoto of {name} region",
  ],
  "eurosat": [
    "a satellite image of {name}",
    "an overhead view of {name} land cover",
    "a remote sensing image of {name}",
    "a satellite photograph showing {name}",
    "aerial imagery of {name} terrain",
    "a {name} scene from satellite",
    "land cover classification: {name}",
    "sentinel-2 image of {name}",
  ],
  "aid": [
    "a high-resolution aerial image of {name}",
    "an aerial photograph of {name} scene",
    "overhead view of {name} area",
    "a remote sensing image of {name}",
    "aerial imagery showing {name}",
    "a {name} captured from aircraft",
    "high-altitude photo of {name}",
    "airborne image of {name} region",
  ],
  "ucm": [
    "an aerial photograph of {name}",
    "overhead imagery of {name} region",
    "a remote sensing photo of {name}",
    "urban scene showing {name}",
    "aerial view of {name} area",
    "a {name} from aerial perspective",
    "land use image of {name}",
    "UC Merced scene: {name}",
  ],
  "generic": [
    "a photo of {name}",
    "an image of {name}",
    "a picture showing {name}",
    "a {name} in the scene",
    "visual representation of {name}",
    "a photograph of {name}",
  ],
}


def get_ensemble_templates(dataset_name: Optional[str]) -> List[str]:
  """
  [Step1: PromptEnsemble]
  获取数据集对应的多模板列表，用于文本嵌入集成。
  
  多模板集成的原理:
    - 单一模板可能无法完整捕捉类别的语义多样性
    - 多模板平均可以获得更鲁棒的文本表示
    - 类似 CLIP 原论文中 ImageNet 零样本分类的做法
  """
  name = (dataset_name or "").lower()
  if "nwpu" in name or "remote" in name:
    return ENSEMBLE_TEMPLATES["nwpu"]
  if "euro" in name or "sat" in name:
    return ENSEMBLE_TEMPLATES["eurosat"]
  if "aid" in name:
    return ENSEMBLE_TEMPLATES["aid"]
  if "ucm" in name:
    return ENSEMBLE_TEMPLATES["ucm"]
  return ENSEMBLE_TEMPLATES["generic"]


__all__ = ["load_class_names", "default_template", "get_ensemble_templates", "ENSEMBLE_TEMPLATES", "canonicalize_class_name"]
