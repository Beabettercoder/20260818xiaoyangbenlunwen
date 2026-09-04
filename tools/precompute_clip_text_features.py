import argparse
import json
from pathlib import Path
from typing import List

import torch

try:
  import clip
except ImportError as exc:  # pragma: no cover - only triggered when dependency missing
  raise ImportError("The `clip` package is required. Install via `pip install git+https://github.com/openai/CLIP.git`.") from exc
from utils.progress_utils import progress_range


DEFAULT_TEMPLATES = {
  "generic": [
    "a photo of a {classname}",
    "a centered photo of the {classname}",
    "a close-up photo of {classname}",
    "an image of {classname}",
    "a detailed view of {classname}",
    "a faraway view of {classname}",
    "a natural image of {classname}",
    "a low angle view of {classname}",
    "a high angle view of {classname}",
  ],
  "nwpu": [
    "a satellite image of {classname}",
    "a remote sensing photo of {classname}",
    "aerial view of {classname}",
    "a satellite photo of a {classname}",
    "a remote sensing scene of {classname}",
    "an overhead view of {classname}",
    "a {classname} in a satellite image",
    "an aerial shot of {classname}",
  ],
  "aid": [
    "a satellite image of {classname}",
    "a remote sensing photo of {classname}",
    "an aerial view of {classname}",
    "a satellite picture of {classname}",
    "a remote sensing scene showing {classname}",
    "an overhead capture of {classname}",
  ],
  "ucm": [
    "a satellite image of {classname}",
    "a remote sensing photograph of {classname}",
    "an aerial view of {classname}",
    "an overhead satellite capture of {classname}",
    "a remote sensing scene of {classname}",
    "a {classname} scene shot from above",
  ],
  "eurosat": [
    "a satellite photo of {classname} scene",
    "an overhead view of {classname}",
    "a remote sensing image of {classname}",
    "a satellite view of {classname}",
    "an aerial view of {classname}",
    "a {classname} scene captured from satellite",
    "a remote sensing scene of {classname}",
    "a satellite picture of {classname}",
  ],
  "isic": [
    "a dermatoscopic photo of {classname}",
    "a skin lesion showing {classname}",
    "a clinical image of {classname}",
    "a close-up dermoscopy of {classname}",
    "a high-resolution image of {classname} lesion",
    "a macro image of {classname}",
  ],
  "cropdisease": [
    "a close-up photo of a plant leaf with {classname}",
    "a crop disease called {classname}",
    "a photo of a leaf showing {classname}",
    "a plant exhibiting {classname}",
    "a macro shot of {classname} symptoms",
    "an agricultural disease named {classname}",
  ],
  "chestx": [
    "a chest x-ray that shows {classname}",
    "a radiograph of chest with {classname}",
    "a medical x-ray indicating {classname}",
    "a frontal chest x-ray showing {classname}",
  ],
}


def parse_args():
  parser = argparse.ArgumentParser(description="Pre-compute CLIP text features for dataset classes.")
  parser.add_argument("--class_name_file", required=True, type=str, help="Text/JSON file listing dataset class names.")
  parser.add_argument("--json_key", default=None, type=str, help="Optional JSON key that explicitly stores the class names list.")
  parser.add_argument("--output", required=True, type=str, help="Path to save the resulting tensor (.pt).")
  parser.add_argument("--dataset", default="generic", type=str, help="Dataset nickname used to pick a default prompt template.")
  parser.add_argument("--dataset_key", default=None, type=str, help="Optional dict key name when saving multiple datasets in one file.")
  parser.add_argument("--prompt", default=None, type=str, help="Custom prompt template. Use {classname} as a placeholder.")
  parser.add_argument("--extra_templates", nargs="*", default=None, help="Additional templates appended to the defaults.")
  parser.add_argument("--clip_model", default="ViT-B/32", type=str, help="CLIP text encoder backbone to use.")
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Device for CLIP inference.")
  return parser.parse_args()


def read_class_names(path: Path, json_key: str = None) -> List[str]:
  if not path.exists():
    raise FileNotFoundError(f"Could not find class name file: {path}")
  if path.suffix.lower() == ".json":
    with path.open("r", encoding="utf-8") as handle:
      meta = json.load(handle)
    keys = [json_key] if json_key else ["class_names", "label_names", "categories", "label2name"]
    for key in keys:
      if not key:
        continue
      if key not in meta:
        continue
      values = meta[key]
      if isinstance(values, dict):
        sorted_items = sorted(values.items(), key=lambda kv: int(kv[0]))
        return [str(v) for _, v in sorted_items]
      if isinstance(values, list):
        return [str(v) for v in values]
    raise KeyError(f"Unable to locate class names in {path}. Provide --json_key to specify the field explicitly.")
  names: List[str] = []
  with path.open("r", encoding="utf-8") as handle:
    for line in handle:
      line = line.strip()
      if not line:
        continue
      parts = line.split()
      if parts[0].isdigit():
        names.append(" ".join(parts[1:]))
      else:
        names.append(line)
  if not names:
    raise ValueError(f"No class names parsed from {path}")
  return names


def resolve_templates(dataset: str, prompt: str, extra: List[str]):
  dataset_key = dataset.lower()
  templates = []
  if prompt:
    templates.append(prompt)
  templates.extend(DEFAULT_TEMPLATES.get(dataset_key, DEFAULT_TEMPLATES["generic"]))
  if extra:
    templates.extend(extra)
  # drop potential duplicates while preserving order
  seen = set()
  deduped = []
  for tpl in templates:
    if tpl in seen:
      continue
    seen.add(tpl)
    deduped.append(tpl)
  return deduped


def main():
  args = parse_args()
  try:
    class_names = read_class_names(Path(args.class_name_file), args.json_key)
  except FileNotFoundError:
    print(f"[WARN] No categories file for dataset '{args.dataset}', skip text features and disable CLIP prior for this dataset")
    return
  templates = resolve_templates(args.dataset, args.prompt, args.extra_templates)
  device = torch.device(args.device)
  model, _ = clip.load(args.clip_model, device=device)
  model.eval()

  text_features = []
  with torch.no_grad():
    for classname in progress_range(class_names, desc=f"CLIP text feats ({args.dataset})"):
      normalized_name = classname.replace("_", " ").replace("-", " ").strip()
      prompts = [tpl.format(classname=normalized_name, name=normalized_name) for tpl in templates]
      tokens = clip.tokenize(prompts).to(device)
      feats = model.encode_text(tokens)
      feats = feats / feats.norm(dim=-1, keepdim=True)
      mean_feat = feats.mean(dim=0)
      mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
      text_features.append(mean_feat)

  text_features = torch.stack(text_features).float().cpu()
  if args.dataset_key:
    output_obj = {args.dataset_key.lower(): text_features}
  else:
    output_obj = text_features
  output_path = Path(args.output)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(output_obj, output_path)
  print(f"Saved text features to {output_path} with shape {tuple(text_features.shape)} using {args.clip_model}")


if __name__ == "__main__":
  main()
