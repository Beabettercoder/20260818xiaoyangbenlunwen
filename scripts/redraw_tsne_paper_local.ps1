$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:TSNE_PAPER_FONT = Join-Path $RepoRoot "fonts\times.ttf"
$env:TSNE_PAPER_FONT_BOLD = Join-Path $RepoRoot "fonts\timesbd.ttf"

@'
import re
from pathlib import Path

import numpy as np

from scripts.plot_tsne_paper_nwpu_source import plot_tsne_figure_matplotlib

root = Path(r"output/tsne_paper_nwpu_source_20260501_v1/main")
coord_files = sorted(root.glob("tsne_paper_NWPU_to_*_main_seed*_coords.npz"))
if not coord_files:
    raise SystemExit(f"No coordinate files found under: {root.resolve()}")

for coord_path in coord_files:
    match = re.match(r"tsne_paper_NWPU_to_(.+)_main_seed(\d+)_coords\.npz", coord_path.name)
    if not match:
        continue
    target, seed = match.group(1), int(match.group(2))
    data = np.load(coord_path, allow_pickle=True)
    plot_tsne_figure_matplotlib(
        "NWPU",
        target,
        list(data["class_names"]),
        data["labels"],
        data["emb_base"],
        data["emb_sga"],
        seed,
        str(root),
        "main",
        300,
        axis_mode="panel",
        point_size=7.0,
    )
'@ | python -
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python .\scripts\combine_tsne_paper_grid.py `
  --input_dir "output\tsne_paper_nwpu_source_20260501_v1\main" `
  --output_dir "output\tsne_paper_nwpu_source_20260501_v1\main" `
  --source_dataset NWPU `
  --figure_tag main `
  --seed 0 `
  --dpi 300 `
  --point_size 7.0
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done. Output: $RepoRoot\output\tsne_paper_nwpu_source_20260501_v1\main"
