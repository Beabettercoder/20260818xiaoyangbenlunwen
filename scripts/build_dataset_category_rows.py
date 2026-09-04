#!/usr/bin/env python3
"""Build paper-ready dataset example rows from local class folders."""

import argparse
import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


CATEGORY_COLUMNS = ["Farmland", "Forest", "Residential", "River", "Water"]


DATASET_SPECS = [
    (
        "NWPU-RESISC45",
        "NWPU",
        {
            "Farmland": "rectangular_farmland",
            "Forest": "forest",
            "Residential": "dense_residential",
            "River": "river",
            "Water": "lake",
        },
    ),
    (
        "EuroSAT",
        "EuroSAT",
        {
            "Farmland": "AnnualCrop",
            "Forest": "Forest",
            "Residential": "Residential",
            "River": "River",
            "Water": "SeaLake",
        },
    ),
    (
        "AID",
        "AID",
        {
            "Farmland": "Farmland",
            "Forest": "Forest",
            "Residential": "DenseResidential",
            "River": "River",
            "Water": "Pond",
        },
    ),
    (
        "UCM",
        "UCM",
        {
            "Farmland": "agricultural",
            "Forest": "forest",
            "Residential": "denseresidential",
            "River": "river",
            "Water": "harbor",
        },
    ),
]


SAMPLE_INDEX_OVERRIDES = {
    ("EuroSAT", "River"): 4,
    ("EuroSAT", "Water"): 14,
    ("UCM", "Forest"): 22,
}


def natural_key(path):
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def list_images(class_dir):
    if not class_dir.is_dir():
        raise FileNotFoundError(f"class folder not found: {class_dir}")
    images = sorted(
        [path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )
    if not images:
        raise FileNotFoundError(f"no image files found in: {class_dir}")
    return images


def load_tile(path, tile_size):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.fit(image, (tile_size, tile_size), method=Image.Resampling.LANCZOS)


def build_row(image_paths, tile_size, gap, border):
    width = len(image_paths) * tile_size + (len(image_paths) - 1) * gap
    height = tile_size
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    x = 0
    for path in image_paths:
        tile = load_tile(path, tile_size)
        canvas.paste(tile, (x, 0))
        if border > 0:
            draw.rectangle(
                [x, 0, x + tile_size - 1, tile_size - 1],
                outline=(210, 210, 210),
                width=border,
            )
        x += tile_size + gap
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Build four dataset example row images.")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Root directory containing the dataset folders")
    parser.add_argument("--output-dir", type=Path, default=Path("output/dataset_category_rows_20260503_v1"))
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--gap", type=int, default=18)
    parser.add_argument("--border", type=int, default=2)
    parser.add_argument("--sample-index", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping_rows = [["dataset", "column", "display_category", "folder", "source_image", "output_image"]]

    for display_name, root_name, class_map in DATASET_SPECS:
        image_paths = []
        for column in CATEGORY_COLUMNS:
            folder_name = class_map[column]
            images = list_images(args.dataset_root / root_name / folder_name)
            index = SAMPLE_INDEX_OVERRIDES.get((root_name, column), args.sample_index)
            index = min(index, len(images) - 1)
            image_paths.append(images[index])

        row = build_row(image_paths, args.tile_size, args.gap, args.border)
        safe_name = display_name.lower().replace("-", "_").replace(" ", "_")
        output_path = args.output_dir / f"{safe_name}_category_row.png"
        row.save(output_path, dpi=(args.dpi, args.dpi))
        print(f"[saved] {output_path} size={row.size}")

        for column, source_path in zip(CATEGORY_COLUMNS, image_paths):
            folder_name = class_map[column]
            mapping_rows.append(
                [
                    display_name,
                    str(CATEGORY_COLUMNS.index(column) + 1),
                    column,
                    folder_name,
                    str(source_path),
                    str(output_path),
                ]
            )

    mapping_path = args.output_dir / "source_mapping.csv"
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(mapping_rows)
    print(f"[saved] {mapping_path}")


if __name__ == "__main__":
    main()
