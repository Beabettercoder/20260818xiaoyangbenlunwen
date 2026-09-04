#!/usr/bin/env python3
"""Stack finished paper figures into one PNG/PDF without redrawing data."""

import argparse
from pathlib import Path

from PIL import Image


def parse_inputs(value):
    paths = [Path(item.strip()) for item in value.split(",") if item.strip()]
    if not paths:
        raise ValueError("--inputs must contain at least one image path")
    return paths


def resize_to_width(image, target_width):
    if image.width == target_width:
        return image
    scale = target_width / image.width
    height = int(round(image.height * scale))
    return image.resize((target_width, height), Image.Resampling.LANCZOS)


def save_canvas(canvas, output_stem, dpi):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    canvas.save(png_path, dpi=(dpi, dpi))
    try:
        canvas.save(pdf_path, "PDF", resolution=dpi)
        saved_pdf = pdf_path
    except PermissionError:
        saved_pdf = output_stem.with_name(output_stem.name + "_new").with_suffix(".pdf")
        canvas.save(saved_pdf, "PDF", resolution=dpi)
        print(f"[warn] PDF is locked, saved alternate: {saved_pdf}")
    print(f"[saved] {png_path} size={canvas.size}")
    print(f"[saved] {saved_pdf}")


def stack_images(paths, output_stem, layout, gap, dpi):
    opened = []
    try:
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"input image not found: {path}")
            opened.append(Image.open(path).convert("RGB"))

        if layout == "vertical":
            target_width = min(image.width for image in opened)
            images = [resize_to_width(image, target_width) for image in opened]
            width = target_width
            height = sum(image.height for image in images) + gap * (len(images) - 1)
            canvas = Image.new("RGB", (width, height), "white")
            y = 0
            for image in images:
                canvas.paste(image, (0, y))
                y += image.height + gap
        else:
            target_height = min(image.height for image in opened)
            images = []
            for image in opened:
                if image.height == target_height:
                    images.append(image)
                else:
                    scale = target_height / image.height
                    width = int(round(image.width * scale))
                    images.append(image.resize((width, target_height), Image.Resampling.LANCZOS))
            width = sum(image.width for image in images) + gap * (len(images) - 1)
            height = target_height
            canvas = Image.new("RGB", (width, height), "white")
            x = 0
            for image in images:
                canvas.paste(image, (x, 0))
                x += image.width + gap

        save_canvas(canvas, output_stem, dpi)
    finally:
        for image in opened:
            image.close()


def main():
    parser = argparse.ArgumentParser(description="Stack finished paper figures into one image.")
    parser.add_argument("--inputs", type=parse_inputs, required=True,
                        help="Comma-separated PNG paths in the desired order.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output stem or path. Suffix is ignored.")
    parser.add_argument("--layout", type=str, default="vertical", choices=["vertical", "horizontal"])
    parser.add_argument("--gap", type=int, default=120)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    output_stem = Path(args.output).with_suffix("")
    stack_images(args.inputs, output_stem, args.layout, args.gap, args.dpi)


if __name__ == "__main__":
    main()
