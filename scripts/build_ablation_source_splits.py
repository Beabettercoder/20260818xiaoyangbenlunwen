"""Build reproducible source-domain split roots for the nine ablation transfers.

The existing NWPU split is used as the reference for the source-domain
base/val ratio.  For AID and UCM, whose repository copies may only contain a
novel split, that split is treated as the pool of real images and is divided
class-wise into base and val.  In each generated source root, every other
domain is evaluation-only: its novel split contains all available real
images, while base/val are empty.  No image is copied or synthesized.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


DOMAINS = ("NWPU", "AID", "UCM", "EuroSAT")


def resolve_image(raw_name, input_root, domain):
    raw = Path(str(raw_name))
    candidates = [raw] if raw.is_absolute() else [
        input_root / raw,
        input_root / domain / raw,
        input_root / domain / "images" / raw,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(
        f"cannot resolve image {raw_name!r} for {domain}; tried "
        + ", ".join(str(x) for x in candidates))


def read_split(input_root, domain, split):
    path = input_root / domain / f"{split}.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    names = payload.get("image_names", [])
    labels = payload.get("image_labels", [])
    if len(names) != len(labels):
        raise ValueError(f"{path}: image_names/image_labels length mismatch")
    return [(resolve_image(name, input_root, domain), label)
            for name, label in zip(names, labels)]


def collect_domain(input_root, domain):
    merged = {}
    origin_counts = {}
    for split in ("base", "val", "novel"):
        rows = read_split(input_root, domain, split)
        origin_counts[split] = len(rows)
        for image, label in rows:
            if image in merged and str(merged[image]) != str(label):
                raise ValueError(f"{domain}: image has conflicting labels: {image}")
            merged[image] = label
    rows = sorted(merged.items(), key=lambda item: item[0])
    if not rows:
        raise ValueError(f"{domain}: no usable images found in base/val/novel JSON")
    return [(image, label) for image, label in rows], origin_counts


def counts(rows):
    return dict(sorted(Counter(str(label) for _, label in rows).items()))


def write_split(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_names": [image for image, _ in rows],
        "image_labels": [label for _, label in rows],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def stable_label_seed(seed, label):
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], "big")


def split_like_nwpu(rows, reference_base, reference_val, seed):
    ref_total = len(reference_base) + len(reference_val)
    if ref_total == 0:
        raise ValueError("NWPU base/val reference is empty")
    reference_val_fraction = len(reference_val) / float(ref_total)
    by_label = {}
    for image, label in rows:
        by_label.setdefault(str(label), []).append((image, label))
    base_rows, val_rows = [], []
    for label in sorted(by_label):
        items = list(by_label[label])
        random.Random(stable_label_seed(seed, label)).shuffle(items)
        # Preserve the NWPU proportion where possible, but retain the
        # >=20/class validation minimum required by the benchmark preflight.
        val_count = max(20, round(len(items) * reference_val_fraction))
        base_count = len(items) - val_count
        if base_count < 10:
            raise ValueError(
                f"{label}: {len(items)} images cannot satisfy base>=10 and val>=20")
        val_rows.extend(items[:val_count])
        base_rows.extend(items[val_count:])
    return (sorted(base_rows), sorted(val_rows), reference_val_fraction)


def validate(rows, split, minimum):
    if not rows:
        return
    per_class = Counter(str(label) for _, label in rows)
    if len(per_class) < 5 or min(per_class.values()) < minimum:
        raise ValueError(
            f"generated {split}: need >=5 classes and >= {minimum}/class; "
            f"counts={dict(per_class)}")


def verify_images(rows_by_domain):
    """Fail early on unreadable files instead of crashing during evaluation."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("--verify-images requires Pillow in the active environment") from exc
    unique_paths = sorted({image for rows in rows_by_domain.values() for image, _ in rows})
    bad = []
    for index, image in enumerate(unique_paths, 1):
        try:
            with Image.open(image) as handle:
                handle.verify()
        except Exception as exc:  # Pillow raises several exception types here.
            bad.append((image, str(exc)))
        if index % 5000 == 0 or index == len(unique_paths):
            print(f"[Verify] {index}/{len(unique_paths)} images checked")
    if bad:
        details = "\n".join(f"  {image}: {reason}" for image, reason in bad[:20])
        extra = "" if len(bad) <= 20 else f"\n  ... and {len(bad) - 20} more"
        raise RuntimeError("unreadable images found:\n" + details + extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-source", choices=DOMAINS, default="NWPU")
    parser.add_argument("--sources", nargs="+", choices=("NWPU", "AID", "UCM"),
                        default=["NWPU", "AID", "UCM"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-images", action="store_true",
                        help="verify every unique source/target image with Pillow before writing splits")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_roots = [output_root / f"{source}_source" for source in args.sources]
    conflicts = [root for root in source_roots if root.exists()]
    if conflicts:
        raise FileExistsError(
            "output already exists; choose a new --output-root: "
            + ", ".join(str(x) for x in conflicts))

    domain_rows = {}
    domain_origins = {}
    for domain in DOMAINS:
        domain_rows[domain], domain_origins[domain] = collect_domain(input_root, domain)
    if args.verify_images:
        verify_images(domain_rows)
    reference_base = read_split(input_root, args.reference_source, "base")
    reference_val = read_split(input_root, args.reference_source, "val")
    if not reference_base or not reference_val:
        raise ValueError(
            f"reference {args.reference_source} must have non-empty base.json and val.json")
    reference_fraction = len(reference_val) / float(len(reference_base) + len(reference_val))

    manifest = {
        "seed": args.seed,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "reference_source": args.reference_source,
        "reference_counts": {
            "base": counts(reference_base),
            "val": counts(reference_val),
            "val_fraction": reference_fraction,
        },
        "policy": {
            "source_base_val": "NWPU reference val proportion, stratified by class; val minimum 20/class, base minimum 10/class",
            "target_novel": "all available real images from base/val/novel; target base/val are empty",
            "source_novel": "empty, matching the NWPU source protocol",
            "images_copied": False,
        },
        "sources": {},
    }

    for source, source_root in zip(args.sources, source_roots):
        source_root.mkdir(parents=True)
        if source == args.reference_source:
            source_base = sorted(reference_base)
            source_val = sorted(reference_val)
            split_mode = "copied_reference_base_val"
        else:
            source_base, source_val, _ = split_like_nwpu(
                domain_rows[source], reference_base, reference_val, args.seed)
            split_mode = "stratified_from_available_images"
        validate(source_base, "source base", 10)
        validate(source_val, "source val", 20)

        source_record = {
            "root": str(source_root),
            "source": source,
            "source_split_mode": split_mode,
            "domains": {},
        }
        for domain in DOMAINS:
            if domain == source:
                base, val, novel = source_base, source_val, []
                role = "source"
            else:
                base, val, novel = [], [], domain_rows[domain]
                role = "target_novel"
                validate(novel, f"{source}->{domain} target novel", 20)
            domain_root = source_root / domain
            write_split(domain_root / "base.json", base)
            write_split(domain_root / "val.json", val)
            write_split(domain_root / "novel.json", novel)
            source_record["domains"][domain] = {
                "role": role,
                "base_count": len(base),
                "val_count": len(val),
                "novel_count": len(novel),
                "base_class_counts": counts(base),
                "val_class_counts": counts(val),
                "novel_class_counts": counts(novel),
                "input_pool_counts": domain_origins[domain],
            }
        manifest["sources"][source] = source_record

    manifest_path = output_root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(f"[DONE] generated {len(args.sources)} source roots under {output_root}")
    print(f"[DONE] NWPU val fraction={reference_fraction:.6f}")
    for source in args.sources:
        source_record = manifest["sources"][source]
        print(f"[{source}] root={source_record['root']}")
        for domain, record in source_record["domains"].items():
            print(f"  {domain}: base={record['base_count']} val={record['val_count']} novel={record['novel_count']}")
    print(f"[Saved] {manifest_path}")


if __name__ == "__main__":
    main()
