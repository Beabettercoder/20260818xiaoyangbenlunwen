"""Create the exact shared-warmup config used by the nine-direction ablation."""
import argparse
import json
from pathlib import Path


SOURCES = ("NWPU", "AID", "UCM")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True,
                        help="Root created by build_ablation_source_splits.py")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="One shared warmup checkpoint, normally baseline/399.tar")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON config path to create")
    args = parser.parse_args()

    split_root = args.split_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"shared checkpoint not found: {checkpoint}")
    missing = [split_root / f"{source}_source" for source in SOURCES
               if not (split_root / f"{source}_source").is_dir()]
    if missing:
        raise FileNotFoundError(
            "missing generated source roots; run build_ablation_source_splits.py first: "
            + ", ".join(str(path) for path in missing))
    if output.exists():
        raise FileExistsError(f"config already exists; choose a new path: {output}")

    config = {
        source: {
            "data_dir": str(split_root / f"{source}_source"),
            "checkpoint": str(checkpoint),
            "pretrained_source": "shared",
        }
        for source in SOURCES
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"[DONE] wrote shared-warmup config: {output}")
    print(f"[Checkpoint] {checkpoint}")
    for source in SOURCES:
        print(f"[{source}] {config[source]['data_dir']}")


if __name__ == "__main__":
    main()
