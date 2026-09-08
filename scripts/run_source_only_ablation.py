"""Sequential, fail-fast A-E experiments. Dry run never imports torch."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"A": ("independent", False, True), "B": ("independent", False, False),
        "C": ("progressive", False, False), "D": ("independent", True, False),
        "E": ("progressive", True, False)}
TARGETS = ("AID", "UCM", "EuroSAT")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def commands(args, arm, shot, name):
    mode, text, disabled = ARMS[arm]
    common = ["--data_dir", str(args.data_dir), "--source_dataset", "NWPU",
              "--name", name, "--save_dir", str(ROOT / "output"), "--n_shot", str(shot),
              "--n_query", "15", "--feature_batch_size", "32", "--eval_num_workers", "2",
              "--text_guide_gradient", "0", "--eval_proto_refine", "0",
              "--text_calibration_weight", "0"]
    train = [sys.executable, str(ROOT / "metatrain_StyleAdv_RN.py"), *common,
             "--warmup", "baseline", "--require_warmup", "1", "--train_n_query", "5",
             "--train_aug", "--stop_epoch", "1" if args.smoke else "200",
             "--train_episodes", "2" if args.smoke else "100",
             "--val_episodes", "2" if args.smoke else "100", "--do_bscdfsl", "0",
             "--target_ssl_weight", "0", "--semantic_anchor", "0",
             "--semantic_drift_control", "0", "--skip_source_test", "1",
             "--train_num_workers", "2", "--pin_memory", "1", "--persistent_workers", "1",
             "--style_attack_mode", mode, "--text_guide_epsilon", str(int(text)),
             "--text_weight_mode", "fixed", "--text_weight_fixed", "1.0",
             "--epsilon_scale_min", "0.3", "--epsilon_scale_max", "1.7",
             "--use_prompt_ensemble", "1"]
    if disabled:
        train.append("--disable_style_adv_generator")
    if text:
        train.append("--use_text_guidance")
    test = [sys.executable, str(ROOT / "test_function_bscdfsl_benchmark.py"), *common,
            "--eval_seed", "0",
            "--target_datasets", ",".join(TARGETS), "--text_guide_epsilon", "0",
            "--n_episodes_test", "2" if args.smoke else "1000"]
    return train, test


def run(command, path, env):
    print(shlex.join(command), flush=True)
    with path.open("x", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace") as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError(f"Command failed; see {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--shots", type=int, nargs="+", choices=[1, 5], default=[5, 1])
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--tag", default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--warmup-sha256", help="Optional full SHA-256 from the trusted server checkpoint")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("tag must contain only letters, digits, underscore or hyphen")
    if len(set(args.shots)) != len(args.shots) or len(set(args.arms)) != len(args.arms):
        parser.error("duplicate arms/shots are not allowed")
    args.data_dir = args.data_dir.resolve()
    tag = ("smoke_" if args.smoke else "formal_") + args.tag
    plans = []
    for shot in args.shots:
        for arm in args.arms:
            name = f"ablation_v2_{arm}_NWPU_{shot}shot_{tag}"
            train, test = commands(args, arm, shot, name)
            plans.append(dict(arm=arm, shot=shot, name=name, train=train, test=test))
    if args.dry_run:
        print(json.dumps(dict(gpu=args.gpu, seed=0, runs=plans), indent=2))
        return
    warmup_dir = ROOT / "output/checkpoints/baseline"
    warmup = warmup_dir / "399.tar"
    if not warmup.is_file():
        raise RuntimeError("Missing warmup baseline/399.tar")
    warmup_sha = sha(warmup)
    if args.warmup_sha256 and warmup_sha != args.warmup_sha256.lower():
        raise RuntimeError("Warmup SHA-256 mismatch")
    epochs = [int(p.stem) for p in warmup_dir.glob("*.tar") if p.stem.isdigit()]
    if max(epochs) != 399:
        raise RuntimeError("Warmup loader would choose an epoch other than 399")
    for plan in plans:
        if (ROOT / "output/checkpoints" / plan["name"]).exists():
            raise FileExistsError(plan["name"])
    log_root = ROOT / "logs/source_only_ablation" / tag
    log_root.mkdir(parents=True, exist_ok=False)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    manifest = dict(protocol="source-only A-E v2", seed=0, gpu=args.gpu,
                    smoke=args.smoke, warmup=str(warmup), warmup_sha256=warmup_sha,
                    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                    git_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode(),
                    code_sha256={p: sha(ROOT / p) for p in tracked if p.endswith((".py", ".sh")) and (ROOT / p).is_file()},
                    split_sha256={str(p): sha(p) for p in sorted(args.data_dir.rglob("*.json"))},
                    runs=plans)
    manifest_path = log_root / "manifest.json"
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with (log_root / "results.csv").open("x", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["arm", "shot", "target", "accuracy_percent", "reported_interval", "episodes", "checkpoint"])
        for plan in plans:
            plan["status"] = "training"
            save()
            run(plan["train"], log_root / (plan["name"] + "_train.log"), env)
            checkpoint = ROOT / "output/checkpoints" / plan["name"] / "best_model.tar"
            if not checkpoint.is_file():
                raise RuntimeError(f"Missing checkpoint: {checkpoint}")
            plan["checkpoint_sha256"] = sha(checkpoint)
            plan["status"] = "testing"
            save()
            run(plan["test"], log_root / (plan["name"] + "_test.log"), env)
            result = (checkpoint.parent / "acc_bscdfsl.txt").read_text()
            matches = re.findall(r"(\d+) test iterations \(([^)]+)\): Acc = ([\d.]+)% \+- ([\d.]+)%", result)
            expected = 2 if args.smoke else 1000
            if len(matches) != 3 or {m[1] for m in matches} != set(TARGETS) or any(int(m[0]) != expected for m in matches):
                raise RuntimeError("Incomplete target evaluation; refusing to mark run successful")
            for episodes, target, acc, interval in matches:
                writer.writerow([plan["arm"], plan["shot"], target, acc, interval, episodes, str(checkpoint)])
            output.flush()
            plan["status"] = "completed"
            save()
    print(f"[DONE] {log_root / 'results.csv'}")


if __name__ == "__main__":
    main()
