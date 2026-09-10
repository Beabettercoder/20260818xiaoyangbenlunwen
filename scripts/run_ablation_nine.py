"""Source-specific source-only ablations. No training starts without preflight."""
import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

from run_source_only_ablation import ROOT, commands, run, sha

SOURCES = ('NWPU', 'AID', 'UCM')
DOMAINS = (*SOURCES, 'EuroSAT')
ARM_MODULES = {
    'A': dict(style_attack=False, progressive=False, text_scales=False, extra_epochs=0),
    'B': dict(style_attack=True, progressive=False, text_scales=False, extra_epochs=200),
    'C': dict(style_attack=True, progressive=True, text_scales=False, extra_epochs=200),
    'D': dict(style_attack=True, progressive=False, text_scales=True, extra_epochs=200),
}

# Loader-only speed settings. These do not change the episode protocol or the
# model/loss switches, so they remain fair across A-D and all source domains.
DEFAULT_TRAIN_WORKERS = 4
DEFAULT_EVAL_WORKERS = 4
DEFAULT_FEATURE_BATCH_SIZE = 64
DEFAULT_PREFETCH_FACTOR = 2


def replace(command, key, value):
    command[command.index(key) + 1] = str(value)


def plans_for(config, args):
    plans = []
    train_workers = int(getattr(args, 'train_workers', DEFAULT_TRAIN_WORKERS))
    eval_workers = int(getattr(args, 'eval_workers', DEFAULT_EVAL_WORKERS))
    feature_batch_size = int(getattr(args, 'feature_batch_size', DEFAULT_FEATURE_BATCH_SIZE))
    prefetch_factor = int(getattr(args, 'prefetch_factor', DEFAULT_PREFETCH_FACTOR))
    for source in args.sources:
        entry = config[source]
        for shot in args.shots:
            for arm in args.arms:
                if arm not in ARM_MODULES:
                    raise ValueError('Only A-D are supported; E must not be scheduled')
                name = f'ablation_nine_{source}_{arm}_{shot}shot_{args.tag}'
                train, test = commands(SimpleNamespace(data_dir=Path(entry['data_dir']), smoke=args.smoke), arm, shot, name)
                for command in (train, test):
                    replace(command, '--source_dataset', source)
                    replace(command, '--eval_num_workers', eval_workers)
                    replace(command, '--feature_batch_size', feature_batch_size)
                    command.extend(['--prefetch_factor', str(prefetch_factor)])
                    command.extend(['--model', 'ResNet10', '--method', 'baseline',
                                    '--train_n_way', '5', '--test_n_way', '5'])
                replace(train, '--train_num_workers', train_workers)
                train.extend(['--start_epoch', '0', '--epsilon_block1', '0.8',
                              '--epsilon_block2', '0.08', '--epsilon_block3', '0.008',
                              '--use_style_prompt', '0', '--clip_model_name', 'ViT-B/32'])
                replace(train, '--warmup', f'ablation_init_{source}_{args.tag}')
                targets = [x for x in DOMAINS if x != source]
                replace(test, '--target_datasets', ','.join(targets))
                if arm == 'A':
                    train = None
                    test = [sys.executable, str(ROOT / 'scripts/evaluate_pretrained_proto.py'),
                            '--data-dir', entry['data_dir'], '--checkpoint', entry['checkpoint'],
                            '--targets', *targets, '--shot', str(shot), '--episodes',
                            '2' if args.smoke else '1000', '--output',
                            str(ROOT / 'output/checkpoints' / name / 'acc_bscdfsl.txt'),
                            '--batch-size', str(feature_batch_size),
                            '--workers', str(eval_workers),
                            '--prefetch-factor', str(prefetch_factor)]
                plans.append(dict(source=source, shot=shot, arm=arm, name=name,
                                  targets=targets, train=train, test=test,
                                  modules={**ARM_MODULES[arm], 'extra_epochs':
                                           0 if arm == 'A' else (1 if args.smoke else 200)},
                                  head='cosine-prototype' if arm == 'A' else 'GNN'))
    return plans


def preflight(config, sources, allow_shared_checkpoint=False):
    records = {}
    for source in sources:
        entry = config[source]
        provenance = entry.get('pretrained_source')
        if provenance != source:
            if not (allow_shared_checkpoint and provenance == 'shared'):
                raise ValueError(
                    f'{source}: pretrained_source must be {source!r}; '
                    "use pretrained_source='shared' together with "
                    '--allow-shared-checkpoint only for the explicitly shared warmup protocol')
        checkpoint = Path(entry['checkpoint'])
        if checkpoint.name != '399.tar':
            raise ValueError(
                f'{source}: checkpoint must be the common warmup file named 399.tar; '
                f'got {checkpoint}')
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        record = {'checkpoint': str(checkpoint), 'sha256': sha(checkpoint), 'splits': {}}
        paths = {}
        for domain in DOMAINS:
            for split in (('base', 'val') if domain == source else ('novel',)):
                path = Path(entry['data_dir']) / domain / f'{split}.json'
                data = json.loads(path.read_text(encoding='utf-8'))
                labels, images = data['image_labels'], data['image_names']
                counts = Counter(labels)
                minimum = 10 if split == 'base' else 20
                if len(labels) != len(images) or len(counts) < 5 or min(counts.values()) < minimum:
                    raise ValueError(f'{path}: need >=5 classes, >= {minimum} images/class; counts={dict(counts)}')
                resolved = {str(Path(p).resolve()) for p in images}
                if len(resolved) != len(images):
                    raise ValueError(f'duplicate images: {path}')
                missing = next((p for p in resolved if not Path(p).is_file()), None)
                if missing:
                    raise FileNotFoundError(f'{path}: missing image {missing}')
                paths[(domain, split)] = resolved
                record['splits'][str(path)] = dict(sha256=sha(path), counts=dict(counts))
        if paths[(source, 'base')] & paths[(source, 'val')]:
            raise ValueError(f'{source}: source base/val image overlap')
        source_images = paths[(source, 'base')] | paths[(source, 'val')]
        for domain in DOMAINS:
            if domain != source and source_images & paths[(domain, 'novel')]:
                raise ValueError(f'{source}->{domain}: image leakage')
        record['pretrained_source'] = provenance
        records[source] = record
    by_hash = {}
    for source, record in records.items():
        by_hash.setdefault(record['sha256'], []).append(source)
    duplicate_groups = [owners for owners in by_hash.values() if len(owners) > 1]
    if duplicate_groups and not allow_shared_checkpoint:
        raise ValueError(
            'Different sources have identical pretrained checkpoint contents; '
            'pass --allow-shared-checkpoint only when this is intentional')
    for owners in duplicate_groups:
        if any(config[source].get('pretrained_source') != 'shared' for source in owners):
            raise ValueError(
                f'identical checkpoint group {owners} must be marked '
                "pretrained_source='shared'")
    shared_sources = [source for source in sources
                      if config[source].get('pretrained_source') == 'shared']
    if shared_sources and not allow_shared_checkpoint:
        raise ValueError(
            "pretrained_source='shared' requires --allow-shared-checkpoint")
    if shared_sources and len({records[source]['sha256'] for source in shared_sources}) != 1:
        raise ValueError('all sources marked shared must use the same checkpoint contents')
    if shared_sources and set(shared_sources) != set(sources):
        raise ValueError(
            'the locked protocol requires every selected source to use the shared '
            "baseline/399.tar; do not mix shared and source-specific warmups")
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=6)
    p.add_argument('--sources', nargs='+', choices=SOURCES, default=list(SOURCES))
    p.add_argument('--shots', nargs='+', type=int, choices=[1, 5], default=[5, 1])
    p.add_argument('--arms', nargs='+', choices=list('ABCD'), default=list('ABCD'))
    p.add_argument('--tag', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--train-workers', dest='train_workers', type=int,
                   default=DEFAULT_TRAIN_WORKERS,
                   help='DataLoader workers for B-D training (loader-only setting)')
    p.add_argument('--eval-workers', dest='eval_workers', type=int,
                   default=DEFAULT_EVAL_WORKERS,
                   help='DataLoader workers for evaluation (loader-only setting)')
    p.add_argument('--feature-batch-size', dest='feature_batch_size', type=int,
                   default=DEFAULT_FEATURE_BATCH_SIZE,
                   help='Feature-extraction batch size for evaluation')
    p.add_argument('--prefetch-factor', dest='prefetch_factor', type=int,
                   default=DEFAULT_PREFETCH_FACTOR,
                   help='DataLoader prefetch factor when workers are enabled')
    p.add_argument('--allow-shared-checkpoint', action='store_true',
                   help='Explicitly allow all source domains to reuse one warmup checkpoint')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if not re.fullmatch(r'[\w-]+', args.tag):
        p.error('invalid tag')
    for values in (args.sources, args.shots, args.arms):
        if len(set(values)) != len(values):
            p.error('duplicate selections')
    if args.train_workers < 0 or args.eval_workers < 0:
        p.error('worker counts must be >= 0')
    if args.feature_batch_size < 1 or args.prefetch_factor < 1:
        p.error('feature batch size and prefetch factor must be >= 1')
    args.tag = ('smoke_' if args.smoke else 'formal_') + args.tag
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    plans = plans_for(config, args)
    if args.dry_run:
        print(json.dumps(plans, indent=2)); return
    records = preflight(config, args.sources, args.allow_shared_checkpoint)
    if args.check_only:
        print(json.dumps(records, indent=2)); print('[PASS] data/path preflight (checkpoint tensors not checked)'); return
    # Validate source backbone coverage before any expensive training.
    from evaluate_pretrained_proto import load_encoder
    for source in args.sources:
        load_encoder(config[source]['checkpoint'], 'cpu')
    log_root = ROOT / 'logs/ablation_nine' / args.tag
    for plan in plans:
        if (ROOT / 'output/checkpoints' / plan['name']).exists():
            raise FileExistsError(plan['name'])
    log_root.mkdir(parents=True, exist_ok=False)
    manifest = dict(config=config, inputs=records, runs=plans, gpu=args.gpu, seed=0,
                    checkpoint_policy=dict(
                        allow_shared_checkpoint=args.allow_shared_checkpoint,
                        description=('All sources intentionally reuse the same '
                                     'baseline/399.tar warmup checkpoint.')
                        if args.allow_shared_checkpoint else
                        'Source-specific warmup checkpoints are required.'),
                    loader=dict(train_workers=args.train_workers,
                                eval_workers=args.eval_workers,
                                feature_batch_size=args.feature_batch_size,
                                prefetch_factor=args.prefetch_factor),
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                    status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).decode(),
                    reference=dict(name='nwpu_5shot_baseline_restore',
                                   checkpoint_sha256='a7de4a088c787839e3343cf79fb57c36fc6d1eae74fb5675878ceabbcdff6eea',
                                   exact_parameter_match_verified=False,
                                   reason='Reference params_meta contains only dataset/model/method/ways/shot/name; no complete training configuration.'),
                    caveat='A uses frozen cosine prototypes; B-D use trained GNN. A-B is not an isolated attack ablation.')
    def save():
        (log_root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save()
    for source in args.sources:
        folder = ROOT / 'output/checkpoints' / f'ablation_init_{source}_{args.tag}'
        folder.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config[source]['checkpoint'], folder / '399.tar')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    import csv
    with (log_root / 'results.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'target', 'arm', 'shot', 'accuracy_percent', 'interval', 'episodes', 'head'])
        for plan in plans:
            plan['status'] = 'running'; save()
            if plan['train']:
                run(plan['train'], log_root / (plan['name'] + '_train.log'), env)
            run(plan['test'], log_root / (plan['name'] + '_test.log'), env)
            folder = ROOT / 'output/checkpoints' / plan['name']
            matches = re.findall(r'(\d+) test iterations \(([^)]+)\): Acc = ([\d.]+)% \+- ([\d.]+)%',
                                 (folder / 'acc_bscdfsl.txt').read_text())
            if len(matches) != 3 or {m[1] for m in matches} != set(plan['targets']) or any(int(m[0]) != (2 if args.smoke else 1000) for m in matches):
                raise RuntimeError(f'incomplete evaluation: {folder}')
            for episodes, target, accuracy, interval in matches:
                writer.writerow([plan['source'], target, plan['arm'], plan['shot'], accuracy, interval, episodes, plan['head']])
            f.flush()
            plan['status'] = 'completed'
            if plan['train']:
                plan['checkpoint_sha256'] = sha(folder / 'best_model.tar')
            save()
    print(f'[DONE] {log_root}')


if __name__ == '__main__':
    main()
