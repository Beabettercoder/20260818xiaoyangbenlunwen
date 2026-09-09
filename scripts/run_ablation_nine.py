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


def replace(command, key, value):
    command[command.index(key) + 1] = str(value)


def plans_for(config, args):
    plans = []
    for source in args.sources:
        entry = config[source]
        for shot in args.shots:
            for arm in args.arms:
                name = f'ablation_nine_{source}_{arm}_{shot}shot_{args.tag}'
                train, test = commands(SimpleNamespace(data_dir=Path(entry['data_dir']), smoke=args.smoke), arm, shot, name)
                for command in (train, test):
                    replace(command, '--source_dataset', source)
                replace(train, '--warmup', f'ablation_init_{source}_{args.tag}')
                targets = [x for x in DOMAINS if x != source]
                replace(test, '--target_datasets', ','.join(targets))
                if arm == 'A':
                    train = None
                    test = [sys.executable, str(ROOT / 'scripts/evaluate_pretrained_proto.py'),
                            '--data-dir', entry['data_dir'], '--checkpoint', entry['checkpoint'],
                            '--targets', *targets, '--shot', str(shot), '--episodes',
                            '2' if args.smoke else '1000', '--output',
                            str(ROOT / 'output/checkpoints' / name / 'acc_bscdfsl.txt')]
                plans.append(dict(source=source, shot=shot, arm=arm, name=name,
                                  targets=targets, train=train, test=test,
                                  head='cosine-prototype' if arm == 'A' else 'GNN'))
    return plans


def preflight(config, sources):
    records = {}
    for source in sources:
        entry = config[source]
        if entry.get('pretrained_source') != source:
            raise ValueError(f'{source}: explicitly confirm pretrained_source; never reuse another source checkpoint')
        checkpoint = Path(entry['checkpoint'])
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
        records[source] = record
    if len({r['sha256'] for r in records.values()}) != len(records):
        raise ValueError('Different sources have identical pretrained checkpoint contents')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=5)
    p.add_argument('--sources', nargs='+', choices=SOURCES, default=list(SOURCES))
    p.add_argument('--shots', nargs='+', type=int, choices=[1, 5], default=[5, 1])
    p.add_argument('--arms', nargs='+', choices=list('ABCD'), default=list('ABCD'))
    p.add_argument('--tag', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if not re.fullmatch(r'[\w-]+', args.tag):
        p.error('invalid tag')
    for values in (args.sources, args.shots, args.arms):
        if len(set(values)) != len(values):
            p.error('duplicate selections')
    args.tag = ('smoke_' if args.smoke else 'formal_') + args.tag
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    plans = plans_for(config, args)
    if args.dry_run:
        print(json.dumps(plans, indent=2)); return
    records = preflight(config, args.sources)
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
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                    status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).decode(),
                    caveat='A uses frozen cosine prototypes; B-E use trained GNN. A-B is not an isolated attack ablation.')
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
