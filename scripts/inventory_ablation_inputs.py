"""Read-only inventory of server splits and checkpoints; never infer provenance."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/mnt/sdc/wzj'))
    args = parser.parse_args()
    report = {'splits': [], 'checkpoints': [], 'note':
              'Checkpoint filenames do not establish training source or protocol.'}
    for folder in (args.root / 'datasets', args.root / 'datasets_causal_splits'):
        if not folder.exists():
            continue
        for path in sorted(folder.rglob('*.json')):
            if path.name not in ('base.json', 'val.json', 'novel.json'):
                continue
            row = {'path': str(path)}
            try:
                data = json.loads(path.read_text(encoding='utf-8-sig'))
                counts = Counter(data['image_labels'])
                row.update(images=len(data['image_names']), classes=len(counts),
                           min_per_class=min(counts.values(), default=0),
                           label_count_matches=len(data['image_labels']) == len(data['image_names']))
            except Exception as exc:
                row['error'] = str(exc)
            report['splits'].append(row)
    for repo in sorted(args.root.glob('SGA-Net*')):
        folder = repo / 'output' / 'checkpoints'
        if folder.is_dir():
            for path in sorted(folder.rglob('*.tar')):
                if path.name in ('399.tar', 'best_model.tar'):
                    report['checkpoints'].append({'path': str(path), 'bytes': path.stat().st_size})
    output = Path('logs') / ('ablation_inputs_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('[Saved]', output.resolve())
    print('[NEXT] Confirm source-specific pretraining provenance before creating the run config.')


if __name__ == '__main__':
    main()
