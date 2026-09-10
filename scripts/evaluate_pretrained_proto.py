"""Evaluate a trusted source-pretrained ResNet10 without an untrained GNN."""
import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_encoder(checkpoint, device):
    import torch
    from methods.backbone_multiblock import model_dict
    model = model_dict['ResNet10']()
    payload = torch.load(checkpoint, map_location='cpu')
    state = payload.get('state', {})
    # Match the training warmup loader exactly, rejecting partial/random encoders.
    state = {k.replace('feature.', ''): v for k, v in state.items() if 'feature.' in k}
    expected = model.state_dict()
    missing = [k for k, v in expected.items() if k not in state or state[k].shape != v.shape]
    if missing:
        raise ValueError(f'{checkpoint}: incompatible backbone keys: {missing}')
    model.load_state_dict({k: state[k] for k in expected}, strict=True)
    return model.to(device).eval()


def main():
    import numpy as np
    import torch
    from data.datamgr import get_few_shot_datamgr
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--targets', nargs='+', required=True)
    p.add_argument('--shot', type=int, choices=[1, 5], required=True)
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=64,
                   help='Feature-extraction batch size')
    p.add_argument('--workers', type=int, default=4,
                   help='DataLoader workers')
    p.add_argument('--prefetch-factor', type=int, default=2,
                   help='DataLoader prefetch factor when workers are enabled')
    args = p.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.prefetch_factor < 1:
        p.error('batch-size and prefetch-factor must be >= 1; workers must be >= 0')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_encoder(args.checkpoint, device)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    with args.output.open('x', encoding='utf-8') as result:
        for target in args.targets:
            loader_kwargs = dict(
                batch_size=args.batch_size, data_root=args.data_dir, split='novel',
                num_workers=args.workers, pin_memory=torch.cuda.is_available(),
            )
            if args.workers > 0:
                loader_kwargs.update(persistent_workers=True,
                                     prefetch_factor=args.prefetch_factor)
            else:
                loader_kwargs['persistent_workers'] = False
            loader = get_few_shot_datamgr(target, episodic=False, image_size=224,
                **loader_kwargs).get_data_loader(aug=False)
            features = {}
            with torch.inference_mode():
                for x, y in loader:
                    values = model(x.to(device)).cpu().numpy()
                    for z, label in zip(values, y.tolist()):
                        features.setdefault(int(label), []).append(z)
            random.seed(0); np.random.seed(0)
            accuracies = []
            for _ in range(args.episodes):
                classes = random.sample(list(features), 5)
                episode = np.array([np.array(features[c])[np.random.permutation(len(features[c]))[:args.shot + 15]] for c in classes])
                prototypes = episode[:, :args.shot].mean(axis=1)
                queries = episode[:, args.shot:].reshape(75, -1)
                prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
                queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
                prediction = (queries @ prototypes.T).argmax(axis=1)
                accuracies.append(100 * np.mean(prediction == np.repeat(np.arange(5), 15)))
            line = f'  {args.episodes} test iterations ({target}): Acc = {np.mean(accuracies):.2f}% +- {1.96 * np.std(accuracies) / np.sqrt(args.episodes):.2f}%'
            print(line, flush=True); result.write(line + '\n'); result.flush()


if __name__ == '__main__':
    main()
