#!/usr/bin/env python3
"""
Resplit datasets for cross-domain few-shot learning:
- Configurable source domain: split into base/val/novel
- All other domains are treated as target: base/val empty, novel keeps all classes
"""

import json
import os
import sys
from collections import defaultdict
import random
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.dataset_utils import get_dkd_split_path, save_dkd_split

SPLIT_SPEC_FILENAME = 'split_spec.json'
SOURCE_CLASS_DISJOINT_MODE = 'class_disjoint'
SOURCE_IMAGE_DISJOINT_MODE = 'image_disjoint_shared_classes'
TARGET_NOVEL_ONLY_MODE = 'novel_only_target'
PROTOCOL_LEGACY = 'legacy'
PROTOCOL_STRICT_PAIRWISE = 'strict_pairwise'
PROTOCOL_DMPC = 'dmpc'
PAIRWISE_PROTOCOL_NAMES = {PROTOCOL_STRICT_PAIRWISE, PROTOCOL_DMPC}
SOURCE_STRICT_PAIRWISE_MODE = 'strict_pairwise_source_image_disjoint_all_classes'
TARGET_STRICT_PAIRWISE_MODE = 'strict_pairwise_target_unlabeled_eval'
SOURCE_DMPC_MODE = 'dmpc_source_image_disjoint_all_classes'
TARGET_DMPC_MODE = 'dmpc_target_unlabeled_eval'
SOURCE_PAIRWISE_MODE_NAMES = {SOURCE_STRICT_PAIRWISE_MODE, SOURCE_DMPC_MODE}
TARGET_PAIRWISE_MODE_NAMES = {TARGET_STRICT_PAIRWISE_MODE, TARGET_DMPC_MODE}

def load_json(json_path):
    """Load a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, json_path):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"  ✅ Saved: {json_path}")

def group_by_class(image_names, image_labels):
    """Group images by their class labels."""
    class_to_images = defaultdict(list)
    for img, label in zip(image_names, image_labels):
        class_to_images[label].append(img)
    return class_to_images


def save_split_spec(data_dir, dataset_name, spec):
    """Persist the protocol used to construct the current dataset splits."""
    spec_path = os.path.join(data_dir, dataset_name, SPLIT_SPEC_FILENAME)
    save_json(spec, spec_path)


def _choose_source_split_mode(total_classes, min_n_way=5, requested_mode='auto'):
    """Choose a valid source split protocol for the current class budget."""
    if requested_mode == 'class_disjoint':
        if total_classes < 3 * min_n_way:
            raise ValueError(
                f'class_disjoint source split requires at least {3 * min_n_way} classes '
                f'for {min_n_way}-way base/val/novel, but got {total_classes}.'
            )
        return SOURCE_CLASS_DISJOINT_MODE, 'forced by --source-split-mode=class_disjoint'

    if requested_mode == 'image_disjoint':
        return SOURCE_IMAGE_DISJOINT_MODE, 'forced by --source-split-mode=image_disjoint'

    if requested_mode != 'auto':
        raise ValueError(f'Unknown source split mode: {requested_mode}')

    if total_classes >= 3 * min_n_way:
        return SOURCE_CLASS_DISJOINT_MODE, (
            f'auto-selected class-disjoint mode: total_classes={total_classes} '
            f'>= 3 * n_way ({3 * min_n_way})'
        )

    return SOURCE_IMAGE_DISJOINT_MODE, (
        f'auto-selected image-disjoint shared-class mode: total_classes={total_classes} '
        f'< 3 * n_way ({3 * min_n_way})'
    )

def _load_complete_or_merged_data(dataset_path):
    """Load full dataset from all_data.json, or merge existing base/val/novel files."""
    all_data_file = os.path.join(dataset_path, 'all_data.json')

    if os.path.isfile(all_data_file):
        print(f"  📁 Loading complete data from: all_data.json")
        complete_data = load_json(all_data_file)
        return complete_data['image_names'], complete_data['image_labels']

    print(f"  ⚠️  all_data.json not found, merging existing splits...")
    json_files = [
        os.path.join(dataset_path, 'base.json'),
        os.path.join(dataset_path, 'val.json'),
        os.path.join(dataset_path, 'novel.json'),
    ]

    all_images = []
    all_labels = []
    for json_file in json_files:
        if os.path.isfile(json_file):
            data = load_json(json_file)
            all_images.extend(data.get('image_names', []))
            all_labels.extend(data.get('image_labels', []))

    return all_images, all_labels

def _backup_original_splits(data_dir, dataset_name, dataset_path):
    """Backup existing split files once before rewriting.

    Keep backups outside class-image root to avoid ImageFolder treating backup
    folders as category directories.
    """
    backup_dir_final = os.path.join(data_dir, '_split_backups', dataset_name, 'backup_original_splits')
    if os.path.exists(backup_dir_final):
        return

    os.makedirs(backup_dir_final, exist_ok=True)
    import shutil

    for split_name in ('base.json', 'val.json', 'novel.json'):
        src = os.path.join(dataset_path, split_name)
        if os.path.isfile(src):
            dst = os.path.join(backup_dir_final, split_name)
            shutil.copy2(src, dst)

    print(f"  📁 Backup saved to: {backup_dir_final}")

def _compute_class_disjoint_source_counts(total_classes, ratio, min_n_way=5):
    """Compute source class counts for a class-disjoint protocol."""
    total_ratio = sum(ratio)
    if total_classes < 3 * min_n_way:
        print(f"  ❌ Too few classes ({total_classes}) for class-disjoint {min_n_way}-way base/val/novel")
        return None

    if total_classes >= total_ratio:
        n_base, n_val, n_novel = ratio
    else:
        print(f"  ⚠️  Warning: {total_classes} classes < target {total_ratio}")
        print(f"  Falling back to minimum valid class-disjoint split...")
        n_val = min_n_way
        n_novel = min_n_way
        n_base = total_classes - n_val - n_novel

    if min(n_base, n_val, n_novel) < min_n_way:
        print(
            f"  ❌ Invalid class-disjoint source split: "
            f"base={n_base}, val={n_val}, novel={n_novel}, need each >= {min_n_way}"
        )
        return None

    return n_base, n_val, n_novel


def _allocate_integer_counts(total_count, ratio, min_each=0):
    """Allocate integer counts according to ratio while respecting a minimum per bucket."""
    if total_count < min_each * len(ratio):
        raise ValueError(
            f"Cannot allocate {len(ratio)} buckets with min_each={min_each} from total_count={total_count}"
        )

    weights = [max(0.0, float(x)) for x in ratio]
    if sum(weights) == 0:
        weights = [1.0] * len(ratio)

    counts = [min_each] * len(ratio)
    remaining = total_count - sum(counts)
    if remaining <= 0:
        return counts

    raw = [remaining * (w / sum(weights)) for w in weights]
    base = [int(x) for x in raw]
    counts = [c + b for c, b in zip(counts, base)]
    remainder = total_count - sum(counts)
    order = sorted(
        range(len(ratio)),
        key=lambda idx: (raw[idx] - base[idx], weights[idx]),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def _build_class_disjoint_source_splits(class_to_images, ratio, seed=42, min_n_way=5):
    """Build source splits with disjoint label spaces across base/val/novel."""
    all_classes = sorted(class_to_images.keys())
    total_classes = len(all_classes)
    split_counts = _compute_class_disjoint_source_counts(total_classes, ratio, min_n_way=min_n_way)
    if split_counts is None:
        return None

    n_base, n_val, n_novel = split_counts
    rng = random.Random(seed)
    rng.shuffle(all_classes)
    base_classes = all_classes[:n_base]
    val_classes = all_classes[n_base:n_base + n_val]
    novel_classes = all_classes[n_base + n_val:n_base + n_val + n_novel]

    def build_split(classes):
        images = []
        labels = []
        for cls in classes:
            images.extend(class_to_images[cls])
            labels.extend([cls] * len(class_to_images[cls]))
        return {'image_names': images, 'image_labels': labels}

    return {
        'mode': SOURCE_CLASS_DISJOINT_MODE,
        'base_classes': base_classes,
        'val_classes': val_classes,
        'novel_classes': novel_classes,
        'base': build_split(base_classes),
        'val': build_split(val_classes),
        'novel': build_split(novel_classes),
    }


def _build_image_disjoint_source_splits(class_to_images, ratio, seed=42):
    """Build source splits by splitting images within each class; labels overlap, images do not."""
    rng = random.Random(seed)
    split_names = ('base', 'val', 'novel')
    split_data = {
        split: {'image_names': [], 'image_labels': []}
        for split in split_names
    }
    per_class_counts = {}

    for cls in sorted(class_to_images.keys()):
        class_images = list(class_to_images[cls])
        rng.shuffle(class_images)
        min_each = 1 if len(class_images) >= len(split_names) else 0
        counts = _allocate_integer_counts(len(class_images), ratio, min_each=min_each)
        per_class_counts[cls] = dict(zip(split_names, counts))

        start = 0
        for split, count in zip(split_names, counts):
            selected = class_images[start:start + count]
            start += count
            split_data[split]['image_names'].extend(selected)
            split_data[split]['image_labels'].extend([cls] * len(selected))

    all_classes = sorted(class_to_images.keys())
    return {
        'mode': SOURCE_IMAGE_DISJOINT_MODE,
        'per_class_counts': per_class_counts,
        'base_classes': all_classes,
        'val_classes': all_classes,
        'novel_classes': all_classes,
        'base': split_data['base'],
        'val': split_data['val'],
        'novel': split_data['novel'],
    }


def _append_samples(split_data, split_name, image_paths, cls):
    split_data[split_name]['image_names'].extend(image_paths)
    split_data[split_name]['image_labels'].extend([cls] * len(image_paths))


def _build_dmpc_source_splits(class_to_images, source_train_ratio=0.8, seed=42):
    """
    Strict pairwise source protocol for this codebase:
    - train/base: image-disjoint subset with all classes
    - val: image-disjoint subset with all classes
    - novel: intentionally empty (source novel testing is not part of DMPC)

    This is only the paper-matched training protocol; it does not implement
    the referenced method itself. It is the closest protocol-compatible
    adaptation for the current StyleAdv training loop, which still needs a
    labeled validation split for checkpoint selection.
    """
    rng = random.Random(seed)
    split_data = {
        'base': {'image_names': [], 'image_labels': []},
        'val': {'image_names': [], 'image_labels': []},
        'novel': {'image_names': [], 'image_labels': []},
    }
    per_class_counts = {}
    all_classes = sorted(class_to_images.keys())

    for cls in all_classes:
        class_images = list(class_to_images[cls])
        rng.shuffle(class_images)
        counts = _allocate_integer_counts(len(class_images), [source_train_ratio, 1.0 - source_train_ratio], min_each=1)
        n_base, n_val = counts
        base_paths = class_images[:n_base]
        val_paths = class_images[n_base:n_base + n_val]
        _append_samples(split_data, 'base', base_paths, cls)
        _append_samples(split_data, 'val', val_paths, cls)
        per_class_counts[cls] = {'base': n_base, 'val': n_val, 'novel': 0}

    return {
        'mode': SOURCE_STRICT_PAIRWISE_MODE,
        'per_class_counts': per_class_counts,
        'base_classes': all_classes,
        'val_classes': all_classes,
        'novel_classes': [],
        'base': split_data['base'],
        'val': split_data['val'],
        'novel': split_data['novel'],
    }


def _build_dmpc_target_splits(class_to_images, unlabeled_ratio=0.2, seed=42):
    """
    Strict pairwise target protocol:
    - unlabeled: 20% target data (class-stratified, image-disjoint)
    - eval/novel: remaining 80% used for few-shot episodic evaluation
    - base/val stay empty for compatibility with existing code paths
    """
    rng = random.Random(seed)
    split_data = {
        'unlabeled': {'image_names': [], 'image_labels': []},
        'eval': {'image_names': [], 'image_labels': []},
        'base': {'image_names': [], 'image_labels': []},
        'val': {'image_names': [], 'image_labels': []},
        'novel': {'image_names': [], 'image_labels': []},
    }
    per_class_counts = {}
    all_classes = sorted(class_to_images.keys())
    unlabeled_indices = []
    eval_indices = []
    global_index = 0

    for cls in all_classes:
        class_images = list(class_to_images[cls])
        rng.shuffle(class_images)
        counts = _allocate_integer_counts(len(class_images), [unlabeled_ratio, 1.0 - unlabeled_ratio], min_each=1)
        n_unlabeled, n_eval = counts
        unlabeled_paths = class_images[:n_unlabeled]
        eval_paths = class_images[n_unlabeled:n_unlabeled + n_eval]

        _append_samples(split_data, 'unlabeled', unlabeled_paths, cls)
        _append_samples(split_data, 'eval', eval_paths, cls)
        _append_samples(split_data, 'novel', eval_paths, cls)

        unlabeled_indices.extend(range(global_index, global_index + n_unlabeled))
        global_index += n_unlabeled
        eval_indices.extend(range(global_index, global_index + n_eval))
        global_index += n_eval
        per_class_counts[cls] = {'unlabeled': n_unlabeled, 'eval': n_eval}

    return {
        'mode': TARGET_STRICT_PAIRWISE_MODE,
        'per_class_counts': per_class_counts,
        'all_classes': all_classes,
        'base': split_data['base'],
        'val': split_data['val'],
        'novel': split_data['novel'],
        'unlabeled': split_data['unlabeled'],
        'eval': split_data['eval'],
        'dkd_unlabeled_indices': unlabeled_indices,
        'dkd_eval_indices': eval_indices,
    }

def resplit_source_domain(
    data_dir,
    dataset_name='NWPU',
    ratio=(25, 10, 10),
    seed=42,
    min_n_way=5,
    source_split_mode='auto',
):
    """
    Resplit source domain with specified ratio.
    
    Args:
        data_dir: Root data directory
        dataset_name: Dataset name (default: NWPU)
        ratio: (base, val, novel) class ratio
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"Resplitting SOURCE domain: {dataset_name}")
    print(f"Target ratio: base={ratio[0]} : val={ratio[1]} : novel={ratio[2]}")
    print(f"{'='*70}")
    
    dataset_path = os.path.join(data_dir, dataset_name)
    
    all_data_file = os.path.join(dataset_path, 'all_data.json')
    print(f"  🔍 Checking for: {all_data_file}")
    print(f"  📂 File exists: {os.path.isfile(all_data_file)}")

    all_images, all_labels = _load_complete_or_merged_data(dataset_path)
    if not all_images:
        print(f"  ❌ No images found in {dataset_name}; skip.")
        return False
    
    # Group by class
    class_to_images = group_by_class(all_images, all_labels)
    all_classes = sorted(class_to_images.keys())
    total_classes = len(all_classes)
    
    print(f"  Total classes: {total_classes}")
    print(f"  Total images: {len(all_images)}")
    
    # Debug: print class range
    if all_classes:
        print(f"  Class range: {min(all_classes)} - {max(all_classes)}")
    
    split_mode, split_reason = _choose_source_split_mode(
        total_classes,
        min_n_way=min_n_way,
        requested_mode=source_split_mode,
    )
    print(f"  Source split mode: {split_mode}")
    print(f"  Reason: {split_reason}")

    if split_mode == SOURCE_CLASS_DISJOINT_MODE:
        split_result = _build_class_disjoint_source_splits(
            class_to_images,
            ratio,
            seed=seed,
            min_n_way=min_n_way,
        )
    else:
        split_result = _build_image_disjoint_source_splits(
            class_to_images,
            ratio,
            seed=seed,
        )

    if split_result is None:
        return False

    base_classes = split_result['base_classes']
    val_classes = split_result['val_classes']
    novel_classes = split_result['novel_classes']
    new_base = split_result['base']
    new_val = split_result['val']
    new_novel = split_result['novel']

    print(f"  New split: base={len(base_classes)}, val={len(val_classes)}, novel={len(novel_classes)}")
    
    _backup_original_splits(data_dir, dataset_name, dataset_path)
    
    # Save new splits to dataset root (not backup)
    final_base = os.path.join(dataset_path, 'base.json')
    final_val = os.path.join(dataset_path, 'val.json')
    final_novel = os.path.join(dataset_path, 'novel.json')
    save_json(new_base, final_base)
    save_json(new_val, final_val)
    save_json(new_novel, final_novel)
    save_split_spec(
        data_dir,
        dataset_name,
        {
            'protocol': PROTOCOL_LEGACY,
            'dataset': dataset_name,
            'role': 'source',
            'source_split_mode': split_mode,
            'class_disjoint': split_mode == SOURCE_CLASS_DISJOINT_MODE,
            'label_overlap_allowed': split_mode == SOURCE_IMAGE_DISJOINT_MODE,
            'reason': split_reason,
            'ratio': list(ratio),
            'n_way': min_n_way,
            'total_classes': total_classes,
        },
    )
    
    print(f"\n  📊 Final split:")
    print(f"     base:  {len(new_base['image_names']):5d} images, {len(base_classes):2d} classes")
    print(f"     val:   {len(new_val['image_names']):5d} images, {len(val_classes):2d} classes")
    print(f"     novel: {len(new_novel['image_names']):5d} images, {len(novel_classes):2d} classes")
    
    return True

def create_target_domain_novel_only(data_dir, dataset_name):
    """
    Create target domain as novel-only.

    For target domains: base.json and val.json are intentionally empty,
    novel.json contains all available classes/images.
    
    Args:
        data_dir: Root data directory
        dataset_name: Dataset name (e.g., EuroSAT, UCM, AID)
    """
    print(f"\n{'='*70}")
    print(f"Creating TARGET domain: {dataset_name}")
    print(f"Strategy: base/val empty, novel keeps ALL classes")
    print(f"{'='*70}")
    
    dataset_path = os.path.join(data_dir, dataset_name)
    
    all_images, all_labels = _load_complete_or_merged_data(dataset_path)
    
    if not all_images:
        print(f"  ⚠️  No existing JSON files found, skipping...")
        return False
    
    class_to_images = group_by_class(all_images, all_labels)
    total_classes = len(class_to_images)
    total_images = len(all_images)

    print(f"  Total images: {total_images}")
    print(f"  Total classes: {total_classes}")

    empty_data = {'image_names': [], 'image_labels': []}
    novel_data = {'image_names': all_images, 'image_labels': all_labels}
    
    _backup_original_splits(data_dir, dataset_name, dataset_path)
    
    # Save splits
    final_base = os.path.join(dataset_path, 'base.json')
    final_val = os.path.join(dataset_path, 'val.json')
    final_novel = os.path.join(dataset_path, 'novel.json')
    
    save_json(empty_data, final_base)
    save_json(empty_data, final_val)
    save_json(novel_data, final_novel)
    save_split_spec(
        data_dir,
        dataset_name,
        {
            'protocol': PROTOCOL_LEGACY,
            'dataset': dataset_name,
            'role': 'target',
            'target_split_mode': TARGET_NOVEL_ONLY_MODE,
            'class_disjoint': True,
            'label_overlap_allowed': False,
        },
    )
    
    print(f"\n  📊 Final structure:")
    print(f"     base.json:  0 images, 0 classes (target-domain empty)")
    print(f"     val.json:   0 images, 0 classes (target-domain empty)")
    print(f"     novel.json: {len(all_images)} images, {total_classes} classes (ALL DATA)")
    
    return True


def create_source_domain_dmpc(data_dir, dataset_name, seed=42, source_train_ratio=0.8):
    """Create strict pairwise source splits for the current codebase."""
    print(f"\n{'='*70}")
    print(f"Creating SOURCE domain (strict pairwise): {dataset_name}")
    print(f"Strategy: base/train and val are image-disjoint but share all classes")
    print(f"Source train ratio: {source_train_ratio:.2f}")
    print(f"{'='*70}")

    dataset_path = os.path.join(data_dir, dataset_name)
    all_images, all_labels = _load_complete_or_merged_data(dataset_path)
    if not all_images:
        print(f"  鈿狅笍  No images found in {dataset_name}; skip.")
        return False

    class_to_images = group_by_class(all_images, all_labels)
    total_classes = len(class_to_images)
    total_images = len(all_images)
    print(f"  Total images: {total_images}")
    print(f"  Total classes: {total_classes}")

    split_result = _build_dmpc_source_splits(
        class_to_images,
        source_train_ratio=source_train_ratio,
        seed=seed,
    )

    _backup_original_splits(data_dir, dataset_name, dataset_path)

    final_base = os.path.join(dataset_path, 'base.json')
    final_val = os.path.join(dataset_path, 'val.json')
    final_novel = os.path.join(dataset_path, 'novel.json')
    final_train = os.path.join(dataset_path, 'train.json')

    save_json(split_result['base'], final_base)
    save_json(split_result['base'], final_train)
    save_json(split_result['val'], final_val)
    save_json(split_result['novel'], final_novel)
    save_split_spec(
        data_dir,
        dataset_name,
        {
            'protocol': PROTOCOL_STRICT_PAIRWISE,
            'dataset': dataset_name,
            'role': 'source',
            'source_split_mode': SOURCE_STRICT_PAIRWISE_MODE,
            'class_disjoint': False,
            'label_overlap_allowed': True,
            'source_train_ratio': source_train_ratio,
            'total_classes': total_classes,
            'compatibility': {
                'base_json': 'train subset with all source classes',
                'val_json': 'validation subset with all source classes',
                'novel_json': 'empty by design under strict pairwise protocol',
                'train_json': 'same content as base.json',
            },
        },
    )

    print(f"\n  馃搳 Final strict pairwise source split:")
    print(f"     base/train: {len(split_result['base']['image_names']):5d} images, {len(split_result['base_classes']):2d} classes")
    print(f"     val:        {len(split_result['val']['image_names']):5d} images, {len(split_result['val_classes']):2d} classes")
    print(f"     novel:      {len(split_result['novel']['image_names']):5d} images, {len(split_result['novel_classes']):2d} classes")
    return True


def create_target_domain_dmpc(data_dir, dataset_name, seed=42, unlabeled_ratio=0.2):
    """Create strict pairwise target splits."""
    print(f"\n{'='*70}")
    print(f"Creating TARGET domain (strict pairwise): {dataset_name}")
    print(f"Strategy: unlabeled={unlabeled_ratio:.2f}, eval/novel={1.0 - unlabeled_ratio:.2f}, image-disjoint, shared classes")
    print(f"{'='*70}")

    dataset_path = os.path.join(data_dir, dataset_name)
    all_images, all_labels = _load_complete_or_merged_data(dataset_path)
    if not all_images:
        print(f"  鈿狅笍  No existing JSON files found, skipping...")
        return False

    class_to_images = group_by_class(all_images, all_labels)
    total_classes = len(class_to_images)
    total_images = len(all_images)
    print(f"  Total images: {total_images}")
    print(f"  Total classes: {total_classes}")

    split_result = _build_dmpc_target_splits(
        class_to_images,
        unlabeled_ratio=unlabeled_ratio,
        seed=seed,
    )

    _backup_original_splits(data_dir, dataset_name, dataset_path)

    final_base = os.path.join(dataset_path, 'base.json')
    final_val = os.path.join(dataset_path, 'val.json')
    final_novel = os.path.join(dataset_path, 'novel.json')
    final_unlabeled = os.path.join(dataset_path, 'unlabeled.json')
    final_eval = os.path.join(dataset_path, 'eval.json')

    save_json(split_result['base'], final_base)
    save_json(split_result['val'], final_val)
    save_json(split_result['novel'], final_novel)
    save_json(split_result['unlabeled'], final_unlabeled)
    save_json(split_result['eval'], final_eval)
    save_dkd_split(
        get_dkd_split_path(dataset_name, dataset_path),
        split_result['dkd_unlabeled_indices'],
        split_result['dkd_eval_indices'],
    )
    save_split_spec(
        data_dir,
        dataset_name,
        {
            'protocol': PROTOCOL_STRICT_PAIRWISE,
            'dataset': dataset_name,
            'role': 'target',
            'target_split_mode': TARGET_STRICT_PAIRWISE_MODE,
            'class_disjoint': False,
            'label_overlap_allowed': True,
            'target_unlabeled_ratio': unlabeled_ratio,
            'total_classes': total_classes,
            'compatibility': {
                'base_json': 'empty compatibility file',
                'val_json': 'empty compatibility file',
                'novel_json': 'same content as eval.json for episodic evaluation',
                'unlabeled_json': 'strict pairwise unlabeled target subset',
                'eval_json': 'strict pairwise target evaluation subset',
            },
        },
    )

    print(f"\n  馃搳 Final strict pairwise target structure:")
    print(f"     unlabeled.json: {len(split_result['unlabeled']['image_names']):5d} images, {len(split_result['all_classes']):2d} classes")
    print(f"     eval.json:      {len(split_result['eval']['image_names']):5d} images, {len(split_result['all_classes']):2d} classes")
    print(f"     novel.json:     {len(split_result['novel']['image_names']):5d} images, {len(split_result['all_classes']):2d} classes (compat)")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Restructure datasets for cross-domain few-shot experiments.'
    )
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Root directory of datasets',
    )
    parser.add_argument(
        '--source',
        default='NWPU',
        help='Source dataset name (e.g., NWPU, EuroSAT, AID, UCM)',
    )
    parser.add_argument(
        '--ratio',
        default='25,10,10',
        help='Source split class ratio as base,val,novel (default: 25,10,10)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for source-domain split',
    )
    parser.add_argument(
        '--n-way',
        type=int,
        default=5,
        help='Minimum classes required for source base/val episodes',
    )
    parser.add_argument(
        '--source-split-mode',
        default='auto',
        choices=['auto', 'class_disjoint', 'image_disjoint'],
        help='Source protocol: auto chooses class-disjoint when possible, otherwise image-disjoint shared-class splits',
    )
    parser.add_argument(
        '--protocol',
        default=PROTOCOL_LEGACY,
        choices=[PROTOCOL_LEGACY, PROTOCOL_STRICT_PAIRWISE, PROTOCOL_DMPC],
        help='Split protocol: legacy keeps old base/val/novel semantics; strict_pairwise uses source train/val + target unlabeled/eval semantics (dmpc kept as backward-compatible alias)',
    )
    parser.add_argument(
        '--source-train-ratio',
        type=float,
        default=0.8,
        help='Strict pairwise source image split ratio for base/train vs val',
    )
    parser.add_argument(
        '--target-unlabeled-ratio',
        type=float,
        default=0.2,
        help='Strict pairwise target unlabeled ratio',
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    source_domain = args.source
    try:
        ratio = tuple(int(x.strip()) for x in args.ratio.split(','))
        if len(ratio) != 3:
            raise ValueError
    except ValueError:
        print('❌ --ratio must be in format base,val,novel (e.g., 25,10,10)')
        sys.exit(1)
    
    print(f"\n{'#'*70}")
    print(f"# Dataset Restructuring for Cross-Domain Few-Shot Learning")
    print(f"# Data directory: {data_dir}")
    print(f"{'#'*70}")
    
    known_domains = ['NWPU', 'EuroSAT', 'UCM', 'AID', 'MSTAR']
    if source_domain not in known_domains:
        print(f"❌ Unknown source domain: {source_domain}")
        print(f"   Known domains: {', '.join(known_domains)}")
        sys.exit(1)

    # 1. Resplit source domain
    print(f"\n{'='*70}")
    print(f"STEP 1: Source Domain ({source_domain})")
    print(f"{'='*70}")
    source_path = os.path.join(data_dir, source_domain)
    if not os.path.isdir(source_path):
        print(f"  ❌ Source domain directory not found: {source_path}")
        sys.exit(1)
    if args.protocol in PAIRWISE_PROTOCOL_NAMES:
        create_source_domain_dmpc(
            data_dir,
            source_domain,
            seed=args.seed,
            source_train_ratio=args.source_train_ratio,
        )
    else:
        resplit_source_domain(
            data_dir,
            source_domain,
            ratio=ratio,
            seed=args.seed,
            min_n_way=args.n_way,
            source_split_mode=args.source_split_mode,
        )
    
    # 2. Create target domain novel-only structure
    print(f"\n{'='*70}")
    target_domains = [d for d in known_domains if d != source_domain]
    print(f"STEP 2: Target Domains ({', '.join(target_domains)})")
    print(f"{'='*70}")

    for dataset in target_domains:
        dataset_path = os.path.join(data_dir, dataset)
        if os.path.isdir(dataset_path):
            if args.protocol in PAIRWISE_PROTOCOL_NAMES:
                create_target_domain_dmpc(
                    data_dir,
                    dataset,
                    seed=args.seed,
                    unlabeled_ratio=args.target_unlabeled_ratio,
                )
            else:
                create_target_domain_novel_only(data_dir, dataset)
        else:
            print(f"\n{'='*70}")
            print(f"Dataset: {dataset}")
            print(f"{'='*70}")
            print(f"  ⚠️  Directory not found: {dataset_path}")
    
    print(f"\n{'#'*70}")
    print(f"# ✅ Restructuring Complete!")
    print(f"#")
    print(f"# Next steps:")
    print(f"#   1. Run: python3 check_dataset_splits.py {data_dir} --source {source_domain}")
    print(f"#   2. Verify the splits are correct")
    print(f"#   3. Start training: sbatch your_training_script.slurm")
    print(f"{'#'*70}\n")
