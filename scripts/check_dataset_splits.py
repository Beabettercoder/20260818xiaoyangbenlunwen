#!/usr/bin/env python3
"""Check if dataset splits have sufficient classes for n-way few-shot learning."""

import json
import os
import sys
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SPLIT_SPEC_FILENAME = 'split_spec.json'
SOURCE_CLASS_DISJOINT_MODE = 'class_disjoint'
SOURCE_IMAGE_DISJOINT_MODE = 'image_disjoint_shared_classes'
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

def count_classes(json_path):
    """Count unique classes in a dataset split."""
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            meta = json.load(f)
        return len(set(meta.get('image_labels', [])))
    except Exception as e:
        return None


def load_split_spec(data_dir, dataset_name):
    spec_path = os.path.join(data_dir, dataset_name, SPLIT_SPEC_FILENAME)
    if not os.path.isfile(spec_path):
        return {}
    try:
        with open(spec_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def check_dataset(data_dir, dataset_name, n_way=5, is_source=False):
    """
    Check if a dataset has sufficient splits for training.
    
    Args:
        data_dir: Root data directory
        dataset_name: Dataset name
        n_way: Required number of classes for few-shot learning
        is_source: True for source domain (needs base/val), False for target domain (only needs novel)
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} ({'SOURCE' if is_source else 'TARGET'})")
    print(f"{'='*60}")
    
    base_file = os.path.join(data_dir, dataset_name, 'base.json')
    val_file = os.path.join(data_dir, dataset_name, 'val.json')
    novel_file = os.path.join(data_dir, dataset_name, 'novel.json')
    unlabeled_file = os.path.join(data_dir, dataset_name, 'unlabeled.json')
    eval_file = os.path.join(data_dir, dataset_name, 'eval.json')
    split_spec = load_split_spec(data_dir, dataset_name)
    protocol = split_spec.get('protocol', PROTOCOL_LEGACY)
    source_mode = split_spec.get('source_split_mode')
    target_mode = split_spec.get('target_split_mode')
    
    base_classes = count_classes(base_file)
    val_classes = count_classes(val_file)
    novel_classes = count_classes(novel_file)
    unlabeled_classes = count_classes(unlabeled_file)
    eval_classes = count_classes(eval_file)
    
    print(f"  base.json:  {base_classes if base_classes is not None else 'NOT FOUND'} classes")
    print(f"  val.json:   {val_classes if val_classes is not None else 'NOT FOUND'} classes")
    print(f"  novel.json: {novel_classes if novel_classes is not None else 'NOT FOUND'} classes")
    if protocol in PAIRWISE_PROTOCOL_NAMES:
        print(f"  protocol:   {protocol}")
    if (not is_source) and target_mode in TARGET_PAIRWISE_MODE_NAMES:
        print(f"  unlabeled.json: {unlabeled_classes if unlabeled_classes is not None else 'NOT FOUND'} classes")
        print(f"  eval.json:      {eval_classes if eval_classes is not None else 'NOT FOUND'} classes")
    if is_source and source_mode:
        print(f"  split_spec: source_split_mode={source_mode}")
    
    # Check requirements
    issues = []
    
    if is_source:
        require_source_novel = source_mode in (SOURCE_CLASS_DISJOINT_MODE, SOURCE_IMAGE_DISJOINT_MODE)
        if protocol in PAIRWISE_PROTOCOL_NAMES or source_mode in SOURCE_PAIRWISE_MODE_NAMES:
            require_source_novel = False

        # Source domain: needs base and val; for explicit modern protocols novel is also mandatory.
        if base_classes is None:
            issues.append("❌ base.json missing")
        elif base_classes < n_way:
            issues.append(f"❌ base.json has only {base_classes} classes (need ≥{n_way})")
        else:
            print(f"  ✅ base.json OK ({base_classes} ≥ {n_way})")
        
        if val_classes is None:
            issues.append("❌ val.json missing")
        elif val_classes < n_way:
            issues.append(f"❌ val.json has only {val_classes} classes (need ≥{n_way})")
        else:
            print(f"  ✅ val.json OK ({val_classes} ≥ {n_way})")
        
        if protocol in PAIRWISE_PROTOCOL_NAMES or source_mode in SOURCE_PAIRWISE_MODE_NAMES:
            if novel_classes in (None, 0):
                print("  ✅ novel.json empty (expected for strict pairwise source protocol)")
            elif novel_classes >= n_way:
                print(f"  ✅ novel.json present ({novel_classes} ≥ {n_way})")
            else:
                issues.append(f"❌ novel.json has only {novel_classes} classes (need 0 or ≥{n_way})")
        if require_source_novel:
            if novel_classes is None:
                issues.append("❌ novel.json missing")
            elif novel_classes < n_way:
                issues.append(f"❌ novel.json has only {novel_classes} classes (need ≥{n_way})")
            else:
                print(f"  ✅ novel.json OK ({novel_classes} ≥ {n_way})")
        elif novel_classes is not None and novel_classes >= n_way:
            print(f"  ✅ novel.json OK ({novel_classes} ≥ {n_way})")
    else:
        # Target domain: only needs novel.json, base/val should be empty
        if base_classes is not None and base_classes > 0:
            issues.append(f"⚠️  base.json should be empty for target domain (has {base_classes} classes)")
        elif base_classes == 0:
            print(f"  ✅ base.json empty (correct for target domain)")
        
        if val_classes is not None and val_classes > 0:
            issues.append(f"⚠️  val.json should be empty for target domain (has {val_classes} classes)")
        elif val_classes == 0:
            print(f"  ✅ val.json empty (correct for target domain)")
        
        if novel_classes is None:
            issues.append("❌ novel.json missing")
        elif novel_classes < n_way:
            issues.append(f"❌ novel.json has only {novel_classes} classes (need ≥{n_way})")
        else:
            print(f"  ✅ novel.json OK ({novel_classes} ≥ {n_way})")

        if protocol in PAIRWISE_PROTOCOL_NAMES or target_mode in TARGET_PAIRWISE_MODE_NAMES:
            if unlabeled_classes is None:
                issues.append("❌ unlabeled.json missing for strict pairwise target protocol")
            elif unlabeled_classes < n_way:
                issues.append(f"❌ unlabeled.json has only {unlabeled_classes} classes (need ≥{n_way})")
            else:
                print(f"  ✅ unlabeled.json OK ({unlabeled_classes} ≥ {n_way})")

            if eval_classes is None:
                issues.append("❌ eval.json missing for strict pairwise target protocol")
            elif eval_classes < n_way:
                issues.append(f"❌ eval.json has only {eval_classes} classes (need ≥{n_way})")
            else:
                print(f"  ✅ eval.json OK ({eval_classes} ≥ {n_way})")
    
    if issues:
        print("\n  Issues:")
        for issue in issues:
            print(f"    {issue}")
        return False
    else:
        print(f"\n  ✅ Dataset structure is correct!")
        return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Check dataset split validity for cross-domain few-shot protocol.'
    )
    parser.add_argument(
        'data_dir',
        help='Root directory of datasets',
    )
    parser.add_argument(
        '--n-way',
        type=int,
        default=5,
        help='Required number of classes for n-way episodes',
    )
    parser.add_argument(
        '--source',
        default='NWPU',
        help='Source dataset name (e.g., NWPU, EuroSAT, AID, UCM)',
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    n_way = args.n_way
    source_domain = args.source
    
    print(f"\nChecking datasets in: {data_dir}")
    print(f"Required n_way: {n_way}")
    
    known_domains = ['NWPU', 'EuroSAT', 'UCM', 'AID', 'MSTAR']
    if source_domain not in known_domains:
        print(f"❌ Unknown source domain: {source_domain}")
        print(f"Known domains: {', '.join(known_domains)}")
        sys.exit(1)

    source_domains = [source_domain]
    target_domains = [d for d in known_domains if d != source_domain]
    
    all_valid = True
    
    # Check source domains
    for dataset in source_domains:
        dataset_path = os.path.join(data_dir, dataset)
        if os.path.isdir(dataset_path):
            valid = check_dataset(data_dir, dataset, n_way, is_source=True)
            all_valid = all_valid and valid
        else:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset} (SOURCE)")
            print(f"{'='*60}")
            print(f"  ⚠️  Directory not found: {dataset_path}")
            all_valid = False
    
    # Check target domains
    for dataset in target_domains:
        dataset_path = os.path.join(data_dir, dataset)
        if os.path.isdir(dataset_path):
            valid = check_dataset(data_dir, dataset, n_way, is_source=False)
            all_valid = all_valid and valid
        else:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset} (TARGET)")
            print(f"{'='*60}")
            print(f"  ⚠️  Directory not found: {dataset_path}")
    
    print(f"\n{'='*60}")
    if all_valid:
        print("✅ All datasets are valid!")
    else:
        print("❌ Some datasets have issues - see above")
    print(f"{'='*60}\n")
    
    sys.exit(0 if all_valid else 1)
