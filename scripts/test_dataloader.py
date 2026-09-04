#!/usr/bin/env python3
"""
Test if dataloaders can successfully load the datasets.
This script simulates the actual training/testing process.
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from data.datamgr import SetDataManager
from options import parse_args

def test_source_domain_training(params):
    """Test if source domain can be loaded for training."""
    print(f"\n{'='*70}")
    print("Testing SOURCE DOMAIN Training Data Loading")
    print(f"{'='*70}")
    
    print(f"  Dataset: {params.dataset}")
    print(f"  Mode: meta-train (base classes)")
    print(f"  n_way: {params.train_n_way}")
    print(f"  n_shot: {params.n_shot}")
    print(f"  n_query: {params.train_n_query}")
    
    try:
        train_datamgr = SetDataManager(
            params.image_size,
            n_way=params.train_n_way,
            n_support=params.n_shot,
            n_query=params.train_n_query,
            n_eposide=100,  # Test with 100 episodes
            data_root=params.data_dir,
            dataset_name=params.dataset,
        )
        
        train_loader = train_datamgr.get_data_loader(
            os.path.join(params.data_dir, params.dataset, 'base.json'),
            aug=params.train_aug
        )
        
        print(f"  ✅ Base dataloader created successfully")
        print(f"     Total episodes: {len(train_loader)}")
        
        # Try to load one batch
        for i, (x, _) in enumerate(train_loader):
            print(f"  ✅ First batch loaded: {x.shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to load training data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_source_domain_validation(params):
    """Test if source domain can be loaded for validation."""
    print(f"\n{'='*70}")
    print("Testing SOURCE DOMAIN Validation Data Loading")
    print(f"{'='*70}")
    
    print(f"  Dataset: {params.dataset}")
    print(f"  Mode: meta-val (val classes)")
    print(f"  n_way: {params.test_n_way}")
    print(f"  n_shot: {params.n_shot}")
    print(f"  n_query: {params.n_query}")
    
    try:
        val_datamgr = SetDataManager(
            params.image_size,
            n_way=params.test_n_way,
            n_support=params.n_shot,
            n_query=params.n_query,
            n_eposide=100,
            data_root=params.data_dir,
            dataset_name=params.dataset,
        )
        
        val_loader = val_datamgr.get_data_loader(
            os.path.join(params.data_dir, params.dataset, 'val.json'),
            aug=False
        )
        
        print(f"  ✅ Val dataloader created successfully")
        print(f"     Total episodes: {len(val_loader)}")
        
        # Try to load one batch
        for i, (x, _) in enumerate(val_loader):
            print(f"  ✅ First batch loaded: {x.shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to load validation data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_target_domain(params, target_dataset):
    """Test if target domain can be loaded for testing."""
    print(f"\n{'='*70}")
    print(f"Testing TARGET DOMAIN: {target_dataset}")
    print(f"{'='*70}")
    
    print(f"  Mode: meta-test (novel classes)")
    print(f"  n_way: {params.test_n_way}")
    print(f"  n_shot: {params.n_shot}")
    print(f"  n_query: {params.n_query}")
    
    try:
        test_datamgr = SetDataManager(
            params.image_size,
            n_way=params.test_n_way,
            n_support=params.n_shot,
            n_query=params.n_query,
            n_eposide=600,
            data_root=params.data_dir,
            dataset_name=target_dataset,
        )
        
        novel_file = os.path.join(params.data_dir, target_dataset, 'novel.json')
        
        if not os.path.isfile(novel_file):
            print(f"  ❌ novel.json not found: {novel_file}")
            return False
        
        test_loader = test_datamgr.get_data_loader(novel_file, aug=False)
        
        print(f"  ✅ Test dataloader created successfully")
        print(f"     Total episodes: {len(test_loader)}")
        
        # Try to load one batch
        for i, (x, _) in enumerate(test_loader):
            print(f"  ✅ First batch loaded: {x.shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to load test data: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_data_leakage_fix(script_path):
    """Check if data leakage fix is applied."""
    print(f"\n{'='*70}")
    print("Checking Data Leakage Fix")
    print(f"{'='*70}")
    
    issues = []
    
    # Check metatrain script
    train_script = os.path.join(script_path, 'metatrain_StyleAdv_RN.py')
    if os.path.isfile(train_script):
        with open(train_script, 'r') as f:
            content = f.read()
            if 'val_file = novel_file' in content:
                issues.append(f"  ❌ {train_script}: Data leakage still present (val_file = novel_file)")
            else:
                print(f"  ✅ {train_script}: No data leakage detected")
    
    # Check test script
    test_script = os.path.join(script_path, 'test_function_fwt_benchmark.py')
    if os.path.isfile(test_script):
        with open(test_script, 'r') as f:
            content = f.read()
            if 'val_file = novel_file' in content:
                issues.append(f"  ❌ {test_script}: Data leakage still present (val_file = novel_file)")
            else:
                print(f"  ✅ {test_script}: No data leakage detected")
    
    if issues:
        print("\n  Issues found:")
        for issue in issues:
            print(issue)
        return False
    
    return True

if __name__ == '__main__':
    print(f"\n{'#'*70}")
    print("# Dataloader Integration Test")
    print(f"{'#'*70}")
    
    # Parse arguments with defaults
    import argparse
    parser = argparse.ArgumentParser()
    
    # Dataset parameters
    parser.add_argument('--dataset', default='NWPU', help='Source dataset name')
    parser.add_argument('--data_dir', required=True, help='Data directory')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    
    # Few-shot parameters
    parser.add_argument('--train_n_way', type=int, default=5, help='N-way for training')
    parser.add_argument('--test_n_way', type=int, default=5, help='N-way for testing')
    parser.add_argument('--n_shot', type=int, default=5, help='N-shot')
    parser.add_argument('--train_n_query', type=int, default=6, help='N-query for training')
    parser.add_argument('--n_query', type=int, default=15, help='N-query for testing')
    parser.add_argument('--train_aug', action='store_true', default=True, help='Training augmentation')
    
    params = parser.parse_args()
    
    all_passed = True
    
    # 1. Check data leakage fix
    if not check_data_leakage_fix(REPO_ROOT):
        all_passed = False
        print("\n  ⚠️  Warning: Data leakage fix not applied! Training may use test data for validation.")
    
    # 2. Test source domain training
    if not test_source_domain_training(params):
        all_passed = False
    
    # 3. Test source domain validation
    if not test_source_domain_validation(params):
        all_passed = False
    
    # 4. Test target domains
    target_datasets = ['EuroSAT', 'UCM', 'AID']
    for target in target_datasets:
        target_path = os.path.join(params.data_dir, target)
        if os.path.isdir(target_path):
            if not test_target_domain(params, target):
                all_passed = False
    
    # Final summary
    print(f"\n{'#'*70}")
    if all_passed:
        print("# ✅ ALL CHECKS PASSED!")
        print("#")
        print("# Your datasets are correctly configured and can be loaded.")
        print("# You can now start training with:")
        print("#   bash scripts/run_train.sh DATA_DIR --source_dataset NWPU --target_dataset EuroSAT")
    else:
        print("# ❌ SOME CHECKS FAILED")
        print("#")
        print("# Please fix the issues above before training.")
    print(f"{'#'*70}\n")
    
    sys.exit(0 if all_passed else 1)
