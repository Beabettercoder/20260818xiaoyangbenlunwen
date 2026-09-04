#!/usr/bin/env python3
"""Check dataset directory structure"""
import os
import sys

def check_structure(dataset_path, dataset_name):
    print(f"\n{'='*70}")
    print(f"Checking: {dataset_name}")
    print(f"Path: {dataset_path}")
    print(f"{'='*70}")
    
    if not os.path.isdir(dataset_path):
        print(f"  ❌ Directory not found")
        return
    
    # List all items
    items = os.listdir(dataset_path)
    print(f"\n  Top-level items ({len(items)}):")
    
    dirs = []
    files = []
    
    for item in sorted(items):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            # Count items in subdirectory
            try:
                subitems = os.listdir(item_path)
                subdirs = sum(1 for x in subitems if os.path.isdir(os.path.join(item_path, x)))
                subfiles = len(subitems) - subdirs
                dirs.append(f"    📁 {item}/ ({subdirs} dirs, {subfiles} files)")
            except:
                dirs.append(f"    📁 {item}/")
        else:
            files.append(f"    📄 {item}")
    
    for d in dirs:
        print(d)
    if files[:5]:  # Show first 5 files
        for f in files[:5]:
            print(f)
        if len(files) > 5:
            print(f"    ... and {len(files)-5} more files")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python scripts/check_dir_structure.py DATA_DIR')
    data_dir = os.path.abspath(os.path.expanduser(sys.argv[1]))
    
    for ds in ['NWPU', 'UCM']:
        check_structure(os.path.join(data_dir, ds), ds)
