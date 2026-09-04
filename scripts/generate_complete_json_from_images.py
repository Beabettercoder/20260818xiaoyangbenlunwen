#!/usr/bin/env python3
"""
Generate complete JSON files from image directories.
This script scans the actual image folders to create complete dataset JSONs.
"""

import os
import json
import sys
from pathlib import Path
from collections import defaultdict

def save_json(data, json_path):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  ✅ Saved: {json_path}")

def scan_dataset_structure(dataset_path, dataset_name):
    """
    Scan dataset directory to find all images and their classes.
    
    Assumes structure like:
    dataset_path/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img3.jpg
    """
    print(f"\n{'='*70}")
    print(f"Scanning: {dataset_name}")
    print(f"{'='*70}")
    
    if not os.path.isdir(dataset_path):
        print(f"  ❌ Directory not found: {dataset_path}")
        return None
    
    # Find all subdirectories (classes)
    class_dirs = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            # Skip special directories
            if item in ['backup_original_splits', 'splits', '__pycache__']:
                continue
            class_dirs.append(item)
    
    if not class_dirs:
        print(f"  ⚠️  No class directories found")
        return None
    
    class_dirs.sort()
    print(f"  Found {len(class_dirs)} class directories")
    
    # Scan images in each class
    class_to_images = {}
    total_images = 0
    
    for class_idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(dataset_path, class_name)
        
        # Find all image files
        images = []
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG', '.TIF', '.TIFF']:
            images.extend([f for f in os.listdir(class_path) if f.endswith(ext)])
        
        if images:
            # Store absolute paths (dataset_path + class_name + img)
            abs_paths = [os.path.join(dataset_path, class_name, img) for img in images]
            class_to_images[class_idx] = abs_paths
            total_images += len(images)
            print(f"    Class {class_idx:3d} ({class_name}): {len(images):4d} images")
    
    print(f"\n  Total: {len(class_to_images)} classes, {total_images} images")
    
    # Create full dataset JSON
    all_images = []
    all_labels = []
    
    for class_idx in sorted(class_to_images.keys()):
        for img_path in class_to_images[class_idx]:
            all_images.append(img_path)
            all_labels.append(class_idx)
    
    return {
        'image_names': all_images,
        'image_labels': all_labels,
    }

def generate_dataset_json(data_dir, dataset_name, image_subdir=None):
    """
    Generate complete JSON for a dataset.
    
    Args:
        data_dir: Root data directory
        dataset_name: Dataset name
        image_subdir: Subdirectory containing images (e.g., 'images', 'Images')
                     If None, will try common names
    """
    dataset_root = os.path.join(data_dir, dataset_name)
    
    # Try to find image directory
    if image_subdir:
        image_path = os.path.join(dataset_root, image_subdir)
    else:
        # Try common directory names
        for possible_name in ['images', 'Images', dataset_name, '']:
            test_path = os.path.join(dataset_root, possible_name) if possible_name else dataset_root
            if os.path.isdir(test_path):
                # Check if this directory has subdirectories (classes)
                has_subdirs = any(
                    os.path.isdir(os.path.join(test_path, item)) 
                    and item not in ['backup_original_splits', 'splits', '__pycache__']
                    for item in os.listdir(test_path)
                )
                if has_subdirs:
                    image_path = test_path
                    if possible_name:
                        print(f"  Using image directory: {possible_name}/")
                    break
        else:
            print(f"  ❌ Could not find image directory")
            return False
    
    # Scan the directory
    data = scan_dataset_structure(image_path, dataset_name)
    
    if data is None:
        return False
    
    # Save complete JSON
    output_file = os.path.join(dataset_root, 'all_data.json')
    save_json(data, output_file)
    
    print(f"\n  📊 Generated complete dataset JSON")
    print(f"     File: all_data.json")
    print(f"     Images: {len(data['image_names'])}")
    print(f"     Classes: {len(set(data['image_labels']))}")
    
    return True

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(
            'Usage: python scripts/generate_complete_json_from_images.py DATA_DIR'
        )
    data_dir = os.path.abspath(os.path.expanduser(sys.argv[1]))
    
    print(f"\n{'#'*70}")
    print(f"# Generate Complete JSON from Image Directories")
    print(f"# Data directory: {data_dir}")
    print(f"{'#'*70}")
    
    # Dataset configurations
    # Empty string '' means class folders are in dataset root
    datasets = {
        'NWPU': '',            # NWPU/class_folders/ (45 classes, 700 images each)
        'EuroSAT': '',         # EuroSAT/class_folders/ (10 classes)
        'UCM': '',             # UCM/class_folders/ (21 classes, 100 images each)
        'AID': '',             # AID/class_folders/ (30 classes)
        'MSTAR': '',           # MSTAR/class_folders/ (if exists)
    }
    
    for dataset_name, image_subdir in datasets.items():
        dataset_path = os.path.join(data_dir, dataset_name)
        if os.path.isdir(dataset_path):
            generate_dataset_json(data_dir, dataset_name, image_subdir)
        else:
            print(f"\n{'='*70}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*70}")
            print(f"  ⚠️  Directory not found: {dataset_path}")
    
    print(f"\n{'#'*70}")
    print(f"# ✅ Complete!")
    print(f"#")
    print(f"# Generated files: <dataset>/all_data.json")
    print(f"# Next: Run restructure_datasets.py to split the data")
    print(f"{'#'*70}\n")
