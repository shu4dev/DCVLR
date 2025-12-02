#!/usr/bin/env python3
"""
Test script to verify that the pipeline correctly loads images from train folders.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from src.filtering.binning import ImageBinner

def test_load_images_from_train_folders(data_dir: str = "./data"):
    """
    Test loading images from train folders.

    Expected structure:
    data/
        HuggingFaceM4__ChartQA/
            train/
                image1.jpg
        derek-thomas__ScienceQA/
            train/
                image2.jpg
        ...
    """
    print("=" * 60)
    print("Testing image loading from train folders")
    print("=" * 60)
    print(f"\nData directory: {data_dir}")

    # Check if data directory exists
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Error: Data directory '{data_dir}' does not exist")
        return

    # List subdirectories
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"\nFound {len(subdirs)} subdirectories:")
    for subdir in subdirs:
        train_folder = subdir / "train"
        has_train = "✓" if train_folder.exists() else "✗"
        print(f"  {has_train} {subdir.name}/")
        if train_folder.exists():
            img_count = len(list(train_folder.glob("*.jpg"))) + \
                       len(list(train_folder.glob("*.jpeg"))) + \
                       len(list(train_folder.glob("*.png")))
            print(f"    → train/ contains {img_count} images")

    print("\n" + "=" * 60)
    print("Loading images using ImageBinner.load_images_from_train_folders()")
    print("=" * 60)

    # Load images
    images = ImageBinner.load_images_from_train_folders(data_dir)

    print(f"\n✓ Successfully loaded {len(images)} images")

    # Group by dataset
    datasets = {}
    for img in images:
        dataset = img['dataset']
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(img)

    print(f"\nImages per dataset:")
    for dataset, imgs in sorted(datasets.items()):
        print(f"  {dataset}: {len(imgs)} images")

    # Show sample images
    if images:
        print(f"\nSample images (first 5):")
        for i, img in enumerate(images[:5], 1):
            print(f"  {i}. Dataset: {img['dataset']}")
            print(f"     File: {img['filename']}")
            print(f"     Path: {img['path']}")
            print()

    print("=" * 60)
    print("Test complete!")
    print("=" * 60)

    return images


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test loading images from train folders"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Data directory containing dataset folders (default: ./data)'
    )

    args = parser.parse_args()

    test_load_images_from_train_folders(args.data_dir)
