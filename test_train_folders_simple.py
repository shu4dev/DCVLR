#!/usr/bin/env python3
"""
Simple test script to verify image loading from train folders without dependencies.
"""

from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_images_from_train_folders(data_dir: str):
    """
    Load all images from train folders within subdirectories of data_dir.

    Expected structure:
    data_dir/
        folder1/
            train/
                image1.jpg
                image2.png
                ...
        folder2/
            train/
                image3.jpg
                ...
        ...

    Args:
        data_dir: Path to the data directory containing subdirectories with train folders

    Returns:
        List of dictionaries with image metadata:
        [{'path': '/path/to/image.jpg', 'dataset': 'folder1', 'filename': 'image1.jpg'}, ...]
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return []

    if not data_path.is_dir():
        logger.error(f"Data path is not a directory: {data_dir}")
        return []

    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    images = []

    # Iterate through subdirectories in data_dir
    for subfolder in data_path.iterdir():
        if not subfolder.is_dir():
            continue

        # Look for train folder within this subfolder
        train_folder = subfolder / 'train'

        if not train_folder.exists() or not train_folder.is_dir():
            logger.warning(f"No train folder found in {subfolder.name}, skipping...")
            continue

        logger.info(f"Loading images from {subfolder.name}/train/")

        # Load all images from train folder
        img_count = 0
        for img_path in train_folder.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                images.append({
                    'path': str(img_path.absolute()),
                    'dataset': subfolder.name,
                    'filename': img_path.name
                })
                img_count += 1

        logger.info(f"  Loaded {img_count} images from {subfolder.name}/train/")

    logger.info(f"Total images loaded: {len(images)} from {len(set(img['dataset'] for img in images))} datasets")

    return images


def main(data_dir: str = "./data"):
    """Main test function."""
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
            img_count = sum(1 for f in train_folder.iterdir()
                          if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'})
            print(f"    → train/ contains {img_count} images")

    print("\n" + "=" * 60)
    print("Loading images from train folders")
    print("=" * 60)

    # Load images
    images = load_images_from_train_folders(data_dir)

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

    main(args.data_dir)
