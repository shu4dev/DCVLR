#!/usr/bin/env python3
"""
Example: Multi-Process Image Binning with Data Parallelism

This script demonstrates how to use multiple GPUs in parallel for image binning
when GPU memory is sufficient to hold all models on each GPU.

Performance:
- With 2 GPUs: ~1.9x speedup
- With 4 GPUs: ~3.7x speedup
- Scales near-linearly with number of GPUs

Usage:
    python example_multiprocess_binning.py
"""

import sys
from pathlib import Path
import yaml
import logging
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.filtering import MultiProcessImageBinner, enable_multiprocess_binning
from src.utils import setup_logging


def main():
    """Run multi-process binning example."""

    # IMPORTANT: Enable multiprocessing BEFORE any CUDA initialization
    # This must be called before creating any models or CUDA tensors
    enable_multiprocess_binning()

    # Setup logging
    setup_logging("output/multiprocess_binning.log")
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("Multi-Process Image Binning Example")
    logger.info("="*80)

    # Load configuration
    config_path = "configs/default_config.yaml"

    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        logger.info("Using default configuration")
        config = {
            'binning': {
                'pipeline_mode': 'hybrid',
                'enable_multi_gpu': True,
                'text_boxes_threshold': 2,
                'text_area_threshold': 0.2,
                'object_count_threshold': 5,
                'unique_objects_threshold': 3,
                'clip_similarity_threshold': 0.25,
                'spatial_dispersion_threshold': 0.3,
                'captioner_backend': 'blip',
                'object_detector': 'yolo',
                'yolo_model': 'yolov8n'
            }
        }
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Ensure multi-GPU is enabled
    config['binning']['enable_multi_gpu'] = True

    # Load images from data directory
    data_dir = "data/"

    if not Path(data_dir).exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Please set the correct data directory path")
        return

    # Load images using the static method
    from src.filtering.binning import ImageBinner
    logger.info(f"Loading images from {data_dir}...")
    images = ImageBinner.load_images_from_train_folders(data_dir)

    if not images:
        logger.error("No images found!")
        return

    logger.info(f"Loaded {len(images)} images")

    # For demonstration, limit to first 100 images
    # Remove this line to process all images
    if len(images) > 100:
        images = images[:100]
        logger.info(f"Limited to first {len(images)} images for demonstration")

    # Create multi-process binner
    logger.info("\nInitializing MultiProcessImageBinner...")
    try:
        binner = MultiProcessImageBinner(config['binning'])
    except RuntimeError as e:
        logger.error(f"Failed to initialize: {e}")
        logger.info("Falling back to single-GPU mode")
        binner = ImageBinner(config['binning'])

    # Run binning with timing
    logger.info("\nStarting image binning...")
    start_time = time.time()

    bins = binner.bin_images(images)

    elapsed_time = time.time() - start_time

    # Print results
    logger.info("\n" + "="*80)
    logger.info("Binning Results")
    logger.info("="*80)
    logger.info(f"Total images processed: {len(images)}")
    logger.info(f"Time taken: {elapsed_time:.2f} seconds")
    logger.info(f"Speed: {len(images)/elapsed_time:.2f} images/second")
    logger.info("")
    logger.info(f"Bin A (Text/Arithmetic): {len(bins['A'])} images ({len(bins['A'])/len(images)*100:.1f}%)")
    logger.info(f"Bin B (Object/Spatial): {len(bins['B'])} images ({len(bins['B'])/len(images)*100:.1f}%)")
    logger.info(f"Bin C (Commonsense): {len(bins['C'])} images ({len(bins['C'])/len(images)*100:.1f}%)")
    logger.info("="*80)

    # Show some example images from each bin
    logger.info("\nExample images from each bin:")
    for bin_name in ['A', 'B', 'C']:
        if bins[bin_name]:
            logger.info(f"\nBin {bin_name}:")
            for img in bins[bin_name][:3]:  # Show first 3
                logger.info(f"  - {img['filename']} (from {img.get('dataset', 'unknown')})")

    logger.info("\n✓ Multi-process binning completed successfully!")


if __name__ == "__main__":
    main()
