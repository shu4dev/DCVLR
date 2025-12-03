#!/usr/bin/env python3
"""
Optimized pipeline script for multi-GPU systems with sufficient VRAM.

This script uses:
- BatchedImageFilter for Stage 1 (5x faster filtering)
- MultiProcessImageBinner for Stage 2 (2x faster binning)

Expected speedup with 2 GPUs:
- Stage 1: 10.6s → 2.1s (5x faster)
- Stage 2: 35s → 18s (2x faster)
- Total: ~45s → ~20s (2.25x faster)

Usage:
    python scripts/run_pipeline_optimized.py --images-dir ./data --num-images 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# IMPORTANT: Enable multiprocessing BEFORE any other imports
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from team1_pipeline_optimized import OptimizedDataSynthesisPipeline


def main():
    """Main entry point for running the optimized pipeline."""

    parser = argparse.ArgumentParser(
        description="Optimized Team-1 Data Synthesis Pipeline - Multi-GPU acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 100 images with optimization
  python scripts/run_pipeline_optimized.py --images-dir ./data --num-images 100

  # Process all images
  python scripts/run_pipeline_optimized.py --images-dir ./data --num-images -1

  # Custom output directory
  python scripts/run_pipeline_optimized.py --images-dir ./data --num-images 100 --output-dir ./results
        """
    )

    # Required arguments
    parser.add_argument(
        '--images-dir',
        type=str,
        required=True,
        help='Directory containing input images'
    )

    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/',
        help='Output directory for results (default: output/)'
    )

    parser.add_argument(
        '--num-images',
        type=int,
        default=100,
        help='Number of images to process (default: 100, -1 for all images)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Configuration file path (default: configs/default_config.yaml)'
    )

    parser.add_argument(
        '--bins-ratio',
        type=float,
        nargs=3,
        default=[0.4, 0.4, 0.2],
        help='Bin ratios for Text:Object:Commonsense (default: 0.4 0.4 0.2)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Validate arguments
    if not Path(args.images_dir).exists():
        print(f"Error: Images directory '{args.images_dir}' does not exist")
        sys.exit(1)

    # Validate bins ratio
    bins_sum = sum(args.bins_ratio)
    if abs(bins_sum - 1.0) > 0.01:
        print(f"Warning: Bins ratio sums to {bins_sum}, normalizing to 1.0")
        args.bins_ratio = [r/bins_sum for r in args.bins_ratio]

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    # Print configuration
    logger.info("=" * 80)
    logger.info("OPTIMIZED Team-1 Data Synthesis Pipeline (Multi-GPU)")
    logger.info("=" * 80)
    logger.info(f"Images directory: {args.images_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of images: {args.num_images}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Bins ratio: {args.bins_ratio}")

    # Check GPU availability
    import torch
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.error("No GPUs detected! This optimized pipeline requires GPUs.")
        sys.exit(1)

    logger.info(f"GPUs detected: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        logger.info(f"  GPU {i}: {props.name} ({mem_gb:.1f}GB)")

    logger.info("")
    logger.info("Optimizations enabled:")
    logger.info("  ✓ Stage 1: Batched filtering (5x faster)")
    logger.info("  ✓ Stage 2: Multi-process binning (2x faster)")
    logger.info("  → Expected speedup: ~2.25x overall")
    logger.info("=" * 80)

    try:
        # Initialize optimized pipeline
        logger.info("Initializing optimized pipeline...")
        pipeline = OptimizedDataSynthesisPipeline(
            config_path=args.config,
            images_dir=args.images_dir,
            output_dir=args.output_dir
        )

        # Run pipeline
        logger.info("Starting optimized pipeline execution...")
        results = pipeline.run(
            num_images=args.num_images,
            bins_ratio=tuple(args.bins_ratio)
        )

        # Print results summary
        logger.info("=" * 80)
        logger.info("Pipeline Results Summary")
        logger.info("=" * 80)
        logger.info(f"Filtered images: {results.get('filtered_count', 0)}")

        if 'bins' in results:
            logger.info("Images per bin:")
            for bin_name, count in results['bins'].items():
                logger.info(f"  Bin {bin_name}: {count} images")

        logger.info(f"Generated Q/A pairs: {results.get('generated_qa', 0)}")
        logger.info(f"Validated Q/A pairs: {results.get('validated_qa', 0)}")

        # Save detailed results
        results_file = Path(args.output_dir) / "pipeline_results.json"
        logger.info(f"Detailed results saved to: {results_file}")

        logger.info("=" * 80)
        logger.info("✓ Optimized pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
