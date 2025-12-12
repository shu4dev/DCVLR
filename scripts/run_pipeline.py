#!/usr/bin/env python3
"""
Main script to run the Team-1 Data Synthesis Pipeline.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import DataSynthesisPipeline


def main():
    """Main entry point for running the pipeline."""
    
    parser = argparse.ArgumentParser(
        description="Team-1 Data Synthesis Pipeline - Generate reasoning-focused VL datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images (default behavior)
  python run_pipeline.py --images-dir ./data/images

  # Process specific number of images
  python run_pipeline.py --images-dir ./data/images --num-images 1000

  # Custom configuration and output directory
  python run_pipeline.py --images-dir ./data/images --config ./my_config.yaml --output-dir ./results

  # Custom bin ratios (Text:Object:Commonsense)
  python run_pipeline.py --images-dir ./data/images --bins-ratio 0.3 0.3 0.4
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
        default=-1,
        help='Number of images to process (default: -1 for all images)'
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
        '--dataset-size',
        type=int,
        default=None,
        help='Target dataset size for binning (default: None, uses all filtered images)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run models on (default: cuda)'
    )

    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Enable optimizations (batched filtering, multi-GPU binning) for 2-4x speedup'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without actually processing'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not Path(args.images_dir).exists():
        print(f"Error: Images directory '{args.images_dir}' does not exist")
        sys.exit(1)
    
    if not Path(args.config).exists():
        print(f"Error: Config file '{args.config}' does not exist")
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
    logger.info("=" * 60)
    logger.info("Team-1 Data Synthesis Pipeline")
    logger.info("=" * 60)
    logger.info(f"Images directory: {args.images_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of images: {args.num_images}")
    logger.info(f"Dataset size: {args.dataset_size if args.dataset_size else 'all filtered images'}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Bins ratio: {args.bins_ratio}")
    logger.info(f"Device: {args.device}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual processing will occur")
        sys.exit(0)
    
    try:
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = DataSynthesisPipeline(
            config_path=args.config,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            device=args.device,
            use_optimization=args.optimize
        )
        
        # Run pipeline
        logger.info("Starting pipeline execution...")
        results = pipeline.run(
            num_images=args.num_images,
            bins_ratio=tuple(args.bins_ratio),
            dataset_size=args.dataset_size
        )
        
        # Print results summary
        logger.info("=" * 60)
        logger.info("Pipeline Results Summary")
        logger.info("=" * 60)
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
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
