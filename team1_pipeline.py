"""
Team-1 Data Synthesis Pipeline
Main pipeline orchestrator for the complete workflow
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yaml
from tqdm import tqdm

from src.filtering import ImageFilter, ImageBinner
from src.synthesis import QAGenerator, FeatureExtractor
from src.validation import DataValidator
from src.utils import setup_logging

# Configure logging
logger = logging.getLogger(__name__)


class DataSynthesisPipeline:
    """
    Complete pipeline for Team-1 Data Synthesis methodology.

    This pipeline handles:
    1. Image filtering and binning
    2. Q/A/Reasoning synthesis
    3. Validation and quality control
    """
    
    def __init__(
        self,
        config_path: str = "configs/default_config.yaml",
        images_dir: Optional[str] = None,
        output_dir: str = "output/",
        llm_model: str = "tiiuae/falcon-7b-instruct",
        device: str = "cuda"
    ):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to configuration YAML file
            images_dir: Directory containing input images
            output_dir: Directory for output files
            llm_model: Model identifier for LLM
            device: Device to run models on ('cuda' or 'cpu')
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.images_dir = images_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Check if intermediate saving is enabled
        self.save_intermediate = self.config.get('output', {}).get('save_intermediate', True)

        # Check Q/A synthesis feature mode
        self.use_full_features = self.config.get('synthesis', {}).get('use_full_features', True)
        caption_only = not self.use_full_features

        # Initialize components
        self.image_filter = ImageFilter(self.config['filtering'])
        self.image_binner = ImageBinner(self.config['binning'])
        self.feature_extractor = FeatureExtractor(device=device, caption_only=caption_only)
        """
        self.qa_generator = QAGenerator(
            model_name=llm_model,
            config=self.config['synthesis'],
            device=device
        )
        """
        """
        self.validator = DataValidator(self.config['validation'])
        """
        
        # Setup logging
        setup_logging(self.output_dir / "pipeline.log")
        logger.info("Pipeline initialized successfully")
    
    def run(
        self,
        num_images: int = 1000,
        bins_ratio: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline end-to-end with automatic resume from intermediate results.

        Args:
            num_images: Target number of images to process
            bins_ratio: Ratio for Bin A (Text), B (Object), C (Commonsense)

        Returns:
            Dictionary containing pipeline results and metrics
        """
        logger.info(f"Starting pipeline with {num_images} images")
        logger.info("Checking for intermediate results to resume from...")
        results = {}

        # Try to resume from Stage 4 (most advanced checkpoint)
        validated_dataset = self.load_stage4_results()
        if validated_dataset is not None:
            logger.info("✓ Resuming from Stage 4 (Validation) - Pipeline already complete!")
            results['validated_qa'] = len(validated_dataset)

            # Load earlier stage results for complete metrics
            qa_dataset = self.load_stage3_results()
            if qa_dataset:
                results['generated_qa'] = len(qa_dataset)

            binned_images = self.load_stage2_results()
            if binned_images:
                results['bins'] = {
                    'A': len(binned_images['A']),
                    'B': len(binned_images['B']),
                    'C': len(binned_images['C'])
                }

            filtered_images = self.load_stage1_results()
            if filtered_images:
                results['filtered_count'] = len(filtered_images)

            # Save validated dataset
            self.save_dataset(validated_dataset)
            logger.info("Pipeline already completed! Using cached results.")
            self.save_results(results)
            return results

        # Try to resume from Stage 3
        qa_dataset = self.load_stage3_results()
        if qa_dataset is not None:
            logger.info("✓ Resuming from Stage 3 (Synthesis) - Skipping to Stage 4...")
            results['generated_qa'] = len(qa_dataset)

            # Load earlier stage results for metrics
            binned_images = self.load_stage2_results()
            if binned_images:
                results['bins'] = {
                    'A': len(binned_images['A']),
                    'B': len(binned_images['B']),
                    'C': len(binned_images['C'])
                }

            filtered_images = self.load_stage1_results()
            if filtered_images:
                results['filtered_count'] = len(filtered_images)

            # Continue from Stage 4
            logger.info("Stage 4: Validating dataset...")
            original_qa_count = len(qa_dataset)
            validated_dataset = self.validation_stage(qa_dataset)
            results['validated_qa'] = len(validated_dataset)
            self.save_stage4_results(validated_dataset, original_qa_count)

            # Save validated dataset
            self.save_dataset(validated_dataset)
            logger.info("Pipeline completed successfully!")
            self.save_results(results)
            return results

        # Try to resume from Stage 2
        binned_images = self.load_stage2_results()
        if binned_images is not None:
            logger.info("✓ Resuming from Stage 2 (Binning) - Skipping to Stage 3...")
            results['bins'] = {
                'A': len(binned_images['A']),
                'B': len(binned_images['B']),
                'C': len(binned_images['C'])
            }

            # Load Stage 1 results for metrics
            filtered_images = self.load_stage1_results()
            if filtered_images:
                results['filtered_count'] = len(filtered_images)

            # Continue from Stage 3
            logger.info("Stage 3: Synthesizing Q/A pairs...")
            qa_dataset = self.synthesis_stage(binned_images)
            results['generated_qa'] = len(qa_dataset)
            self.save_stage3_results(qa_dataset)

            # Stage 4: Validation
            logger.info("Stage 4: Validating dataset...")
            original_qa_count = len(qa_dataset)
            validated_dataset = self.validation_stage(qa_dataset)
            results['validated_qa'] = len(validated_dataset)
            self.save_stage4_results(validated_dataset, original_qa_count)

            # Save validated dataset
            self.save_dataset(validated_dataset)
            logger.info("Pipeline completed successfully!")
            self.save_results(results)
            return results

        # Try to resume from Stage 1
        filtered_images = self.load_stage1_results()
        if filtered_images is not None:
            logger.info("✓ Resuming from Stage 1 (Filtering) - Skipping to Stage 2...")
            results['filtered_count'] = len(filtered_images)
        else:
            # No cached results - start from beginning
            logger.info("No intermediate results found - Starting from Stage 1...")
            logger.info("Stage 1: Filtering images...")
            filtered_images = self.filter_stage(num_images)
            results['filtered_count'] = len(filtered_images)
            self.save_stage1_results(filtered_images)

        # Stage 2: Binning
        logger.info("Stage 2: Binning images...")
        binned_images = self.bin_stage(filtered_images, bins_ratio)
        results['bins'] = {
            'A': len(binned_images['A']),
            'B': len(binned_images['B']),
            'C': len(binned_images['C'])
        }
        self.save_stage2_results(binned_images)

        # Stage 3: Synthesis
        logger.info("Stage 3: Synthesizing Q/A pairs...")
        qa_dataset = self.synthesis_stage(binned_images)
        results['generated_qa'] = len(qa_dataset)
        self.save_stage3_results(qa_dataset)

        # Stage 4: Validation
        logger.info("Stage 4: Validating dataset...")
        original_qa_count = len(qa_dataset)
        validated_dataset = self.validation_stage(qa_dataset)
        results['validated_qa'] = len(validated_dataset)
        self.save_stage4_results(validated_dataset, original_qa_count)

        # Save validated dataset
        self.save_dataset(validated_dataset)

        logger.info("Pipeline completed successfully!")
        self.save_results(results)

        return results
    
    def filter_stage(self, num_images: int) -> List[Dict]:
        """
        Stage 1: Filter and preprocess images.

        Applies:
        - Resolution filtering
        - NSFW content filtering
        - Watermark detection
        - Duplicate removal

        Args:
            num_images: Number of images to process. Use -1 to process all images.
        """
        if not self.images_dir:
            raise ValueError("Images directory not specified")

        # Load images
        image_paths = self._load_image_paths()
        logger.info(f"Found {len(image_paths)} images")

        # Handle "process all images" case
        process_all = (num_images == -1)
        if process_all:
            images_to_check = image_paths
            logger.info("Processing all images (num_images=-1)")
        else:
            # Check extra images to account for filtering losses
            images_to_check = image_paths[:num_images * 2]

        filtered_images = []

        for img_path in tqdm(images_to_check, desc="Filtering"):
            # Apply filters
            if self.image_filter.check_resolution(img_path):
                if self.image_filter.check_nsfw(img_path):
                    if not self.image_filter.is_duplicate(img_path):
                        filtered_images.append({
                            'path': img_path,
                            'id': Path(img_path).stem
                        })

            # Only stop early if we have a specific target
            if not process_all and len(filtered_images) >= num_images:
                break

        logger.info(f"Filtered to {len(filtered_images)} images")
        return filtered_images
    
    def bin_stage(
        self,
        images: List[Dict],
        bins_ratio: Tuple[float, float, float],
        filter_by_complexity: bool = False
    ) -> Dict[str, List[Dict]]:
        """
        Stage 2: Categorize images into bins.

        Bins:
        - A: Text/Arithmetic (text-heavy images)
        - B: Object/Spatial (object-rich images)
        - C: Commonsense/Attribute (general images)

        Args:
            images: List of image dictionaries
            bins_ratio: Ratio for (A, B, C) bins
            filter_by_complexity: If True, pre-filter images by visual complexity
        """
        # Optional: Filter by complexity before binning
        if filter_by_complexity:
            logger.info("Pre-filtering images by complexity...")
            filtered_images = []
            for img_data in tqdm(images, desc="Complexity filtering"):
                if self.image_binner.filter_by_complexity(img_data['path']):
                    filtered_images.append(img_data)
            logger.info(f"Complexity filter: {len(filtered_images)}/{len(images)} images passed")
            images = filtered_images

        binned = self.image_binner.bin_images(images)

        # Balance bins according to ratio
        target_a = int(len(images) * bins_ratio[0])
        target_b = int(len(images) * bins_ratio[1])
        target_c = len(images) - target_a - target_b

        balanced = {
            'A': binned['A'][:target_a],
            'B': binned['B'][:target_b],
            'C': binned['C'][:target_c]
        }

        logger.info(f"Binned images - A: {len(balanced['A'])}, "
                   f"B: {len(balanced['B'])}, C: {len(balanced['C'])}")

        return balanced
    
    def synthesis_stage(self, binned_images: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Stage 3: Generate Q/A/Reasoning for each image.
        
        Uses:
        - OCR for text extraction
        - Object detection for spatial info
        - Image captioning for context
        - LLM for Q/A generation
        """
        qa_dataset = []
        
        for bin_type, images in binned_images.items():
            logger.info(f"Generating Q/A for Bin {bin_type}")
            
            for img_data in tqdm(images, desc=f"Bin {bin_type}"):
                # Extract features
                features = self.feature_extractor.extract_all(img_data['path'])
                
                # Generate Q/A
                qa = self.qa_generator.generate(
                    image_features=features,
                    bin_type=bin_type
                )
                
                if qa:
                    qa['image'] = img_data['path']
                    qa['bin'] = bin_type
                    qa_dataset.append(qa)
        
        logger.info(f"Generated {len(qa_dataset)} Q/A pairs")
        return qa_dataset
    
    def validation_stage(self, qa_dataset: List[Dict]) -> List[Dict]:
        """
        Stage 4: Validate and clean the dataset.
        
        Performs:
        - Format validation
        - Source grounding checks
        - Deduplication
        - Reasoning validation
        """
        validated = []
        
        for qa in tqdm(qa_dataset, desc="Validating"):
            if self.validator.validate(qa):
                validated.append(qa)
        
        # Remove duplicates
        validated = self.validator.remove_duplicates(validated)
        
        logger.info(f"Validated {len(validated)} Q/A pairs "
                   f"({len(qa_dataset) - len(validated)} removed)")
        
        return validated

    def load_stage1_results(self) -> Optional[List[Dict]]:
        """Load Stage 1 (Filtering) results from disk if available."""
        stage1_dir = self.output_dir / "intermediate" / "stage1_filtering"
        output_path = stage1_dir / "filtered_images.jsonl"

        if not output_path.exists():
            return None

        try:
            filtered_images = []
            with open(output_path, 'r') as f:
                for line in f:
                    filtered_images.append(json.loads(line))

            logger.info(f"Loaded {len(filtered_images)} images from Stage 1 cache")
            return filtered_images
        except Exception as e:
            logger.warning(f"Failed to load Stage 1 results: {e}")
            return None

    def save_stage1_results(self, filtered_images: List[Dict]):
        """Save Stage 1 (Filtering) results to disk."""
        if not self.save_intermediate:
            return

        stage1_dir = self.output_dir / "intermediate" / "stage1_filtering"
        stage1_dir.mkdir(parents=True, exist_ok=True)

        # Save filtered image list
        output_path = stage1_dir / "filtered_images.jsonl"
        with open(output_path, 'w') as f:
            for img in filtered_images:
                f.write(json.dumps(img) + '\n')

        # Save summary statistics
        summary = {
            'stage': 'Stage 1 - Filtering',
            'total_filtered': len(filtered_images),
            'images': [img['path'] for img in filtered_images]
        }
        summary_path = stage1_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Stage 1 results saved to {stage1_dir}")

    def load_stage2_results(self) -> Optional[Dict[str, List[Dict]]]:
        """Load Stage 2 (Binning) results from disk if available."""
        stage2_dir = self.output_dir / "intermediate" / "stage2_binning"

        binned_images = {'A': [], 'B': [], 'C': []}
        all_exist = True

        for bin_type in ['A', 'B', 'C']:
            bin_path = stage2_dir / f"bin_{bin_type}.jsonl"
            if not bin_path.exists():
                all_exist = False
                break

        if not all_exist:
            return None

        try:
            for bin_type in ['A', 'B', 'C']:
                bin_path = stage2_dir / f"bin_{bin_type}.jsonl"
                with open(bin_path, 'r') as f:
                    for line in f:
                        binned_images[bin_type].append(json.loads(line))

            total = sum(len(imgs) for imgs in binned_images.values())
            logger.info(f"Loaded {total} binned images from Stage 2 cache (A:{len(binned_images['A'])}, B:{len(binned_images['B'])}, C:{len(binned_images['C'])})")
            return binned_images
        except Exception as e:
            logger.warning(f"Failed to load Stage 2 results: {e}")
            return None

    def save_stage2_results(self, binned_images: Dict[str, List[Dict]]):
        """Save Stage 2 (Binning) results to disk."""
        if not self.save_intermediate:
            return

        stage2_dir = self.output_dir / "intermediate" / "stage2_binning"
        stage2_dir.mkdir(parents=True, exist_ok=True)

        # Save each bin separately
        for bin_type, images in binned_images.items():
            bin_path = stage2_dir / f"bin_{bin_type}.jsonl"
            with open(bin_path, 'w') as f:
                for img in images:
                    f.write(json.dumps(img) + '\n')

        # Save all binned images together
        all_binned_path = stage2_dir / "all_binned_images.jsonl"
        with open(all_binned_path, 'w') as f:
            for bin_type, images in binned_images.items():
                for img in images:
                    img_with_bin = img.copy()
                    img_with_bin['bin'] = bin_type
                    f.write(json.dumps(img_with_bin) + '\n')

        # Save summary statistics
        summary = {
            'stage': 'Stage 2 - Binning',
            'bin_distribution': {
                'A': len(binned_images['A']),
                'B': len(binned_images['B']),
                'C': len(binned_images['C'])
            },
            'total_binned': sum(len(images) for images in binned_images.values())
        }
        summary_path = stage2_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Stage 2 results saved to {stage2_dir}")

    def load_stage3_results(self) -> Optional[List[Dict]]:
        """Load Stage 3 (Synthesis) results from disk if available."""
        stage3_dir = self.output_dir / "intermediate" / "stage3_synthesis"
        output_path = stage3_dir / "generated_qa_pairs.jsonl"

        if not output_path.exists():
            return None

        try:
            qa_dataset = []
            with open(output_path, 'r') as f:
                for line in f:
                    qa_dataset.append(json.loads(line))

            logger.info(f"Loaded {len(qa_dataset)} Q/A pairs from Stage 3 cache")
            return qa_dataset
        except Exception as e:
            logger.warning(f"Failed to load Stage 3 results: {e}")
            return None

    def save_stage3_results(self, qa_dataset: List[Dict]):
        """Save Stage 3 (Synthesis) results to disk."""
        if not self.save_intermediate:
            return

        stage3_dir = self.output_dir / "intermediate" / "stage3_synthesis"
        stage3_dir.mkdir(parents=True, exist_ok=True)

        # Save all generated Q/A pairs
        output_path = stage3_dir / "generated_qa_pairs.jsonl"
        with open(output_path, 'w') as f:
            for qa in qa_dataset:
                f.write(json.dumps(qa) + '\n')

        # Save by bin type
        bins = {'A': [], 'B': [], 'C': []}
        for qa in qa_dataset:
            bin_type = qa.get('bin', 'C')
            bins[bin_type].append(qa)

        for bin_type, qa_list in bins.items():
            if qa_list:
                bin_path = stage3_dir / f"bin_{bin_type}_qa_pairs.jsonl"
                with open(bin_path, 'w') as f:
                    for qa in qa_list:
                        f.write(json.dumps(qa) + '\n')

        # Save summary statistics
        summary = {
            'stage': 'Stage 3 - Q/A Synthesis',
            'total_generated': len(qa_dataset),
            'by_bin': {
                'A': len(bins['A']),
                'B': len(bins['B']),
                'C': len(bins['C'])
            }
        }
        summary_path = stage3_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Stage 3 results saved to {stage3_dir}")

    def load_stage4_results(self) -> Optional[List[Dict]]:
        """Load Stage 4 (Validation) results from disk if available."""
        stage4_dir = self.output_dir / "intermediate" / "stage4_validation"
        output_path = stage4_dir / "validated_qa_pairs.jsonl"

        if not output_path.exists():
            return None

        try:
            validated_dataset = []
            with open(output_path, 'r') as f:
                for line in f:
                    validated_dataset.append(json.loads(line))

            logger.info(f"Loaded {len(validated_dataset)} validated Q/A pairs from Stage 4 cache")
            return validated_dataset
        except Exception as e:
            logger.warning(f"Failed to load Stage 4 results: {e}")
            return None

    def save_stage4_results(self, validated_dataset: List[Dict], original_count: int):
        """Save Stage 4 (Validation) results to disk."""
        if not self.save_intermediate:
            return

        stage4_dir = self.output_dir / "intermediate" / "stage4_validation"
        stage4_dir.mkdir(parents=True, exist_ok=True)

        # Save validated Q/A pairs
        output_path = stage4_dir / "validated_qa_pairs.jsonl"
        with open(output_path, 'w') as f:
            for qa in validated_dataset:
                f.write(json.dumps(qa) + '\n')

        # Save summary statistics
        removed = original_count - len(validated_dataset)
        removal_rate = (removed / original_count * 100) if original_count > 0 else 0

        summary = {
            'stage': 'Stage 4 - Validation',
            'original_count': original_count,
            'validated_count': len(validated_dataset),
            'removed_count': removed,
            'removal_rate_percent': round(removal_rate, 2)
        }
        summary_path = stage4_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Stage 4 results saved to {stage4_dir}")

    def save_dataset(self, dataset: List[Dict]):
        """Save the validated dataset to file."""
        output_path = self.output_dir / "synthetic_qa_dataset.jsonl"
        
        with open(output_path, 'w') as f:
            for entry in dataset:
                f.write(json.dumps(entry) + '\n')
        
        logger.info(f"Dataset saved to {output_path}")
    
    def save_results(self, results: Dict):
        """Save pipeline results to file."""
        output_path = self.output_dir / "pipeline_results.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def _load_image_paths(self) -> List[str]:
        """
        Load all image paths from train subdirectories within the images directory.

        Expected structure:
        images_dir/
            dataset1/
                train/
                    image1.jpg
                    image2.png
            dataset2/
                train/
                    image3.jpg

        Returns:
            List of absolute image paths from all dataset/train folders
        """
        # Use the static method from ImageBinner to load images from train folders
        images_data = ImageBinner.load_images_from_train_folders(self.images_dir)

        # Extract just the paths
        return [img['path'] for img in images_data]


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Team-1 Data Synthesis Pipeline"
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        required=True,
        help='Directory containing input images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/',
        help='Output directory for results'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=10,
        help='Number of images to process'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Configuration file path'
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = DataSynthesisPipeline(
        config_path=args.config,
        images_dir=args.images_dir,
        output_dir=args.output_dir
    )

    results = pipeline.run(
        num_images=args.num_images
    )
    
    print("\nPipeline Results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
