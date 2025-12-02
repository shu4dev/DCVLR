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
from src.benchmarking import ModelTrainer, BenchmarkEvaluator
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
    4. Model fine-tuning and benchmarking
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
        
        # Initialize components
        self.image_filter = ImageFilter(self.config['filtering'])
        self.image_binner = ImageBinner(self.config['binning'])
        self.feature_extractor = FeatureExtractor(device=device)
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
        bins_ratio: Tuple[float, float, float] = (0.4, 0.4, 0.2),
        skip_benchmarking: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline end-to-end.
        
        Args:
            num_images: Target number of images to process
            bins_ratio: Ratio for Bin A (Text), B (Object), C (Commonsense)
            skip_benchmarking: Whether to skip the benchmarking stage
        
        Returns:
            Dictionary containing pipeline results and metrics
        """
        logger.info(f"Starting pipeline with {num_images} images")
        results = {}
        
        # Stage 1: Filtering
        logger.info("Stage 1: Filtering images...")
        filtered_images = self.filter_stage(num_images)
        results['filtered_count'] = len(filtered_images)
        
        # Stage 2: Binning
        logger.info("Stage 2: Binning images...")
        binned_images = self.bin_stage(filtered_images, bins_ratio)
        results['bins'] = {
            'A': len(binned_images['A']),
            'B': len(binned_images['B']),
            'C': len(binned_images['C'])
        }
        
        # Stage 3: Synthesis
        logger.info("Stage 3: Synthesizing Q/A pairs...")
        qa_dataset = self.synthesis_stage(binned_images)
        results['generated_qa'] = len(qa_dataset)
        
        # Stage 4: Validation
        logger.info("Stage 4: Validating dataset...")
        validated_dataset = self.validation_stage(qa_dataset)
        results['validated_qa'] = len(validated_dataset)
        
        # Save validated dataset
        self.save_dataset(validated_dataset)
        
        # Stage 5: Benchmarking (optional)
        if not skip_benchmarking:
            logger.info("Stage 5: Benchmarking...")
            benchmark_results = self.benchmark_stage(validated_dataset)
            results['benchmarks'] = benchmark_results
        
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
    
    def benchmark_stage(self, dataset: List[Dict]) -> Dict[str, float]:
        """
        Stage 5: Fine-tune model and evaluate on benchmarks.
        
        Benchmarks:
        - TextVQA
        - DocVQA
        - ChartQA
        """
        # Initialize trainer
        trainer = ModelTrainer(
            model_name=self.config['benchmarking']['model'],
            device=self.device
        )
        
        # Fine-tune model
        logger.info("Fine-tuning model...")
        model = trainer.train(dataset)
        
        # Evaluate on benchmarks
        evaluator = BenchmarkEvaluator(model, self.device)
        
        results = {}
        for benchmark in ['textvqa', 'docvqa', 'chartqa']:
            logger.info(f"Evaluating on {benchmark}...")
            score = evaluator.evaluate(benchmark)
            results[benchmark] = score
            logger.info(f"{benchmark} score: {score:.2f}%")
        
        return results
    
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
    
    def benchmark_yolo_models(self, images: List[Dict]) -> None:
        """
        Benchmark YOLO models on a set of images.

        This method is useful for evaluating different YOLO versions
        before deciding which to use in production.

        Args:
            images: List of image dictionaries with 'path' key
        """
        if not self.config['binning'].get('enable_multi_yolo', False):
            logger.warning(
                "Multi-YOLO benchmarking not enabled. "
                "Set enable_multi_yolo: true in config to use this feature."
            )
            return

        image_paths = [img['path'] for img in images]
        logger.info(f"Benchmarking YOLO models on {len(image_paths)} images...")

        results_df = self.image_binner.benchmark_yolo_models(image_paths)

        # Save results
        output_path = self.output_dir / "yolo_benchmark_results.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"YOLO benchmark results saved to {output_path}")

        # Print summary
        print("\nYOLO Benchmark Summary:")
        print("=" * 60)
        for model_name in results_df['model_name'].unique():
            model_results = results_df[results_df['model_name'] == model_name]
            avg_time = model_results['avg_inference_time'].iloc[0]
            pass_rate = model_results['passes_filter'].sum() / len(model_results) * 100
            print(f"{model_name:12s} - Avg time: {avg_time:.3f}s, Pass rate: {pass_rate:.1f}%")

    def _load_image_paths(self) -> List[str]:
        """Load all image paths from the images directory, including subdirectories."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

        paths = []
        for ext in image_extensions:
            # Use ** to recursively search all subdirectories
            paths.extend(Path(self.images_dir).glob(f"**/*{ext}"))
            paths.extend(Path(self.images_dir).glob(f"**/*{ext.upper()}"))

        return [str(p) for p in paths]


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
    parser.add_argument(
        '--skip-benchmarking',
        action='store_true',
        help='Skip benchmarking stage'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = DataSynthesisPipeline(
        config_path=args.config,
        images_dir=args.images_dir,
        output_dir=args.output_dir
    )
    
    results = pipeline.run(
        num_images=args.num_images,
        skip_benchmarking=args.skip_benchmarking
    )
    
    print("\nPipeline Results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
