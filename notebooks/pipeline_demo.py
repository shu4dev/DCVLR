"""
Team-1 Data Synthesis Pipeline Demo

This script demonstrates how to use the Team-1 Data Synthesis Pipeline to generate
reasoning-focused Vision-Language datasets.
"""

# Import required libraries
import sys
import os
from pathlib import Path
import json
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append('..')

# Import pipeline
from team1_pipeline import DataSynthesisPipeline


def main():
    # ========================================================================
    # 1. Initialize the Pipeline
    # ========================================================================

    # Configuration
    config = {
        'images_dir': '../data/TextVQA',  # Using TextVQA dataset
        'output_dir': '../output/demo',
        'config_path': '../configs/default_config.yaml',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Initialize pipeline
    pipeline = DataSynthesisPipeline(
        config_path=config['config_path'],
        images_dir=config['images_dir'],
        output_dir=config['output_dir'],
        device=config['device']
    )

    print(f"Pipeline initialized on {config['device']}")
    print(f"Using images from: {config['images_dir']}")

    # ========================================================================
    # 2. Run Individual Pipeline Stages
    # ========================================================================

    # Stage 1: Filter Images
    print("\nStage 1: Filtering images...")
    filtered_images = pipeline.filter_stage(num_images=100)
    print(f"Filtered to {len(filtered_images)} images")

    # Display sample filtered images
    if filtered_images:
        fig, axes = plt.subplots(1, min(3, len(filtered_images)), figsize=(12, 4))
        if min(3, len(filtered_images)) == 1:
            axes = [axes]
        for i, img_data in enumerate(filtered_images[:3]):
            img = Image.open(img_data['path'])
            axes[i].imshow(img)
            axes[i].set_title(f"Image {i+1}")
            axes[i].axis('off')
        plt.savefig(Path(config['output_dir']) / 'sample_filtered_images.png')
        plt.close()

    # Stage 2: Bin Images
    print("\nStage 2: Binning images...")
    binned_images = pipeline.bin_stage(filtered_images, bins_ratio=(0.4, 0.4, 0.2))

    # Display bin distribution
    bin_counts = {k: len(v) for k, v in binned_images.items()}
    plt.figure(figsize=(8, 6))
    plt.bar(bin_counts.keys(), bin_counts.values())
    plt.xlabel('Bin Category')
    plt.ylabel('Number of Images')
    plt.title('Image Distribution Across Bins')
    for i, (k, v) in enumerate(bin_counts.items()):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.savefig(Path(config['output_dir']) / 'bin_distribution.png')
    plt.close()

    print(f"Bin A (Text/Arithmetic): {bin_counts['A']} images")
    print(f"Bin B (Object/Spatial): {bin_counts['B']} images")
    print(f"Bin C (Commonsense/Attribute): {bin_counts['C']} images")

    # Stage 3: Generate Q/A Pairs
    print("\nStage 3: Generating Q/A pairs...")
    qa_dataset = pipeline.synthesis_stage(binned_images)
    print(f"Generated {len(qa_dataset)} Q/A pairs")

    # Display sample Q/A pairs
    if qa_dataset:
        sample_qa = qa_dataset[0]
        print("\nSample Q/A pair:")
        print(f"Image: {sample_qa['image']}")
        print(f"Bin: {sample_qa['bin']}")
        print(f"Question: {sample_qa.get('question', 'N/A')}")
        print(f"Answer: {sample_qa.get('answer', 'N/A')}")
        print(f"Reasoning: {sample_qa.get('reasoning', 'N/A')[:200]}...")

    # Stage 4: Validate Dataset
    print("\nStage 4: Validating dataset...")
    validated_dataset = pipeline.validation_stage(qa_dataset)
    print(f"Validated {len(validated_dataset)} Q/A pairs")
    print(f"Removed {len(qa_dataset) - len(validated_dataset)} invalid entries")

    # Save validated dataset
    pipeline.save_dataset(validated_dataset)

    # ========================================================================
    # 3. Analyze Generated Dataset
    # ========================================================================

    # Load and analyze the dataset
    dataset_path = Path(config['output_dir']) / 'synthetic_qa_dataset.jsonl'

    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    # Display basic statistics
    print("\nDataset Statistics:")
    print(f"Total Q/A pairs: {len(df)}")
    print(f"\nDistribution by bin:")
    print(df['bin'].value_counts())

    # Question length distribution
    df['question_length'] = df['question'].str.split().str.len()
    df['answer_length'] = df['answer'].str.split().str.len()
    df['reasoning_length'] = df['reasoning'].str.split().str.len()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(df['question_length'], bins=20)
    axes[0].set_xlabel('Question Length (words)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Question Length Distribution')

    axes[1].hist(df['answer_length'], bins=20)
    axes[1].set_xlabel('Answer Length (words)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Answer Length Distribution')

    axes[2].hist(df['reasoning_length'], bins=20)
    axes[2].set_xlabel('Reasoning Length (words)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Reasoning Length Distribution')

    plt.tight_layout()
    plt.savefig(Path(config['output_dir']) / 'length_distribution.png')
    plt.close()

    # ========================================================================
    # 4. Run Complete Pipeline
    # ========================================================================

    # Run the complete pipeline end-to-end
    print("\nRunning complete pipeline...")
    results = pipeline.run(
        num_images=100,
        bins_ratio=(0.4, 0.4, 0.2),
        skip_benchmarking=True  # Skip benchmarking for demo
    )

    # Display results
    print("\nPipeline Results:")
    print(json.dumps(results, indent=2))

    # ========================================================================
    # 5. Sample Outputs
    # ========================================================================

    # Display sample Q/A pairs from each bin
    for bin_type in ['A', 'B', 'C']:
        bin_samples = df[df['bin'] == bin_type].head(2)

        print(f"\n{'='*60}")
        print(f"Bin {bin_type} Samples:")
        print(f"{'='*60}")

        for idx, row in bin_samples.iterrows():
            print(f"\nQuestion: {row['question']}")
            print(f"Answer: {row['answer']}")
            print(f"Reasoning: {row['reasoning'][:150]}...")
            print("-" * 40)

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
