#!/usr/bin/env python3
"""
Team-1 Image Binning Pipeline Demo

This script demonstrates how to use the Team-1 Pipeline
to filter and bin images into categories.
"""

import sys
import os
from pathlib import Path
import json
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path if needed
# sys.path.append('..')

# Import pipeline
from team1_pipeline import DataSynthesisPipeline


def initialize_pipeline():
    """Initialize the pipeline with configuration."""
    config = {
        'images_dir': './data',  # Using TextVQA dataset
        'output_dir': './output/demo',
        'config_path': './configs/default_config.yaml',
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

    return pipeline, config


def run_filter_stage(pipeline, num_images=100):
    """Stage 1: Filter Images."""
    print("\n" + "="*60)
    print("Stage 1: Filtering images...")
    print("="*60)

    filtered_images = pipeline.filter_stage(num_images=num_images)
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
        plt.tight_layout()
        plt.savefig('output/demo/filtered_images.png')
        print("Saved filtered images preview to output/demo/filtered_images.png")
        plt.close()

    return filtered_images


def run_bin_stage(pipeline, filtered_images, bins_ratio=(0.4, 0.4, 0.2)):
    """Stage 2: Bin Images."""
    print("\n" + "="*60)
    print("Stage 2: Binning images...")
    print("="*60)

    binned_images = pipeline.bin_stage(filtered_images, bins_ratio=bins_ratio)

    # Display bin distribution
    bin_counts = {k: len(v) for k, v in binned_images.items()}

    plt.figure(figsize=(8, 6))
    plt.bar(bin_counts.keys(), bin_counts.values())
    plt.xlabel('Bin Category')
    plt.ylabel('Number of Images')
    plt.title('Image Distribution Across Bins')
    for i, (k, v) in enumerate(bin_counts.items()):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('output/demo/bin_distribution.png')
    print("Saved bin distribution to output/demo/bin_distribution.png")
    plt.close()

    print(f"Bin A (Text/Arithmetic): {bin_counts.get('A', 0)} images")
    print(f"Bin B (Object/Spatial): {bin_counts.get('B', 0)} images")
    print(f"Bin C (Commonsense/Attribute): {bin_counts.get('C', 0)} images")

    return binned_images


def save_binning_results(binned_images, output_dir):
    """Save binning results to JSON file."""
    print("\n" + "="*60)
    print("Saving binning results...")
    print("="*60)

    output_path = Path(output_dir) / 'binned_images.json'
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for saving
    results = {
        'bin_A': [{'path': img['path'], 'filename': Path(img['path']).name} for img in binned_images['A']],
        'bin_B': [{'path': img['path'], 'filename': Path(img['path']).name} for img in binned_images['B']],
        'bin_C': [{'path': img['path'], 'filename': Path(img['path']).name} for img in binned_images['C']],
        'summary': {
            'total_images': len(binned_images['A']) + len(binned_images['B']) + len(binned_images['C']),
            'bin_A_count': len(binned_images['A']),
            'bin_B_count': len(binned_images['B']),
            'bin_C_count': len(binned_images['C'])
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Binning results saved to: {output_path}")
    print(f"Total images: {results['summary']['total_images']}")
    print(f"  Bin A (Text/Arithmetic): {results['summary']['bin_A_count']}")
    print(f"  Bin B (Object/Spatial): {results['summary']['bin_B_count']}")
    print(f"  Bin C (Commonsense/Attribute): {results['summary']['bin_C_count']}")

    return results


def display_sample_images(binned_images, output_dir):
    """Display sample images from each bin."""
    print("\n" + "="*60)
    print("Sample Images from Each Bin")
    print("="*60)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for bin_idx, (bin_name, images) in enumerate([('A', binned_images['A']),
                                                    ('B', binned_images['B']),
                                                    ('C', binned_images['C'])]):
        print(f"\nBin {bin_name} ({len(images)} images):")

        for img_idx in range(3):
            ax = axes[bin_idx, img_idx]
            if img_idx < len(images):
                try:
                    img = Image.open(images[img_idx]['path'])
                    ax.imshow(img)
                    ax.set_title(f"Bin {bin_name} - Image {img_idx+1}")
                    print(f"  - {Path(images[img_idx]['path']).name}")
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error loading\n{e}', ha='center', va='center')
                    ax.set_title(f"Bin {bin_name} - Error")
            else:
                ax.text(0.5, 0.5, 'No image', ha='center', va='center')
                ax.set_title(f"Bin {bin_name} - N/A")
            ax.axis('off')

    plt.tight_layout()
    sample_output_path = Path(output_dir) / 'binned_samples.png'
    plt.savefig(sample_output_path)
    print(f"\nSaved sample images to: {sample_output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("="*60)
    print("Team-1 Image Binning Pipeline Demo")
    print("="*60)

    # 1. Initialize the Pipeline
    pipeline, config = initialize_pipeline()

    # 2. Run Pipeline Stages
    filtered_images = run_filter_stage(pipeline, num_images=100)
    binned_images = run_bin_stage(pipeline, filtered_images, bins_ratio=(0.4, 0.4, 0.2))

    # 3. Save Binning Results
    results = save_binning_results(binned_images, config['output_dir'])

    # 4. Display Sample Images from Each Bin
    display_sample_images(binned_images, config['output_dir'])

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print(f"\nResults saved to: {config['output_dir']}")
    print(f"  - binned_images.json: Binning results with image paths")
    print(f"  - bin_distribution.png: Distribution chart")
    print(f"  - binned_samples.png: Sample images from each bin")


if __name__ == "__main__":
    main()
