"""
Example script showing how to use detailed binning display with user-defined criteria.
"""

import sys
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.filtering.binning import ImageBinner


def example_user_criteria():
    """
    Define example user criteria functions.

    Each criterion function should accept the details dictionary and return either:
    - A boolean (True/False)
    - A dict with 'passes' (bool) and 'message' (str) keys
    """

    def high_text_density(details):
        """Check if image has very high text density (>40% text area)"""
        text_ratio = details['bin_a_criteria']['text_area_ratio']
        passes = text_ratio > 0.4
        return {
            'passes': passes,
            'message': f"Text area ratio is {text_ratio:.2%} (threshold: >40%)"
        }

    def complex_scene(details):
        """Check if image has complex scene (many objects + high dispersion)"""
        num_objects = details['bin_b_criteria']['num_objects']
        dispersion = details['bin_b_criteria']['spatial_dispersion']
        passes = num_objects > 8 and dispersion > 0.5
        return {
            'passes': passes,
            'message': f"{num_objects} objects with {dispersion:.2f} dispersion"
        }

    def has_meaningful_caption(details):
        """Check if caption is meaningful (>5 words)"""
        caption = details['bin_c_criteria']['caption']
        word_count = len(caption.split()) if caption else 0
        passes = word_count > 5
        return {
            'passes': passes,
            'message': f"Caption has {word_count} words"
        }

    def minimum_complexity(details):
        """Check if image meets minimum complexity across all criteria"""
        text_boxes = details['bin_a_criteria']['num_text_boxes']
        objects = details['bin_b_criteria']['num_objects']
        clip_sim = details['bin_c_criteria']['clip_similarity']

        complexity_score = text_boxes * 0.3 + objects * 0.4 + clip_sim * 0.3
        passes = complexity_score > 2.0
        return {
            'passes': passes,
            'message': f"Complexity score: {complexity_score:.2f} (threshold: >2.0)"
        }

    return {
        'High Text Density': high_text_density,
        'Complex Scene': complex_scene,
        'Meaningful Caption': has_meaningful_caption,
        'Minimum Complexity': minimum_complexity
    }


def main():
    """Main function to demonstrate detailed binning."""

    # Load configuration
    config_path = Path(__file__).parent / 'configs' / 'default_config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get binning config
    binning_config = config.get('binning', {})

    print("Initializing ImageBinner...")
    binner = ImageBinner(binning_config)

    # Example 1: Process a single image with details
    print("\n" + "="*80)
    print("EXAMPLE 1: Single image with detailed output")
    print("="*80)

    # Replace this with an actual image path from your dataset
    test_image = "/path/to/your/test/image.jpg"

    # Uncomment to test with actual image:
    # details = binner.categorize_image(test_image, return_details=True)
    # binner.display_image_results(details)

    print("To use: details = binner.categorize_image(image_path, return_details=True)")
    print("        binner.display_image_results(details)")

    # Example 2: Process images with user criteria
    print("\n" + "="*80)
    print("EXAMPLE 2: Binning with user-defined criteria")
    print("="*80)

    # Define user criteria
    user_criteria = example_user_criteria()

    print("User criteria defined:")
    for name in user_criteria.keys():
        print(f"  - {name}")

    # Example usage with image list
    print("\nTo use with bin_images():")
    print("  bins = binner.bin_images(")
    print("      images=image_list,")
    print("      display_details=True,")
    print("      user_criteria=user_criteria")
    print("  )")

    # Example 3: Load images from data directory
    print("\n" + "="*80)
    print("EXAMPLE 3: Process all images from data directory")
    print("="*80)

    data_dir = config.get('data', {}).get('input_dir', 'data/processed')
    print(f"Data directory: {data_dir}")

    # Uncomment to actually process images:
    images = ImageBinner.load_images_from_train_folders(data_dir)
    print(f"Loaded {len(images)} images")
   
    # # Process with details
    bins = binner.bin_images(
        images=images,
        display_details=True,
        user_criteria=user_criteria
    )
    print("\nExample user criteria functions:")
    print("-" * 80)
    print("""
def high_text_density(details):
    text_ratio = details['bin_a_criteria']['text_area_ratio']
    passes = text_ratio > 0.4
    return {
        'passes': passes,
        'message': f"Text area ratio is {text_ratio:.2%}"
    }

def complex_scene(details):
    num_objects = details['bin_b_criteria']['num_objects']
    dispersion = details['bin_b_criteria']['spatial_dispersion']
    passes = num_objects > 8 and dispersion > 0.5
    return {
        'passes': passes,
        'message': f"{num_objects} objects with {dispersion:.2f} dispersion"
    }

# Use them:
user_criteria = {
    'High Text Density': high_text_density,
    'Complex Scene': complex_scene
}

bins = binner.bin_images(
    images=image_list,
    display_details=True,
    user_criteria=user_criteria
)
    """)


if __name__ == '__main__':
    main()
