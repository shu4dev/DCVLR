# Detailed Binning Display Guide

This guide explains how to use the detailed binning display feature in `binning.py` to see how each image performs against all bin criteria.

## Overview

The detailed binning display feature provides comprehensive information about:
- How each image performs against **Bin A** (Text/Arithmetic) criteria
- How each image performs against **Bin B** (Object/Spatial) criteria
- How each image performs against **Bin C** (Commonsense/Attribute) criteria
- How each image performs against **user-defined custom criteria**
- Which bin the image was ultimately assigned to

## Basic Usage

### 1. Single Image Analysis

To analyze a single image with detailed output:

```python
from src.filtering.binning import ImageBinner

# Initialize binner
binner = ImageBinner(config)

# Get detailed results
details = binner.categorize_image(
    image_path="path/to/image.jpg",
    return_details=True  # This returns detailed dict instead of just bin letter
)

# Display formatted results
binner.display_image_results(details)
```

### 2. Batch Processing with Details

To process multiple images and display details for each:

```python
# Load images
images = ImageBinner.load_images_from_train_folders("data/processed")

# Process with detailed display
bins = binner.bin_images(
    images=images,
    display_details=True  # Shows details for each image as it's processed
)
```

## Example Output

When you display detailed results, you'll see output like this:

```
================================================================================
IMAGE: example_image.jpg
PATH: /full/path/to/example_image.jpg
ASSIGNED BIN: B
================================================================================

[BIN A - Text/Arithmetic Criteria]
  Text Boxes: 1 (threshold: >2) ✗ FAIL
  Text Area Ratio: 0.0523 (threshold: >0.2) ✗ FAIL
  → Overall Bin A: ✗ FAILS

[BIN B - Object/Spatial Criteria]
  Object Count: 12 (threshold: >5) ✓ PASS
  Unique Classes: 7 (threshold: >3) ✓ PASS
  Spatial Dispersion: 0.4521 (threshold: >0.3) ✓ PASS
  → Overall Bin B: ✓ PASSES

[BIN C - Commonsense/Attribute Criteria]
  Caption: 'a group of people standing around a table'
  CLIP Similarity: 0.2843 (threshold: >=0.25) ✓ PASS
  → Overall Bin C: ✗ FAILS
================================================================================
```

## User-Defined Criteria

You can add custom criteria to evaluate images based on your specific needs.

### Defining Custom Criteria

Each criterion is a function that:
- Takes the `details` dictionary as input
- Returns either:
  - A boolean (`True`/`False`)
  - A dict with `'passes'` (bool) and `'message'` (str) keys

#### Example 1: Simple Boolean Check

```python
def has_many_text_boxes(details):
    """Check if image has more than 10 text boxes"""
    return details['bin_a_criteria']['num_text_boxes'] > 10
```

#### Example 2: Check with Message

```python
def high_text_density(details):
    """Check if image has very high text density"""
    text_ratio = details['bin_a_criteria']['text_area_ratio']
    passes = text_ratio > 0.4
    return {
        'passes': passes,
        'message': f"Text area ratio is {text_ratio:.2%} (threshold: >40%)"
    }
```

#### Example 3: Complex Multi-Criteria Check

```python
def complex_scene(details):
    """Check if image has a complex scene with many diverse objects"""
    num_objects = details['bin_b_criteria']['num_objects']
    unique_classes = details['bin_b_criteria']['unique_classes']
    dispersion = details['bin_b_criteria']['spatial_dispersion']

    # Complex scene: >10 objects, >5 unique classes, and high dispersion
    passes = (
        num_objects > 10 and
        unique_classes > 5 and
        dispersion > 0.5
    )

    return {
        'passes': passes,
        'message': (
            f"{num_objects} objects, {unique_classes} unique classes, "
            f"dispersion {dispersion:.2f}"
        )
    }
```

### Using Custom Criteria

```python
# Define your criteria
user_criteria = {
    'High Text Density': high_text_density,
    'Complex Scene': complex_scene,
    'Has Many Text Boxes': has_many_text_boxes
}

# Option 1: Display for single image
details = binner.categorize_image(image_path, return_details=True)
binner.display_image_results(details, user_criteria=user_criteria)

# Option 2: Use in batch processing
bins = binner.bin_images(
    images=images,
    display_details=True,
    user_criteria=user_criteria
)
```

## Accessing Details Programmatically

The `details` dictionary returned by `categorize_image()` has this structure:

```python
{
    'image_path': '/full/path/to/image.jpg',
    'filename': 'image.jpg',
    'assigned_bin': 'B',

    'bin_a_criteria': {
        'num_text_boxes': 1,
        'text_boxes_threshold': 2,
        'text_boxes_passes': False,
        'text_area_ratio': 0.0523,
        'text_area_threshold': 0.2,
        'text_area_passes': False,
        'overall_passes': False
    },

    'bin_b_criteria': {
        'num_objects': 12,
        'object_count_threshold': 5,
        'object_count_passes': True,
        'unique_classes': 7,
        'unique_objects_threshold': 3,
        'unique_classes_passes': True,
        'spatial_dispersion': 0.4521,
        'spatial_dispersion_threshold': 0.3,
        'spatial_dispersion_passes': True,
        'overall_passes': True
    },

    'bin_c_criteria': {
        'caption': 'a group of people standing around a table',
        'clip_similarity': 0.2843,
        'clip_threshold': 0.25,
        'similarity_passes': True,
        'overall_passes': False
    }
}
```

### Example: Filter Images by Custom Logic

```python
# Get details for all images
image_details = []
for img_data in images:
    details = binner.categorize_image(img_data['path'], return_details=True)
    image_details.append(details)

# Filter images that pass multiple criteria
complex_images = [
    d for d in image_details
    if (d['bin_b_criteria']['num_objects'] > 10 and
        d['bin_b_criteria']['spatial_dispersion'] > 0.5)
]

# Find images with text but also objects
hybrid_images = [
    d for d in image_details
    if (d['bin_a_criteria']['text_area_ratio'] > 0.1 and
        d['bin_b_criteria']['num_objects'] > 5)
]

print(f"Found {len(complex_images)} complex images")
print(f"Found {len(hybrid_images)} hybrid text+object images")
```

## Use Cases

### 1. Debugging Binning Decisions

Use detailed display to understand why an image was assigned to a specific bin:

```python
# Process suspicious images with details
suspicious_images = [img for img in images if 'suspicious' in img['path']]
bins = binner.bin_images(suspicious_images, display_details=True)
```

### 2. Finding Edge Cases

Identify images that almost pass criteria but don't quite make it:

```python
def near_threshold(details):
    """Find images close to bin B threshold"""
    num_objects = details['bin_b_criteria']['num_objects']
    threshold = details['bin_b_criteria']['object_count_threshold']

    # Within 2 of the threshold
    near = abs(num_objects - threshold) <= 2
    return {
        'passes': near,
        'message': f"Objects: {num_objects}, Threshold: {threshold}"
    }

user_criteria = {'Near Bin B Threshold': near_threshold}
bins = binner.bin_images(images, display_details=True, user_criteria=user_criteria)
```

### 3. Custom Quality Filters

Create custom quality checks for your specific use case:

```python
def publication_quality(details):
    """Images suitable for publication need high quality across all dimensions"""

    # Not too much text (not a document scan)
    text_ok = details['bin_a_criteria']['text_area_ratio'] < 0.5

    # Some objects present (not blank/abstract)
    objects_ok = details['bin_b_criteria']['num_objects'] >= 3

    # Good caption match (meaningful content)
    caption_ok = details['bin_c_criteria']['clip_similarity'] > 0.3

    passes = text_ok and objects_ok and caption_ok

    return {
        'passes': passes,
        'message': (
            f"Text: {text_ok}, Objects: {objects_ok}, Caption: {caption_ok}"
        )
    }
```

### 4. Exporting Results to CSV

```python
import pandas as pd

# Collect all details
all_details = []
for img_data in images:
    details = binner.categorize_image(img_data['path'], return_details=True)
    all_details.append(details)

# Flatten for DataFrame
rows = []
for d in all_details:
    row = {
        'filename': d['filename'],
        'assigned_bin': d['assigned_bin'],
        'text_boxes': d['bin_a_criteria']['num_text_boxes'],
        'text_area_ratio': d['bin_a_criteria']['text_area_ratio'],
        'num_objects': d['bin_b_criteria']['num_objects'],
        'unique_classes': d['bin_b_criteria']['unique_classes'],
        'spatial_dispersion': d['bin_b_criteria']['spatial_dispersion'],
        'caption': d['bin_c_criteria']['caption'],
        'clip_similarity': d['bin_c_criteria']['clip_similarity']
    }
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv('binning_results.csv', index=False)
print("Results exported to binning_results.csv")
```

## Configuration

The thresholds used for binning can be configured in your `default_config.yaml`:

```yaml
binning:
  text_boxes_threshold: 2         # Bin A: minimum text boxes
  text_area_threshold: 0.2        # Bin A: minimum text area ratio
  object_count_threshold: 5       # Bin B: minimum object count
  unique_objects_threshold: 3     # Bin B: minimum unique classes
  spatial_dispersion_threshold: 0.3  # Bin B: minimum spatial dispersion
  clip_similarity_threshold: 0.25    # Bin C: minimum CLIP similarity
```

## Complete Example Script

See [example_detailed_binning.py](../example_detailed_binning.py) for a complete working example with multiple user-defined criteria.

## Tips

1. **Performance**: When processing many images, consider using `display_details=False` for bulk processing, then re-process specific images with `display_details=True` for investigation.

2. **Custom Criteria**: Keep criterion functions simple and focused. Each should check one specific aspect.

3. **Thresholds**: Experiment with different threshold values to find what works best for your dataset.

4. **Logging**: The detailed display uses `print()` for immediate feedback. Check the logger output for additional processing information.

5. **Memory**: Processing with `return_details=True` requires more memory. For very large datasets, process in batches.
