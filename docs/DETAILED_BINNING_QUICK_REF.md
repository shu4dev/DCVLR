# Detailed Binning Quick Reference

## Single Image with Details

```python
from src.filtering.binning import ImageBinner

binner = ImageBinner(config)

# Get details
details = binner.categorize_image("image.jpg", return_details=True)

# Display results
binner.display_image_results(details)
```

## Batch Processing with Details

```python
images = ImageBinner.load_images_from_train_folders("data/processed")

bins = binner.bin_images(
    images=images,
    display_details=True  # Shows details for each image
)
```

## Add Custom Criteria

```python
def my_criterion(details):
    """Custom check for complex images"""
    num_objects = details['bin_b_criteria']['num_objects']
    passes = num_objects > 10
    return {
        'passes': passes,
        'message': f"Found {num_objects} objects"
    }

user_criteria = {
    'My Custom Check': my_criterion
}

# Use with single image
details = binner.categorize_image("image.jpg", return_details=True)
binner.display_image_results(details, user_criteria=user_criteria)

# Or with batch processing
bins = binner.bin_images(
    images=images,
    display_details=True,
    user_criteria=user_criteria
)
```

## Access Details Programmatically

```python
details = binner.categorize_image("image.jpg", return_details=True)

# Access values
bin_assigned = details['assigned_bin']
num_text_boxes = details['bin_a_criteria']['num_text_boxes']
num_objects = details['bin_b_criteria']['num_objects']
caption = details['bin_c_criteria']['caption']
clip_score = details['bin_c_criteria']['clip_similarity']

# Check if passes specific criteria
passes_bin_a = details['bin_a_criteria']['overall_passes']
passes_bin_b = details['bin_b_criteria']['overall_passes']
```

## Export Results to CSV

```python
import pandas as pd

all_details = []
for img_data in images:
    details = binner.categorize_image(img_data['path'], return_details=True)
    all_details.append({
        'filename': details['filename'],
        'bin': details['assigned_bin'],
        'text_boxes': details['bin_a_criteria']['num_text_boxes'],
        'text_area': details['bin_a_criteria']['text_area_ratio'],
        'objects': details['bin_b_criteria']['num_objects'],
        'unique_classes': details['bin_b_criteria']['unique_classes'],
        'caption': details['bin_c_criteria']['caption'],
        'clip_similarity': details['bin_c_criteria']['clip_similarity']
    })

df = pd.DataFrame(all_details)
df.to_csv('results.csv', index=False)
```

## Example Output Format

```
================================================================================
IMAGE: example.jpg
PATH: /full/path/to/example.jpg
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

[USER-DEFINED CRITERIA]
  High Text Density: ✗ FAIL
    → Text area ratio is 5.23% (threshold: >40%)
  Complex Scene: ✓ PASS
    → 12 objects with 0.45 dispersion
================================================================================
```

## Details Dictionary Structure

```python
{
    'image_path': str,
    'filename': str,
    'assigned_bin': 'A' | 'B' | 'C',

    'bin_a_criteria': {
        'num_text_boxes': int,
        'text_boxes_threshold': int,
        'text_boxes_passes': bool,
        'text_area_ratio': float,
        'text_area_threshold': float,
        'text_area_passes': bool,
        'overall_passes': bool
    },

    'bin_b_criteria': {
        'num_objects': int,
        'object_count_threshold': int,
        'object_count_passes': bool,
        'unique_classes': int,
        'unique_objects_threshold': int,
        'unique_classes_passes': bool,
        'spatial_dispersion': float,
        'spatial_dispersion_threshold': float,
        'spatial_dispersion_passes': bool,
        'overall_passes': bool
    },

    'bin_c_criteria': {
        'caption': str,
        'clip_similarity': float,
        'clip_threshold': float,
        'similarity_passes': bool,
        'overall_passes': bool
    }
}
```

## Common Custom Criteria Examples

### High Text Density
```python
def high_text_density(details):
    ratio = details['bin_a_criteria']['text_area_ratio']
    return ratio > 0.4
```

### Complex Scene
```python
def complex_scene(details):
    objects = details['bin_b_criteria']['num_objects']
    dispersion = details['bin_b_criteria']['spatial_dispersion']
    return objects > 10 and dispersion > 0.5
```

### Good Caption Match
```python
def good_caption(details):
    similarity = details['bin_c_criteria']['clip_similarity']
    return similarity > 0.35
```

### Minimum Quality Bar
```python
def minimum_quality(details):
    has_content = (
        details['bin_a_criteria']['num_text_boxes'] > 0 or
        details['bin_b_criteria']['num_objects'] > 3
    )
    good_caption = details['bin_c_criteria']['clip_similarity'] > 0.2
    return has_content and good_caption
```

For complete documentation, see [DETAILED_BINNING_GUIDE.md](DETAILED_BINNING_GUIDE.md)
