# Data Directory Structure

This document describes the expected data directory structure for the DCVLR pipeline.

## Expected Structure

The pipeline expects all datasets to be organized under a main data directory (default: `./data`) with the following structure:

```
data/
├── HuggingFaceM4__ChartQA/
│   └── train/
│       ├── image1.jpg
│       ├── image2.png
│       └── ...
├── derek-thomas__ScienceQA/
│   └── train/
│       ├── image1.jpg
│       └── ...
├── vidore__infovqa_train/
│   └── train/
│       └── ...
├── Luckyjhg__Geo170K/
│   └── train/
│       └── ...
├── lmms-lab__multimodal-open-r1-8k-verified/
│   └── train/
│       └── ...
├── Zhiqiang007__MathV360K/
│   └── train/
│       └── ...
└── oumi-ai__walton-multimodal-cold-start-r1-format/
    └── train/
        └── ...
```

## Key Points

1. **Main Data Directory**: All datasets should be stored in a common directory (e.g., `./data`)

2. **Dataset Folders**: Each dataset has its own subfolder with a sanitized name (e.g., `HuggingFaceM4__ChartQA` where `/` is replaced with `__`)

3. **Train Subdirectory**: Each dataset folder must contain a `train/` subdirectory

4. **Image Files**: All images should be placed directly in the `train/` folder

5. **Supported Formats**:
   - `.jpg` / `.jpeg`
   - `.png`
   - `.bmp`
   - `.gif`
   - `.tiff`
   - `.webp`

## How the Pipeline Loads Images

The pipeline uses the `ImageBinner.load_images_from_train_folders()` method which:

1. Scans all subdirectories in the data directory
2. Looks for a `train/` folder in each subdirectory
3. Loads all images from each `train/` folder
4. Returns a list of image metadata with:
   - `path`: Absolute path to the image
   - `dataset`: Name of the dataset folder
   - `filename`: Image filename

## Example Usage

### Using the Pipeline

```python
from team1_pipeline import DataSynthesisPipeline

# Initialize pipeline with your data directory
pipeline = DataSynthesisPipeline(
    config_path='configs/default_config.yaml',
    images_dir='./data',  # Points to the main data directory
    output_dir='./output'
)

# The pipeline will automatically load all images from dataset/train folders
results = pipeline.run(num_images=-1)  # -1 means process all images
```

### Testing Image Loading

Use the provided test script to verify your data structure:

```bash
# Test with default data directory (./data)
python test_train_folders_simple.py

# Test with custom data directory
python test_train_folders_simple.py --data-dir /path/to/your/data
```

### Exporting Images from Hugging Face

Use the export script to download and organize datasets:

```bash
python scripts/export_images.py
```

This will automatically create the correct structure under `./data/`

## Running the Pipeline

### Process All Images

```bash
python scripts/run_pipeline.py --images-dir ./data --num-images -1
```

### Process Specific Number of Images

```bash
python scripts/run_pipeline.py --images-dir ./data --num-images 1000
```

### Custom Configuration

```bash
python scripts/run_pipeline.py \
  --images-dir ./data \
  --output-dir ./results \
  --config ./configs/custom_config.yaml \
  --bins-ratio 0.3 0.3 0.4
```

## Notes

- If a dataset folder doesn't have a `train/` subdirectory, it will be skipped with a warning
- Empty `train/` folders are allowed but contribute no images
- The pipeline recursively processes all datasets in the data directory
- Image paths are stored as absolute paths for reliability
