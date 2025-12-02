# Pipeline Image Loading Flow

## Before Changes

```
User specifies: --images-dir ./data
                      |
                      v
         _load_image_paths() in team1_pipeline.py
                      |
                      v
         glob(f"**/*{ext}") - searches recursively
                      |
                      v
         Returns: ALL images found anywhere under ./data
         
         Problem: No structure, mixes datasets, no train/validation separation
```

## After Changes

```
User specifies: --images-dir ./data
                      |
                      v
         _load_image_paths() in team1_pipeline.py
                      |
                      v
    ImageBinner.load_images_from_train_folders(./data)
                      |
                      v
         Scans subdirectories in ./data
                      |
                      v
         For each subdirectory (e.g., HuggingFaceM4__ChartQA):
            - Check if train/ folder exists
            - Load all images from train/ folder
            - Track dataset name and metadata
                      |
                      v
         Returns: 
         [
           {'path': '/data/HuggingFaceM4__ChartQA/train/img1.jpg',
            'dataset': 'HuggingFaceM4__ChartQA',
            'filename': 'img1.jpg'},
           {'path': '/data/derek-thomas__ScienceQA/train/img2.jpg',
            'dataset': 'derek-thomas__ScienceQA',
            'filename': 'img2.jpg'},
           ...
         ]
         
         Benefits: 
         - Structured organization
         - Dataset tracking
         - Only uses train/ folders
         - Easy to add new datasets
```

## Data Structure

```
./data/
│
├── HuggingFaceM4__ChartQA/
│   ├── train/                    ← Pipeline loads from here
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── validation/               ← Ignored (for future use)
│       └── ...
│
├── derek-thomas__ScienceQA/
│   └── train/                    ← Pipeline loads from here
│       ├── 000001.jpg
│       └── ...
│
├── vidore__infovqa_train/
│   └── train/                    ← Pipeline loads from here
│       └── ...
│
├── Luckyjhg__Geo170K/
│   └── train/                    ← Pipeline loads from here
│       └── ...
│
├── lmms-lab__multimodal-open-r1-8k-verified/
│   └── train/                    ← Pipeline loads from here
│       └── ...
│
├── Zhiqiang007__MathV360K/
│   └── train/                    ← Pipeline loads from here
│       └── ...
│
└── oumi-ai__walton-multimodal-cold-start-r1-format/
    └── train/                    ← Pipeline loads from here
        └── ...
```

## Pipeline Execution Flow

```
1. User runs: python scripts/run_pipeline.py --images-dir ./data --num-images -1
                                    |
                                    v
2. DataSynthesisPipeline.__init__()
   - Sets self.images_dir = "./data"
                                    |
                                    v
3. pipeline.run() called
   - Calls filter_stage()
                                    |
                                    v
4. filter_stage() calls _load_image_paths()
   - Loads images from all dataset/train/ folders
   - Returns list of image paths
                                    |
                                    v
5. Applies filters to loaded images
   - Resolution check
   - NSFW check
   - Duplicate removal
                                    |
                                    v
6. bin_stage() categorizes images
   - Bin A: Text/Arithmetic
   - Bin B: Object/Spatial
   - Bin C: Commonsense/Attribute
                                    |
                                    v
7. synthesis_stage() generates Q/A pairs
                                    |
                                    v
8. validation_stage() validates dataset
                                    |
                                    v
9. Results saved to output directory
```

## Key Methods

### team1_pipeline.py:_load_image_paths()
```python
def _load_image_paths(self) -> List[str]:
    """Load images from train folders."""
    images_data = ImageBinner.load_images_from_train_folders(self.images_dir)
    return [img['path'] for img in images_data]
```

### src/filtering/binning.py:load_images_from_train_folders()
```python
@staticmethod
def load_images_from_train_folders(data_dir: str) -> List[Dict]:
    """
    Scans data_dir for subdirectories.
    For each subdirectory, looks for train/ folder.
    Returns metadata for all images found.
    """
    # Implementation at lines 860-928 in binning.py
```

## Testing

```bash
# Verify structure
python test_train_folders_simple.py

# Expected output:
# ============================================================
# Testing image loading from train folders
# ============================================================
# 
# Data directory: ./data
# 
# Found 7 subdirectories:
#   ✓ HuggingFaceM4__ChartQA/
#     → train/ contains 5000 images
#   ✓ derek-thomas__ScienceQA/
#     → train/ contains 3000 images
#   ...
# 
# ============================================================
# Loading images from train folders
# ============================================================
# 
# ✓ Successfully loaded 25000 images
# 
# Images per dataset:
#   HuggingFaceM4__ChartQA: 5000 images
#   derek-thomas__ScienceQA: 3000 images
#   ...
```
