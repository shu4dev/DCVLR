# Data Synthesis Pipeline

Implementation of reasoning-focused data synthesis workflow. The pipeline curates raw images, synthesizes question/answer/reasoning triples with a large language model, and validates the generations to produce high-quality vision-language datasets.

## Highlights
- **End-to-end pipeline** – filtering, binning, synthesis, and validation in a single orchestrator
- **Modular stages** – swap filtering, LLM, or validation components by editing `configs/default_config.yaml`
- **Multi-GPU optimization** – enable with `--optimize` flag for 2-4x speedup
- **Flexible captioning** – choose between BLIP, BLIP-2, or Moondream API for image captions
- **Dual OCR backends** – PaddleOCR (lightweight) or DeepSeek-OCR (high accuracy)
- **Feature extraction modes** – full features (OCR+objects+captions) or caption-only for 70% faster processing
- **Detailed binning analysis** – view how each image scores against all bin criteria with custom filters
- **Intermediate saves** – automatically saves results after each stage for debugging and recovery
- **Automatic resume** – detects interrupted runs and resumes from the last completed stage
- **Scriptable + importable** – run via CLI or embed with the `DataSynthesisPipeline` class
- **Reproducible config** – every stage is parameterized by YAML and persisted with outputs/logs

## Repository Layout
```
DCVLR/
├── src/
│   ├── pipeline/
│   │   └── core.py                  # Main pipeline orchestrator
│   ├── filtering/
│   │   ├── filters.py               # Image filtering (standard + batched)
│   │   ├── binning.py               # Image binning
│   │   ├── binning_multiprocess.py  # Multi-GPU binning
│   │   └── yolov11.py               # YOLO detection
│   ├── synthesis/
│   │   ├── qa_generator.py          # Q/A generation (WIP)
│   │   ├── feature_extractor.py     # Feature extraction (WIP)
│   │   ├── deepseek_qa_generator.py # DeepSeek API script
│   │   └── api_client.py            # API utilities
│   ├── validation/
│   │   └── validator.py             # Data validation
│   └── utils/
│       ├── logging.py               # Logging configuration
│       ├── gpu.py                   # GPU management
│       ├── image_collector.py       # Image collection utilities
│       └── dataset_converter.py     # Dataset format conversion
├── scripts/
│   ├── run_pipeline.py              # CLI (supports --optimize flag)
│   └── export_images.py             # Export images from Hugging Face
├── examples/
│   └── pipeline_demo.py             # Demo script
├── configs/
│   └── default_config.yaml          # Tunable thresholds and model names
├── docs/                            # Extended documentation
├── tests/                           # Unit tests
├── requirements.txt                 # Python dependencies
└── README.md
```

## Prerequisites
- Python 3.9+ (3.10 recommended).
- CUDA-capable GPU with ≥16 GB VRAM for BLIP/YOLO/LLM inference. CPU-only is supported for experimentation but will be slow.
- System packages for OpenCV (FFmpeg/libjpeg) and PaddleOCR (see their docs) if you plan to run the filtering/binning stage locally.

## Installation
```bash
git clone https://github.com/shu4dev/DCVLR.git
cd DCVLR

# Optional but recommended: create an isolated environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install runtime dependencies
pip install -r requirements.txt

# Or install in editable mode with extras for development/notebooks
pip install -e .[dev,notebook]
```

## Configuring the Pipeline
The default configuration lives in `configs/default_config.yaml`. Each block corresponds to a pipeline stage:

- `filtering` – min resolution, NSFW and watermark thresholds, duplicate detection.
- `binning` – OCR/text thresholds, object detector selection (YOLO/SAM), captioning backend (BLIP/BLIP-2/Moondream), CLIP similarity cutoffs.
- `synthesis` – LLM ID, decoding params, feature extraction mode (full/caption-only), model configurations.
- `validation` – minimum lengths and reasoning/grounding checks.
- `output` – intermediate saves toggle, compression, output formats.

Example snippet:
```yaml
filtering:
  min_resolution: 256
  nsfw_threshold: 0.5

binning:
  text_boxes_threshold: 2
  object_count_threshold: 5

  # Captioning Backend: 'blip' (fast), 'blip2' (quality), or 'moondream' (API)
  captioner_backend: 'blip'
  moondream_api_key: null       # Required for Moondream

  # Object Detection: 'yolo' (fast) or 'sam' (thorough)
  object_detector: 'yolo'
  yolo_model: 'yolov8n'

synthesis:
  llm_model: "tiiuae/falcon-7b-instruct"
  temperature: 0.7

  # Feature mode: true (detailed) or false (fast, caption-only)
  use_full_features: true

output:
  save_intermediate: true       # Save results after each stage
```
Copy the file, tweak the values, and pass the new path through `--config` (CLI) or the `config_path` argument (Python).

### Object Detector Selection: YOLO vs SAM

The binning stage uses object detection to categorize images into Bin B (Object/Spatial). You can choose between two object detection backends:

  ```yaml
  binning:
    object_detector: 'yolo'
    yolo_model: 'yolov8n'  # Options: yolov8n (fastest), yolov8s, yolov9s, yolov10s, yolov11s
  ```

### Captioning Backend Selection: BLIP vs BLIP-2 vs Moondream

The pipeline supports three captioning backends for generating image descriptions:

#### BLIP (Default)
  ```yaml
  binning:
    captioner_backend: 'blip'
  ```

#### BLIP-2
  ```yaml
  binning:
    captioner_backend: 'blip2'
  ```

#### Moondream API
  ```yaml
  binning:
    captioner_backend: 'moondream'
    moondream_api_key: 'YOUR_API_KEY'
    moondream_caption_length: 'normal'  # 'short', 'normal', or 'long'
  ```

### Detailed Binning Analysis

View how each image performs against all bin criteria plus custom user-defined criteria:

```python
from src.filtering import ImageBinner

binner = ImageBinner(config)

# Get detailed results
details = binner.categorize_image("image.jpg", return_details=True)
binner.display_image_results(details)

# Add custom criteria
def complex_scene(details):
    objects = details['bin_b_criteria']['num_objects']
    return objects > 10

user_criteria = {'Complex Scene': complex_scene}
binner.display_image_results(details, user_criteria=user_criteria)
```

Example output shows how the image performs against:
- Bin A criteria (text boxes, text area ratio)
- Bin B criteria (object count, unique classes, spatial dispersion)
- Bin C criteria (caption, CLIP similarity)
- Custom user-defined criteria


### OCR Backend Selection: PaddleOCR vs DeepSeek-OCR

The pipeline supports two OCR backends for text detection based on the pipeline mode:

#### Hybrid Mode - PaddleOCR (Default)


  ```yaml
  binning:
    pipeline_mode: 'hybrid'  # Automatically uses PaddleOCR
  ```

#### DeepSeek Unified Mode - DeepSeek-OCR
  ```yaml
  binning:
    pipeline_mode: 'deepseek_unified'  # Uses DeepSeek-OCR for all tasks
  ```

## Data Structure

The pipeline expects images organized in a specific structure:

```
data/
├── dataset1/
│   └── train/
│       ├── image1.jpg
│       ├── image2.png
│       └── ...
├── dataset2/
│   └── train/
│       └── ...
└── dataset3/
    └── train/
        └── ...
```

**Key points**:
- Each dataset has its own folder under the main data directory
- Each dataset folder must contain a `train/` subdirectory
- All images go directly in the `train/` folder
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.webp`
- Dataset names can use `__` instead of `/` (e.g., `HuggingFaceM4__ChartQA`)

**Exporting datasets from Hugging Face**:
```bash
python scripts/export_images.py
```

This verifies that your data is organized correctly and shows how many images were found in each dataset.

## Usage

> **💡 TIP:** The pipeline automatically resumes from the last completed stage if interrupted!
> Just rerun your command - no extra flags needed. [Learn more ↓](#-automatic-resume-behavior)

### 1. End-to-end CLI

**Process all images**:
```bash
python scripts/run_pipeline.py \
  --images-dir ./data \
  --num-images -1

# If interrupted, rerun the same command - it will resume automatically!
```

**Process specific number of images**:
```bash
python scripts/run_pipeline.py \
  --images-dir ./data \
  --output-dir ./output/run_001 \
  --num-images 1000 \
  --config configs/default_config.yaml \
  --bins-ratio 0.4 0.4 0.2 \
  --device cuda

# ✓ Automatic resume enabled - crashes won't lose your progress!
```

**What you'll see when resuming:**
```
$ python scripts/run_pipeline.py --images-dir ./data
Starting pipeline with 1000 images
Checking for intermediate results to resume from...
✓ Resuming from Stage 2 (Binning) - Skipping to Stage 3...
Loaded 1000 binned images from Stage 2 cache (A:400, B:400, C:200)
Stage 3: Synthesizing Q/A pairs...
[Pipeline continues from Stage 3...]
```

**Important flags**:
- `--images-dir` **[required]** – Path to data directory containing dataset folders with train/ subdirectories
- `--num-images` – Number of images to process (`-1` for all images, default: `-1`)
- `--output-dir` – Output directory for results (default: `output/`)
- `--config` – Path to configuration YAML file (default: `configs/default_config.yaml`)
- `--bins-ratio` – Bin ratios for Text:Object:Commonsense (default: `0.4 0.4 0.2`)
- `--dataset-size` – Target dataset size for binning (default: None, uses all filtered images)
- `--device` – Device to use: `cuda` or `cpu` (default: `cuda`)
- `--optimize` – Enable optimizations (batched filtering, multi-GPU binning) for 2-4x speedup
- `--verbose` – Enable detailed debug logging
- `--dry-run` – Validate configuration without running the pipeline

### 2. Python API

The Python API also benefits from automatic resume - just call `pipeline.run()` again!

```python
from src.pipeline import DataSynthesisPipeline

pipeline = DataSynthesisPipeline(
    config_path="configs/default_config.yaml",
    images_dir="data/images",
    output_dir="output/experiment_01",
    device="cuda"
)

# First run - may get interrupted
results = pipeline.run(
    num_images=500,
    bins_ratio=(0.4, 0.4, 0.2)
)
# Automatically resumes from last completed stage!

print(results["filtered_count"], "images survived filtering")
```
Each stage (`filter_stage`, `bin_stage`, `synthesis_stage`, `validation_stage`) can be called individually for experiments or notebooks.

### 3. Optimized Pipeline (Multi-GPU)

For systems with multiple GPUs and sufficient VRAM, enable optimizations with the `--optimize` flag:

```bash
python scripts/run_pipeline.py \
  --images-dir ./data \
  --num-images 100 \
  --optimize
```

**Performance improvements:**
- 2 GPUs: ~2.3x faster
- 4 GPUs: ~4.3x faster

The `--optimize` flag enables:
- **Batched filtering** (5x faster Stage 1)
- **Multi-GPU parallelism** (2x faster Stage 2)

## Outputs
Running the pipeline creates an output directory (default `output/`) containing:

```
output/
├── synthetic_qa_dataset.jsonl      # Main output: Q/A pairs
├── pipeline_results.json            # Pipeline metrics and counts
├── pipeline.log                     # Detailed execution logs
└── intermediate/                    # Intermediate results (if enabled)
    ├── stage1_filtering/
    │   ├── filtered_images.jsonl    # All filtered images
    │   └── summary.json             # Stage 1 statistics
    ├── stage2_binning/
    │   ├── bin_A.jsonl              # Bin A images
    │   ├── bin_B.jsonl              # Bin B images
    │   ├── bin_C.jsonl              # Bin C images
    │   ├── all_binned_images.jsonl  # All binned images
    │   └── summary.json             # Stage 2 statistics
    ├── stage3_synthesis/
    │   ├── generated_qa_pairs.jsonl # All generated Q/A pairs
    │   ├── bin_A_qa_pairs.jsonl     # Bin A Q/A pairs
    │   ├── bin_B_qa_pairs.jsonl     # Bin B Q/A pairs
    │   ├── bin_C_qa_pairs.jsonl     # Bin C Q/A pairs
    │   └── summary.json             # Stage 3 statistics
    └── stage4_validation/
        ├── validated_qa_pairs.jsonl # Validated Q/A pairs
        └── summary.json             # Stage 4 statistics
```

### pipeline_results.json
Summary metrics from the pipeline run:
```json
{
  "filtered_count": 8542,
  "bins": {
    "A": 3417,
    "B": 3417,
    "C": 1708
  },
  "generated_qa": 8542,
  "validated_qa": 8203
}
```

### pipeline.log
Per-stage logs with timestamps, model loading info, and progress details.

## Pipeline Stages

The pipeline consists of 4 stages that run sequentially:

### Stage 1: Filtering
Removes low-quality, inappropriate, or duplicate images:
- **Resolution check**: 224×224 to 4096×4096 pixels
- **NSFW detection**: Filters inappropriate content
- **Duplicate removal**: Uses perceptual hashing
- **Result**: Typically 70-90% of images pass

### Stage 2: Binning
Categorizes images into three bins based on visual content:

#### Bin A: Text/Arithmetic (📝 Text-heavy)
- Uses OCR to detect text regions
- Criteria: >2 text boxes OR >20% text area
- Examples: Documents, charts, equations, forms

#### Bin B: Object/Spatial (📦 Object-rich)
- Uses YOLO/SAM for object detection
- Criteria: >3 unique classes OR >5 objects OR high spatial dispersion
- Examples: Street scenes, rooms, product catalogs, crowded environments

#### Bin C: Commonsense/Attribute (🤔 General reasoning)
- Uses BLIP for captioning + CLIP for validation
- Default category for images not in A or B
- Examples: Single subjects, landscapes, close-ups, simple scenes

Images are then balanced according to the specified ratio (default 40:40:20).

**Detailed Binning Analysis**: View how each image performs against all bin criteria plus custom user-defined criteria. See [docs/DETAILED_BINNING_GUIDE.md](docs/DETAILED_BINNING_GUIDE.md) for details.

## Quick Start

### Standard Pipeline (Single GPU)

```bash
# 1. Clone and install
git clone https://github.com/shu4dev/DCVLR.git
cd DCVLR
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Organize your data
mkdir -p data/my_dataset/train
# Copy your images to data/my_dataset/train/

# 3. Test data structure
python test_train_folders_simple.py --data-dir ./data

# 4. Run the pipeline
python scripts/run_pipeline.py \
  --images-dir ./data \
  --num-images 100 \
  --verbose

# 5. Check outputs
ls output/
cat output/synthetic_qa_dataset.jsonl | head -5
```

### Optimized Pipeline (Multi-GPU)

For systems with 2+ GPUs:

```bash
# Steps 1-3 same as above

# 4. Run optimized pipeline (2-4x faster!)
python scripts/run_pipeline.py \
  --images-dir ./data \
  --num-images 100 \
  --optimize

# Automatically enables:
# - Batched filtering (5x faster)
# - Multi-GPU parallelism (2x faster)
```

## Example Datasets

The pipeline works with any image dataset. Here are some examples:

**Supported datasets** (via `scripts/export_images.py`):
- `HuggingFaceM4/ChartQA` - Chart and graph images
- `derek-thomas/ScienceQA` - Science question images
- `vidore/infovqa_train` - Information-seeking VQA
- `Luckyjhg/Geo170K` - Geometry images
- `lmms-lab/multimodal-open-r1-8k-verified` - Multimodal reasoning
- `Zhiqiang007/MathV360K` - Mathematical images
- `oumi-ai/walton-multimodal-cold-start-r1-format` - General VQA

**Adding your own dataset**:
1. Create folder: `mkdir -p data/my_custom_dataset/train`
2. Add images: Copy `.jpg`, `.png`, etc. to the `train/` folder
3. Run pipeline: `python scripts/run_pipeline.py --images-dir ./data`