# Team-1 Data Synthesis Pipeline

Implementation of the Team-1 reasoning-focused data synthesis workflow (see `Implmentation.pdf`). The pipeline curates raw images, synthesizes question/answer/reasoning triples with a large language model, and validates the generations to produce high-quality vision-language datasets.

## Highlights
- **End-to-end pipeline** – filtering, binning, synthesis, and validation live in a single orchestrator (`team1_pipeline.py`).
- **Modular stages** – swap filtering, LLM, or validation components by editing `configs/default_config.yaml`.
- **Flexible captioning** – Choose between BLIP, BLIP-2, or Moondream API for image captions.
- **Feature extraction modes** – Full features (OCR+objects+captions) or caption-only for 70% faster processing.
- **Intermediate saves** – Automatically saves results after each stage for debugging and recovery.
- **Scriptable + importable** – run via `scripts/run_pipeline.py` or embed with the `DataSynthesisPipeline` class.
- **Reproducible config** – every stage is parameterized by YAML and persisted along with outputs/logs.

## Repository Layout
```
team1-data-synthesis/
├── configs/
│   └── default_config.yaml          # Tunable thresholds and model names
├── docs/                            # Extended documentation
├── notebooks/
│   └── pipeline_demo.ipynb          # Walk-through notebook
├── scripts/
│   ├── run_pipeline.py              # CLI for the full pipeline
│   └── export_images.py             # Export images from Hugging Face datasets
├── src/
│   ├── filtering/
│   │   ├── binning.py               # Text/object/commonsense binning
│   │   └── image_filter.py          # Resolution/NSFW/watermark filtering
│   ├── synthesis/                   # Q/A generation + feature extraction
│   ├── validation/                  # Dataset validation utilities
│   └── utils/                       # Shared helpers (logging, GPU management)
├── tests/                           # Unit tests
├── team1_pipeline.py                # High-level orchestrator
├── test_train_folders_simple.py     # Test data structure
├── requirements.txt                 # Python dependencies
├── setup.py                         # Editable install entry-point
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

#### YOLO (You Only Look Once) - Default
- **Pros**: Fast, lightweight, provides object class labels (person, car, dog, etc.)
- **Cons**: May miss unusual or unlabeled objects
- **Best for**: Production use, speed-critical pipelines, when you need object type information
- **Configuration**:
  ```yaml
  binning:
    object_detector: 'yolo'
    yolo_model: 'yolov8n'  # Options: yolov8n (fastest), yolov8s, yolov9s, yolov10s, yolov11s
  ```

#### SAM (Segment Anything Model)
- **Pros**: Finds all objects/regions regardless of type, more thorough detection
- **Cons**: Slower, requires more memory, no object classification (estimates unique classes by mask diversity)
- **Best for**: When you need comprehensive object detection, research experiments
- **Requirements**: Install SAM separately: `pip install segment-anything`
- **Configuration**:
  ```yaml
  binning:
    object_detector: 'sam'
    sam_model_type: 'vit_b'  # Options: vit_b (375MB), vit_l (1.2GB), vit_h (2.4GB)
    sam_checkpoint: 'models/sam_vit_b_01ec64.pth'  # Download from Meta AI
  ```

**Fallback behavior**: If SAM fails to load (not installed or checkpoint missing), the pipeline automatically falls back to YOLO.

### Captioning Backend Selection: BLIP vs BLIP-2 vs Moondream

The pipeline supports three captioning backends for generating image descriptions:

#### BLIP (Default)
- **Pros**: Fast (~0.5s/image), lightweight (~1GB VRAM), good quality
- **Cons**: Lower quality than BLIP-2 or Moondream
- **Best for**: Quick processing, limited GPU memory, standard quality needs
- **Configuration**:
  ```yaml
  binning:
    captioner_backend: 'blip'
  ```

#### BLIP-2
- **Pros**: Excellent quality, runs locally, no API costs
- **Cons**: Heavy (~10GB VRAM), slower (~1.0s/image)
- **Best for**: High-quality captions, sufficient GPU memory, offline processing
- **Configuration**:
  ```yaml
  binning:
    captioner_backend: 'blip2'
  ```

#### Moondream API (New!)
- **Pros**: Excellent quality, no GPU needed, fast (~0.3s/image), cloud-based
- **Cons**: Requires API key, costs ~$0.002 per image, needs internet
- **Best for**: Limited GPU memory, large-scale processing, cloud workflows
- **Configuration**:
  ```yaml
  binning:
    captioner_backend: 'moondream'
    moondream_api_key: 'YOUR_API_KEY'
    moondream_caption_length: 'normal'  # 'short', 'normal', or 'long'
  ```

Get your Moondream API key at [moondream.ai](https://moondream.ai). See [MOONDREAM_INTEGRATION.md](MOONDREAM_INTEGRATION.md) for detailed setup instructions.

### OCR Backend Selection: DeepSeek-OCR vs PaddleOCR

The pipeline supports two OCR backends for text detection:

#### DeepSeek-OCR - Default (with automatic fallback)
- **Pros**: High accuracy, better at complex layouts and multi-language text
- **Cons**: Heavy (~10GB VRAM), slower
- **Best for**: High-quality text extraction, research
- **Configuration**: Automatically used if available, falls back to PaddleOCR on OOM

#### PaddleOCR - Fallback
- **Pros**: Lightweight (~200MB VRAM), fast, good accuracy
- **Cons**: May struggle with complex layouts
- **Best for**: Production, limited GPU memory, speed-critical pipelines
- **Configuration**:
  ```yaml
  binning:
    use_paddle_ocr: true  # Force PaddleOCR instead of DeepSeek
  ```

**Automatic fallback**: The pipeline tries DeepSeek-OCR first. If it fails due to OOM or isn't installed, it automatically falls back to PaddleOCR.

### Multi-GPU Support

The pipeline automatically detects and uses multiple GPUs when available:

```yaml
binning:
  enable_multi_gpu: true  # Default: true (auto-detect)
```

**Model distribution across GPUs**:
- GPU 0: OCR (DeepSeek or PaddleOCR)
- GPU 1: Object Detection (YOLO or SAM)
- GPU 2: CLIP (similarity)
- GPU 3: BLIP (captioning)

If fewer GPUs are available, models are distributed optimally. Single GPU mode is automatic when only one GPU is detected.

## Key Configuration Options

### Feature Extraction Mode (Q/A Synthesis)

Control how much visual information is extracted for Q/A generation:

```yaml
synthesis:
  use_full_features: true  # Options: true or false
```

| Mode | What's Extracted | Processing Speed | Memory | Best For |
|------|-----------------|------------------|--------|----------|
| **true** (Full) | OCR + objects + spatial relations + captions | Slower (~1.6s/image) | 5-10GB | Documents, charts, technical images |
| **false** (Caption-only) | Captions only | **70% faster** (~0.5s/image) | 1-2GB | Photos, simple scenes, large batches |

**Example Q/A outputs:**

*Full features:* "What is the revenue shown in the 2023 report?" → "$100M" (uses OCR text)

*Caption-only:* "What type of information is displayed?" → "Sales data" (uses caption only)

See [FEATURE_EXTRACTION_MODES.md](FEATURE_EXTRACTION_MODES.md) for detailed comparison.

### Intermediate Results Saving

Save pipeline results after each stage for debugging and recovery:

```yaml
output:
  save_intermediate: true  # Options: true or false
```

**When enabled**, creates `output/intermediate/` with:
- `stage1_filtering/` - Filtered image lists and statistics
- `stage2_binning/` - Images by bin with distribution stats
- `stage3_synthesis/` - Generated Q/A pairs before validation
- `stage4_validation/` - Validated Q/A pairs with removal rates

**Benefits**: Resume from failures, debug issues, analyze per-stage results.

See [INTERMEDIATE_SAVES_FEATURE.md](INTERMEDIATE_SAVES_FEATURE.md) for details.

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

This automatically downloads and organizes datasets into the correct structure.

**Testing your data structure**:
```bash
python test_train_folders_simple.py --data-dir ./data
```

This verifies that your data is organized correctly and shows how many images were found in each dataset.

## Usage
### 1. End-to-end CLI

**Process all images**:
```bash
python scripts/run_pipeline.py \
  --images-dir ./data \
  --num-images -1
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
```

**Important flags**:
- `--images-dir` **[required]** – Path to data directory containing dataset folders with train/ subdirectories
- `--num-images` – Number of images to process (`-1` for all images, default: `-1`)
- `--output-dir` – Output directory for results (default: `output/`)
- `--config` – Path to configuration YAML file (default: `configs/default_config.yaml`)
- `--bins-ratio` – Bin ratios for Text:Object:Commonsense (default: `0.4 0.4 0.2`)
- `--llm-model` – Override the LLM model from config
- `--device` – Device to use: `cuda` or `cpu` (default: `cuda`)
- `--verbose` – Enable detailed debug logging
- `--dry-run` – Validate configuration without running the pipeline

### 2. Python API
```python
from team1_pipeline import DataSynthesisPipeline

pipeline = DataSynthesisPipeline(
    config_path="configs/default_config.yaml",
    images_dir="data/images",
    output_dir="output/experiment_01",
    llm_model="tiiuae/falcon-7b-instruct",
    device="cuda"
)

results = pipeline.run(
    num_images=500,
    bins_ratio=(0.4, 0.4, 0.2)
)

print(results["filtered_count"], "images survived filtering")
```
Each stage (`filter_stage`, `bin_stage`, `synthesis_stage`, `validation_stage`) can be called individually for experiments or notebooks.

### 3. Notebook Demo
`notebooks/pipeline_demo.ipynb` mirrors the CLI flow but also visualizes intermediate artifacts (bin distribution, example Q/A pairs, reasoning lengths). After installing the optional notebook extras, open it with Jupyter and update the `images_dir` cell.

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

### synthetic_qa_dataset.jsonl
Newline-delimited JSON records, one per line:
```json
{
  "image": "/absolute/path/to/image.jpg",
  "bin": "A",
  "question": "What is the value shown in 2020?",
  "answer": "45",
  "reasoning": "Looking at the chart, the blue bar for year 2020 reaches the 45 mark on the y-axis."
}
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

### intermediate/ (Optional)
Intermediate results saved after each pipeline stage. This feature is controlled by the `save_intermediate` setting in the configuration file (default: `true`).

**Benefits of intermediate saves**:
- **Resume capability**: Restart from any stage if the pipeline crashes
- **Stage inspection**: Debug issues by examining outputs at each stage
- **Partial reruns**: Modify and rerun specific stages without redoing earlier work
- **Analysis**: Study how images are distributed across bins or which Q/A pairs fail validation

**To disable intermediate saves**: Set `save_intermediate: false` in `configs/default_config.yaml` under the `output` section.

Each stage's `summary.json` contains statistics like counts, distributions, and removal rates.

You can specify a custom output directory via `--output-dir` (CLI) or `output_dir` (Python API) to keep multiple runs organized.

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

### Stage 3: Q/A Synthesis
Generates question-answer-reasoning triplets:

**Feature Extraction Modes**:
- **Full Features** (default): Extracts OCR text, objects, spatial relations, and captions for detailed Q/A pairs
- **Caption-Only**: Uses only image captions for faster, lighter processing (70% faster, 80% less VRAM)

Configure in `configs/default_config.yaml`:
```yaml
synthesis:
  use_full_features: true  # false for caption-only mode
```

See [FEATURE_EXTRACTION_MODES.md](FEATURE_EXTRACTION_MODES.md) for detailed comparison and use cases.

**Q/A Generation**:
- Uses LLM with bin-specific prompts (or caption-only prompts) to generate contextual questions
- Creates detailed reasoning chains grounded in image content
- Different question types per bin (text comprehension, spatial reasoning, commonsense)

### Stage 4: Validation
Quality control for generated Q/A pairs:
- Format validation (required fields, proper structure)
- Source grounding (answers relate to image content)
- Quality checks (clarity, completeness, logic)
- Deduplication (removes duplicate Q/A pairs)

## Performance

Estimated processing time for 10,000 images on a single RTX 3090 GPU:
- **Filtering**: 1-2 hours
- **Binning**: 3-4 hours (OCR + YOLO + BLIP inference)
- **Synthesis**: 2-3 hours (if LLM generation enabled)
- **Validation**: 30 minutes
- **Total**: ~6-9 hours

Multi-GPU setup can reduce total time by 40-60%.

## Additional Documentation

### Core Documentation
- [configs/default_config.yaml](configs/default_config.yaml) - Full configuration options with detailed comments
- [DATA_STRUCTURE.md](DATA_STRUCTURE.md) - Complete guide to data organization
- [PIPELINE_DETAILED_BREAKDOWN.md](PIPELINE_DETAILED_BREAKDOWN.md) - In-depth stage-by-stage breakdown with examples
- [PIPELINE_FLOW.md](PIPELINE_FLOW.md) - Visual flow diagrams

### Feature Guides
- [MOONDREAM_INTEGRATION.md](MOONDREAM_INTEGRATION.md) - Complete guide to Moondream API captioning
- [MOONDREAM_QUICK_START.md](MOONDREAM_QUICK_START.md) - 3-step Moondream setup
- [FEATURE_EXTRACTION_MODES.md](FEATURE_EXTRACTION_MODES.md) - Full vs caption-only feature extraction
- [FEATURE_MODES_QUICK_GUIDE.md](FEATURE_MODES_QUICK_GUIDE.md) - Quick reference for feature modes
- [INTERMEDIATE_SAVES_FEATURE.md](INTERMEDIATE_SAVES_FEATURE.md) - Intermediate results saving guide

## Troubleshooting

### "No images found"
Check your data structure. Run:
```bash
python test_train_folders_simple.py --data-dir ./data
```

This verifies that images are in the correct `dataset/train/` structure.

### "Out of memory (OOM)"
1. **Use caption-only mode**: `use_full_features: false` (saves 80% VRAM)
2. **Use Moondream API**: `captioner_backend: 'moondream'` (no local VRAM needed)
3. Enable multi-GPU in config: `enable_multi_gpu: true`
4. Force PaddleOCR instead of DeepSeek: `use_paddle_ocr: true`
5. Use smaller YOLO model: `yolo_model: 'yolov8n'`
6. Reduce batch sizes in config

### "Low quality Q/A pairs"
1. **Try full features mode**: `use_full_features: true` for more detailed Q/A
2. **Try better captioning**: `captioner_backend: 'blip2'` or `'moondream'`
3. Review feature extraction quality (OCR, object detection working correctly?)
4. Check LLM prompts and generation parameters
5. Adjust validation thresholds in config
6. Ensure images are properly categorized into bins

### "Models not loading"
1. Check all dependencies are installed: `pip install -r requirements.txt`
2. For SAM, install separately: `pip install segment-anything`
3. Download SAM checkpoint if using SAM object detection
4. Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

## Quick Start

Here's a complete workflow from setup to running the pipeline:

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

## License

See LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:
```bibtex
@software{dcvlr_pipeline,
  title={DCVLR: Data Curation for Vision-Language Reasoning},
  author={Team-1},
  year={2024},
  url={https://github.com/shu4dev/DCVLR}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation in the `docs/` folder
- Review the detailed breakdowns in `PIPELINE_DETAILED_BREAKDOWN.md`