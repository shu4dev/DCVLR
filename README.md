# Team-1 Data Synthesis Pipeline

Implementation of the Team-1 reasoning-focused data synthesis workflow (see `Implmentation.pdf`). The pipeline curates raw images, synthesizes question/answer/reasoning triples with a large language model, validates the generations, and fine-tunes/evaluates a downstream VLM.

## Highlights
- **End-to-end pipeline** – filtering, binning, synthesis, validation, and benchmarking live in a single orchestrator (`team1_pipeline.py`).
- **Modular stages** – swap filtering, LLM, or benchmarking components by editing `configs/default_config.yaml`.
- **Scriptable + importable** – run via `scripts/run_pipeline.py` or embed with the `DataSynthesisPipeline` class.
- **Reproducible config** – every stage is parameterized by YAML and persisted along with outputs/logs.

## Repository Layout
```
team1-data-synthesis/
├── configs/
│   └── default_config.yaml          # Tunable thresholds and model names
├── docs/                            # (Optional) extended module notes
├── notebooks/
│   └── pipeline_demo.ipynb          # Walk-through notebook
├── scripts/
│   └── run_pipeline.py              # CLI for the full pipeline
├── src/
│   ├── benchmarking/                # Fine-tuning/eval helpers (stubs)
│   ├── filtering/
│   │   ├── binning.py               # Text/object/commonsense binning
│   │   └── image_filter.py          # Resolution/NSFW/watermark filtering
│   ├── synthesis/                   # Q/A generation + feature extraction (stubs)
│   ├── validation/                  # Dataset validation utilities (stubs)
│   └── utils/                       # Shared helpers (logging, etc.)
├── tests/                           # Placeholder for unit tests
├── team1_pipeline.py                # High-level orchestrator
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
git clone https://github.com/<your-org>/team1-data-synthesis.git
cd team1-data-synthesis

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
- `binning` – OCR/text thresholds, YOLO object count, CLIP similarity cutoffs.
- `synthesis` – LLM ID, decoding params, OCR/YOLO/BLIP model names used by the feature extractor.
- `validation` – minimum lengths and reasoning/grounding checks.
- `benchmarking` – target model, hyper-parameters, and evaluation benchmarks.

Example snippet:
```yaml
filtering:
  min_resolution: 256
  nsfw_threshold: 0.5

binning:
  text_boxes_threshold: 2
  object_count_threshold: 5

synthesis:
  llm_model: "tiiuae/falcon-7b-instruct"
  temperature: 0.7
```
Copy the file, tweak the values, and pass the new path through `--config` (CLI) or the `config_path` argument (Python).

## Usage
### 1. End-to-end CLI
```bash
python scripts/run_pipeline.py \
  --images-dir /path/to/images \
  --output-dir ./output/run_001 \
  --num-images 1000 \
  --config configs/default_config.yaml \
  --bins-ratio 0.4 0.4 0.2 \
  --device cuda \
  --skip-benchmarking   # optional flag
```

Important flags:
- `--llm-model` to override the generation model defined in the config.
- `--bins-ratio` to control the A/B/C split (text/object/commonsense).
- `--skip-benchmarking` when you only need the dataset, not model fine-tuning.
- `--verbose` for debug logging; `--dry-run` to validate inputs without running.

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
    bins_ratio=(0.4, 0.4, 0.2),
    skip_benchmarking=True
)

print(results["filtered_count"], "images survived filtering")
```
Each stage (`filter_stage`, `bin_stage`, `synthesis_stage`, `validation_stage`, `benchmark_stage`) can be called individually for experiments or notebooks.

### 3. Notebook Demo
`notebooks/pipeline_demo.ipynb` mirrors the CLI flow but also visualizes intermediate artifacts (bin distribution, example Q/A pairs, reasoning lengths). After installing the optional notebook extras, open it with Jupyter and update the `images_dir` cell.

## Outputs
Running the pipeline creates an output directory (default `output/`) containing:
- `synthetic_qa_dataset.jsonl` – newline-delimited records:
  ```json
  {
    "image": "path/to/image.jpg",
    "bin": "A",
    "question": "...",
    "answer": "...",
    "reasoning": "..."
  }
  ```
- `pipeline_results.json` – summary counts and benchmark scores.
- `pipeline.log` – per-stage logs produced by `setup_logging`.
You can supply your own directory via the `output_dir` argument to keep multiple runs.

## Development & Testing
```bash
pip install -e .[dev]
pytest          # run unit tests (add cases under tests/)
black .         # format
flake8 src/     # lint
```
The `tests/` directory currently contains placeholders – add unit tests alongside new modules. When extending the pipeline, prefer small mocks for GPU-heavy components so CI can run without specialized hardware.

## License
Distributed under the MIT License (see `LICENSE`). If you publish work using this implementation, please reference the Team-1 methodology described in `Implmentation.pdf`.
