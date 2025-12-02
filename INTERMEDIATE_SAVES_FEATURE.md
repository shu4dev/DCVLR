# Intermediate Results Saving Feature

## Overview

The pipeline now automatically saves intermediate results after each stage, making it easier to:
- **Debug issues** by inspecting outputs at each stage
- **Resume from failures** without reprocessing earlier stages
- **Analyze the pipeline** to understand filtering/binning distributions
- **Rerun specific stages** with different parameters

## What Was Added

### 1. Configuration Support
- The pipeline reads `save_intermediate` from `configs/default_config.yaml`
- Default: `true` (enabled)
- Set to `false` to disable intermediate saves

### 2. Four New Save Methods

Added to `team1_pipeline.py`:

#### `save_stage1_results(filtered_images)`
Saves filtering results:
- `intermediate/stage1_filtering/filtered_images.jsonl` - All filtered images
- `intermediate/stage1_filtering/summary.json` - Statistics

#### `save_stage2_results(binned_images)`
Saves binning results:
- `intermediate/stage2_binning/bin_A.jsonl` - Bin A images
- `intermediate/stage2_binning/bin_B.jsonl` - Bin B images
- `intermediate/stage2_binning/bin_C.jsonl` - Bin C images
- `intermediate/stage2_binning/all_binned_images.jsonl` - All bins combined
- `intermediate/stage2_binning/summary.json` - Bin distribution statistics

#### `save_stage3_results(qa_dataset)`
Saves Q/A synthesis results:
- `intermediate/stage3_synthesis/generated_qa_pairs.jsonl` - All Q/A pairs
- `intermediate/stage3_synthesis/bin_A_qa_pairs.jsonl` - Bin A Q/A pairs
- `intermediate/stage3_synthesis/bin_B_qa_pairs.jsonl` - Bin B Q/A pairs
- `intermediate/stage3_synthesis/bin_C_qa_pairs.jsonl` - Bin C Q/A pairs
- `intermediate/stage3_synthesis/summary.json` - Generation statistics

#### `save_stage4_results(validated_dataset, original_count)`
Saves validation results:
- `intermediate/stage4_validation/validated_qa_pairs.jsonl` - Validated Q/A pairs
- `intermediate/stage4_validation/summary.json` - Validation statistics (removal rate, etc.)

### 3. Integration with Pipeline

Modified the `run()` method to call save methods after each stage:

```python
# Stage 1: Filtering
filtered_images = self.filter_stage(num_images)
self.save_stage1_results(filtered_images)  # NEW

# Stage 2: Binning
binned_images = self.bin_stage(filtered_images, bins_ratio)
self.save_stage2_results(binned_images)  # NEW

# Stage 3: Synthesis
qa_dataset = self.synthesis_stage(binned_images)
self.save_stage3_results(qa_dataset)  # NEW

# Stage 4: Validation
validated_dataset = self.validation_stage(qa_dataset)
self.save_stage4_results(validated_dataset, original_qa_count)  # NEW
```

### 4. Documentation Updates

Updated `README.md` with:
- New output directory structure showing intermediate results
- Benefits of intermediate saves
- How to disable the feature

## Output Directory Structure

```
output/
├── synthetic_qa_dataset.jsonl      # Final output
├── pipeline_results.json            # Summary metrics
├── pipeline.log                     # Execution logs
└── intermediate/                    # NEW: Intermediate results
    ├── stage1_filtering/
    │   ├── filtered_images.jsonl
    │   └── summary.json
    ├── stage2_binning/
    │   ├── bin_A.jsonl
    │   ├── bin_B.jsonl
    │   ├── bin_C.jsonl
    │   ├── all_binned_images.jsonl
    │   └── summary.json
    ├── stage3_synthesis/
    │   ├── generated_qa_pairs.jsonl
    │   ├── bin_A_qa_pairs.jsonl
    │   ├── bin_B_qa_pairs.jsonl
    │   ├── bin_C_qa_pairs.jsonl
    │   └── summary.json
    └── stage4_validation/
        ├── validated_qa_pairs.jsonl
        └── summary.json
```

## Example Summary Files

### Stage 1 Summary
```json
{
  "stage": "Stage 1 - Filtering",
  "total_filtered": 1000,
  "images": ["/path/to/img1.jpg", "/path/to/img2.jpg", ...]
}
```

### Stage 2 Summary
```json
{
  "stage": "Stage 2 - Binning",
  "bin_distribution": {
    "A": 300,
    "B": 500,
    "C": 200
  },
  "total_binned": 1000
}
```

### Stage 3 Summary
```json
{
  "stage": "Stage 3 - Q/A Synthesis",
  "total_generated": 1000,
  "by_bin": {
    "A": 300,
    "B": 500,
    "C": 200
  }
}
```

### Stage 4 Summary
```json
{
  "stage": "Stage 4 - Validation",
  "original_count": 1000,
  "validated_count": 950,
  "removed_count": 50,
  "removal_rate_percent": 5.0
}
```

## Usage

### Enable (Default)
```yaml
# configs/default_config.yaml
output:
  save_intermediate: true
```

### Disable
```yaml
# configs/default_config.yaml
output:
  save_intermediate: false
```

### Running the Pipeline
```bash
python scripts/run_pipeline.py \
  --images-dir ./data \
  --num-images 100 \
  --output-dir ./output
```

The intermediate results will be automatically saved to `./output/intermediate/`

## Use Cases

### 1. Debugging
If Stage 3 (Q/A synthesis) produces poor results:
```bash
# Check binning distribution
cat output/intermediate/stage2_binning/summary.json

# Inspect images in Bin A
cat output/intermediate/stage2_binning/bin_A.jsonl | head -10
```

### 2. Resume from Failure
If pipeline crashes during Stage 3:
1. Load intermediate results from Stage 2
2. Restart from Stage 3 with modified parameters
3. Skip expensive filtering and binning steps

### 3. Analysis
```bash
# See how many Q/A pairs were removed in validation
cat output/intermediate/stage4_validation/summary.json

# Compare generated vs validated Q/A counts
cat output/intermediate/stage3_synthesis/summary.json
```

### 4. Reprocessing
Want to try different validation thresholds?
1. Use `stage3_synthesis/generated_qa_pairs.jsonl` as input
2. Run validation with new thresholds
3. No need to regenerate Q/A pairs

## Testing

Run the test suite to verify functionality:
```bash
python test_intermediate_saves_simple.py
```

Expected output:
```
✓✓✓ ALL TESTS PASSED! ✓✓✓
```

## Performance Impact

- **Disk space**: ~2-3x the final dataset size (stores intermediate versions)
- **Time overhead**: <5% (file I/O is fast compared to model inference)
- **Memory**: No impact (saves happen after each stage completes)

## Files Modified

1. `team1_pipeline.py` - Added 4 save methods and integrated into run()
2. `README.md` - Documented output structure and feature usage
3. `test_intermediate_saves_simple.py` - Test suite for verification

## Backward Compatibility

✓ Fully backward compatible - existing code continues to work
✓ Feature can be disabled via config
✓ No changes to API or method signatures
