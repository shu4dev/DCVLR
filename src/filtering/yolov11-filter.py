from ultralytics import YOLO
from datasets import load_dataset
import random
import json
import time
import numpy as np
from pathlib import Path

import os

"""## Setup"""

# Check if test images already exist
data_path = 'data'
test_dir = Path(data_path)
existing_images = list(test_dir.glob("*.jpg")) if test_dir.exists() else []

if existing_images:
    print(f"✅ Found existing {len(existing_images)} images in {test_dir}")
    print(f"Skipping dataset download and image generation...")
    print(f"Using images from: {test_dir.absolute()}")
else:
    # Load MathVista dataset
    print("Loading MathVista dataset...")
    dataset = load_dataset("AI4Math/MathVista", split="testmini")

    # Sample random images
    sample_size = 10
    sampled_indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    sampled_data = [dataset[i] for i in sampled_indices]

    # Save images locally for testing
    test_dir.mkdir(exist_ok=True)

    print(f"\nSaving {len(sampled_data)} images to {test_dir}...")

    for idx, sample in enumerate(sampled_data):
        # Use decoded_image field instead of image
        img = sample['decoded_image']

        if img is None:
            print(f"Warning: Image {idx} is None, skipping...")
            continue

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save the image
        img.save(test_dir / f"img_{idx:02d}.jpg")

        # Save metadata
        metadata = {
            'pid': sample.get('pid', ''),
            'question': sample.get('question', ''),
            'query': sample.get('query', ''),
            'answer': sample.get('answer', ''),
            'question_type': sample.get('question_type', ''),
            'answer_type': sample.get('answer_type', ''),
            'metadata': sample.get('metadata', {})
        }
        with open(test_dir / f"img_{idx:02d}.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f"✅ Saved {len(list(test_dir.glob('*.jpg')))} images to {test_dir}")
    print(f"Images saved to: {test_dir.absolute()}")

images = list(test_dir.glob("*.jpg"))

# === LOAD MODELS ===
print("LOADING MODELS...")

# --- YOLO ---
yolo_models = {
    'YOLOv8s': YOLO('yolov8s.pt'),
    'YOLOv9s': YOLO('yolov9s.pt'),
    'YOLOv10s': YOLO('yolov10s.pt'),
    'YOLOv11s': YOLO('yolo11s.pt'),
}
print("Loaded YOLO v8, v9, v10, v11")

# TESTING YOLO MODELS
def test_yolo_model(yolo_model, model_name, images):
    print("\n" + "="*50)
    print(f"TESTING {model_name} ON MATH IMAGES")
    print("="*50)

    results_list = []
    times = []

    for img_path in images:
        start = time.time()
        results = yolo_model(img_path, verbose=False)[0]
        times.append(time.time() - start)

        boxes = results.boxes
        classes = boxes.cls.cpu().numpy() if len(boxes) > 0 else []
        centers = boxes.xywh[:, :2].cpu().numpy() if len(boxes) > 0 else np.array([])

        unique_classes = len(np.unique(classes))
        total_instances = len(boxes)

        # Spatial dispersion
        if len(centers) > 1:
            img_diag = np.sqrt(results.orig_shape[0]**2 + results.orig_shape[1]**2)
            spatial_dispersion = np.std(centers, axis=0).mean() / img_diag
        else:
            spatial_dispersion = 0

        # Check filtering criteria
        passes_filter = (unique_classes > 3 or total_instances > 5 or spatial_dispersion > 0.3)

        results_list.append({
            'image': img_path.name,
            'unique_classes': unique_classes,
            'total_instances': total_instances,
            'spatial_dispersion': spatial_dispersion,
            'passes_filter': passes_filter
        })

    print(f"{model_name} - Avg time per image: {np.mean(times):.3f}s")
    print(f"{model_name} - Total time: {sum(times):.3f}s")
    print(f"{model_name} - Images passing filter: {sum(r['passes_filter'] for r in results_list)}/{len(results_list)}")

    return results_list, times

yolo_results_all = {}

for model_name, yolo_model in yolo_models.items():
    results_list, times = test_yolo_model(yolo_model, model_name, images)
    yolo_results_all[model_name] = {'results': results_list, 'times': times}

print("\n✅ Finished YOLO model inference and collected results for all models.")

"""## Combine All Results and Save to CSV

"""

all_results = []

# Process YOLO results
for model_name, data in yolo_results_all.items():
    avg_time = np.mean(data['times'])
    for res in data['results']:
        res['model_type'] = 'YOLO'
        res['model_name'] = model_name
        res['avg_inference_time'] = avg_time
        all_results.append(res)

print("✅ Combined YOLO and SAM results into a single list.")

import pandas as pd

df_results = pd.DataFrame(all_results)

# Define the base path to save CSV files in Google Drive
base_save_path = Path(directory_path)

# Save the combined DataFrame to a general CSV file (as done before)
csv_combined_path = base_save_path / "all_model_inference_results.csv"
df_results.to_csv(csv_combined_path, index=False)
print(f"✅ Combined results saved to {csv_combined_path}")

# Save individual model results to separate CSV files
print("\nSaving individual model results...")
for model_name in df_results['model_name'].unique():
    model_df = df_results[df_results['model_name'] == model_name]
    individual_csv_path = base_save_path / f"{model_name.lower().replace(' ', '_')}_results.csv"
    model_df.to_csv(individual_csv_path, index=False)
    print(f"✅ Results for {model_name} saved to {individual_csv_path}")

print("\nDataFrame head (combined results):\n")
print(df_results.head())