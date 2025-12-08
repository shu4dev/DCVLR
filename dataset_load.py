import pandas as pd
from datasets import Dataset, Features, Image, Value

# 1. Load your JSONL
input_file = "output/intermediate/stage4_qa_generation/hard_qa_pairs.jsonl"
print(f"Loading {input_file}...")
df = pd.read_json(input_file, lines=True)

# 2. Create the Dataset
dataset = Dataset.from_pandas(df)

# 3. Define a function to read the image bytes explicitly
def embed_image(example):
    image_path = example["path"]
    try:
        # Read the file bytes into memory
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        # Return the dictionary format expected by HF Image feature
        return {"path": {"bytes": img_bytes, "path": None}}
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return {"path": None} # Handle missing files gracefully

print("Embedding images (reading bytes from disk)...")
# This step forces the data to become binary bytes instead of file paths
dataset = dataset.map(embed_image, num_proc=4) 

# 4. Filter out any images that failed to load
dataset = dataset.filter(lambda x: x["path"] is not None)

# 5. Cast the column to the Image feature
# The data is already in bytes, so this tells HF to treat it as an image
dataset = dataset.cast_column("path", Image())

# 6. Save to Parquet
output_filename = "train.parquet"
print(f"Saving to {output_filename}...")
dataset.to_parquet(output_filename)

print("Success! The 'train.parquet' file is now self-contained.")
print("Upload this new file to Hugging Face and delete the old one.")