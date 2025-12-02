import sys
from PIL import Image, ImageOps
import requests
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration


def generate_captions(images, isurl):
    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # Load the BLIP-2 processor and model from Hugging Face Hub
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

    # Set pad_token if not already set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    model.to(device)




    i = 1
    for img in images:
        print("====================================================")
        print("Processing image #", i)

        try:
            #if(isurl):
            #    raw_image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
            #else:

            # Prepare image for the model: processor does all preprocessing internally
            inputs = processor(images=img, return_tensors="pt").to(device, torch.float16)

            # Generate a caption using the model
            generated_ids = model.generate(
                **inputs,
                pad_token_id=processor.tokenizer.pad_token_id,
                attention_mask=inputs.get('attention_mask', None)
            )

            # Decode generated output IDs to a readable string caption
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()     
            print("Caption generated: ", caption)
        except Exception as e:
            print(f"ERROR occurred: {e}")

        print("====================================================")
        i += 1
    
    print("All images have been processed")
