# from blip2 import generate_captions
from datasets import load_dataset
from PIL import Image
import requests

print("Imports COMPLETE")

img_urls = ['https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png', 'https://www.georgiaaquarium.org/wp-content/uploads/2018/09/whale-shark-3-1024x323.jpg', 'https://www.wardvillage.com/wp-content/uploads/2019/11/Ward16WebsiteRefresh04.2024-scaled.jpg', 'https://upload.wikimedia.org/wikipedia/commons/1/11/Queen_Elizabeth_II_official_portrait_for_1959_tour_%28retouched%29_%28cropped%29_%283-to-4_aspect_ratio%29.jpg', 'https://starwalk.space/gallery/images/what-is-space/750x422.jpg']

db_names = ["kakaobrain/coyo-700m", "HuggingFaceM4/ChartQA", "derek-thomas/ScienceQA", "vidore/infovqa_train", "lmms-lab/multimodal-open-r1-8k-verified", "Zhiqiang007/MathV360K", "Luckyjhg/Geo170K", "HuggingFaceM4/OBELICS"]

# "kakaobrain/coyo-700m" <-- 'url' not 'image'
# datacomp <-- must download file from github to then download dataset
# laion5b-downloader <-- must also actually download images
# VCR <-- must also be downloaded

# Load the given amount samples from the train split
for name in db_names:
    try:
        ds = load_dataset(name, split="train", streaming=True)
        ds_objects = [sample for _, sample in zip(range(10), ds)]
        print("Dataset ", name, " DOWNLOADED")
        db_imgs = [] 
   
        try:
            if (name == "kakaobrain/coyo-700m"):
                db_urls = [item['url'] for item in ds_objects]  # This puts all URLs into a Python list
        
                for url in db_urls:
                    img = Image.open(requests.get(url, stream=True).raw).convert('RGB')        
                    db_imgs.append(img)

            else: 
                db_imgs = [item['image'] for item in ds_objects]  # This puts all 30 URLs into a Python list

            print("Images for Database: ", name)
            for img in db_imgs:
                print(img)
                print("----------------------------------------------------------")

        except Exception as e:
            print(f"Failed to load {name}: {e}")
    except Exception as e:
        print(f"Failed to load {name}: {e}")
    print("==========================================================")





#generate_captions(db_img_urls)


