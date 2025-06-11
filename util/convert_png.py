import os
from PIL import Image
import pillow_avif  

input_dir = "/Users/suincho/code/src/github.com/Sue0515/PetalMind/data/flowers"  
image_exts = (".jpg", ".jpeg", ".png", ".webp", ".avif")

for filename in os.listdir(input_dir):
    if filename.lower().endswith(image_exts):
        input_path = os.path.join(input_dir, filename)
        name_wo_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(input_dir, f"{name_wo_ext}.png")

        if os.path.exists(output_path):
            continue 
        try:
            with Image.open(input_path) as img: 
                img.convert("RGBA").save(output_path, "PNG")
                print(f"Converted: {filename} → {output_path}")
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")
