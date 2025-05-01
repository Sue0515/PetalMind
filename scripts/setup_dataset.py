import os 
import json 
import logging
import random 
import shutil 

from typing import Dict, List, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# def create_directory_structure():
#     directories = [
#         "data/cache",
#         "data/flowers",
#         "data/generated",
#         "data/metadata",
#         "models/weights",
#     ]
    
#     for directory in directories:
#         os.makedirs(directory, exist_ok=True)
#         logger.info(f"Created directory: {directory}")


def load_flower_metadata(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def verify_flower_images(metadata: List[Dict[str, Any]], image_dir: str) -> List[str]:
    missing_images = []
    for flower in metadata:
        flower_id = flower['id']
        image_path = os.path.join(image_dir, f"{flower_id}.jpg")

        if not os.path.exists(image_path):
            missing_images.append(flower_id)

    return missing_images 


def color_name_to_rgb(color_name: str) -> tuple:
    from models.recommendation.color_harmony import color_name_to_rgb
    return color_name_to_rgb(color_name)


#### WORK IN PROGRESS 