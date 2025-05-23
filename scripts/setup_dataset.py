import os
import json
import logging
import random
from typing import Dict, List, Any
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directory_structure():
    directories = [
        "data/cache",
        "data/flowers",
        "data/generated",
        "data/metadata",
        "models/weights",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


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


def is_harmonious(color1: str, color2: str) -> float:
    from models.recommendation.color_harmony import is_harmonious_color
    return is_harmonious_color(color1, color2)


def create_harmony_groups(metadata: List[Dict[str, Any]], output_path: str) -> List[Dict[str, Any]]:
    flowers_by_size = {
        'large': [],
        'medium': [],
        'small': []
    }

    flowers_map = {flower['id']: flower for flower in metadata}

    for flower in metadata:
        size = flower['size'].lower()
        if size in flowers_by_size:
            flowers_by_size[size].append(flower['id'])

    harmony_groups = []
    
    # 각 대형 꽃을 기준으로 조화로운 그룹 생성
    for large_id in flowers_by_size['large']:
        large_flower = flowers_map[large_id]
        large_color = large_flower['color'].lower()
        
        # 대형 꽃과 조화로운 중형 꽃 찾기
        harmonious_medium = []
        for medium_id in flowers_by_size['medium']:
            medium_flower = flowers_map[medium_id]
            medium_color = medium_flower['color'].lower()
 
            harmony_score = is_harmonious(large_color, medium_color)
            
            # 충분히 조화로운 경우 추가
            if harmony_score >= 0.7:
                harmonious_medium.append(medium_id)
        
        # 대형 꽃과 조화로운 소형 꽃 찾기
        harmonious_small = []
        for small_id in flowers_by_size['small']:
            small_flower = flowers_map[small_id]
            small_color = small_flower['color'].lower()

            harmony_score = is_harmonious(large_color, small_color)

            if harmony_score >= 0.7:
                harmonious_small.append(small_id)
        
        num_groups = min(5, len(harmonious_medium) * len(harmonious_small))
        
        for _ in range(num_groups):
            num_medium = random.randint(1, min(2, len(harmonious_medium)))
            selected_medium = random.sample(harmonious_medium, num_medium)

            num_small = random.randint(1, min(3, len(harmonious_small)))
            selected_small = random.sample(harmonious_small, num_small)

            harmony_score = 0.0
            count = 0

            all_flowers = [large_id] + selected_medium + selected_small
            for i in range(len(all_flowers)):
                for j in range(i+1, len(all_flowers)):
                    flower1 = flowers_map[all_flowers[i]]
                    flower2 = flowers_map[all_flowers[j]]
                    score = is_harmonious(flower1['color'].lower(), flower2['color'].lower())
                    harmony_score += score
                    count += 1

            avg_harmony = harmony_score / count if count > 0 else 0.5

            harmony_group = {
                "main_flower": large_id,
                "medium_flowers": selected_medium,
                "small_flowers": selected_small,
                "harmony_score": round(min(1.0, avg_harmony), 2)
            }
            
            harmony_groups.append(harmony_group)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(harmony_groups, f, indent=2)
    
    logger.info(f"Created {len(harmony_groups)} harmony groups in {output_path}")
    
    return harmony_groups


def copy_sample_flower_images(output_dir: str):

    logger.info(f"In a real project, this function would download flower images to {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    
    sample_dir = "sample_images"
    if os.path.exists(sample_dir):
        for img_file in os.listdir(sample_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(
                    os.path.join(sample_dir, img_file),
                    os.path.join(output_dir, img_file)
                )
        logger.info(f"Copied sample images to {output_dir}")


def main():

    logger.info("Starting dataset setup")

    create_directory_structure()

    metadata_path = "data/metadata/flowers_metadata.json"

    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        logger.info("Please create a flowers_metadata.json file in the data/metadata directory")
        return

    logger.info(f"Loading flower metadata from {metadata_path}")
    metadata = load_flower_metadata(metadata_path)
    logger.info(f"Loaded metadata for {len(metadata)} flowers")

    image_dir = "data/flowers"

    copy_sample_flower_images(image_dir)

    missing_images = verify_flower_images(metadata, image_dir)
    if missing_images:
        logger.warning(f"Missing images for {len(missing_images)} flowers: {missing_images}")


    harmony_groups_path = "data/metadata/harmony_groups.json"
    logger.info(f"Creating harmony groups in {harmony_groups_path}")
    harmony_groups = create_harmony_groups(metadata, harmony_groups_path)
    
    logger.info("Dataset setup completed successfully")

if __name__ == "__main__":
    main()