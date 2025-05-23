import os
import pickle
import json
import logging
import torch
from tqdm import tqdm
from typing import Dict, List, Any

from models.embedding.clip_extractor import CLIPExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_flower_metadata(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_embeddings(metadata: List[Dict[str, Any]], 
                       image_dir: str, 
                       output_path: str,
                       batch_size: int = 16):

    clip_extractor = CLIPExtractor()
    logger.info("Initialized CLIP extractor")

    embeddings = {}

    progress_bar = tqdm(metadata, desc="Extracting CLIP embeddings")

    for flower in progress_bar:
        flower_id = flower['id']
        image_path = os.path.join(image_dir, f"{flower_id}.jpg")
        description = flower['description']
 
        if not os.path.exists(image_path):
            logger.warning(f"Image not found for {flower_id}: {image_path}")
            continue
        
        try:
            img_embedding, text_embedding = clip_extractor.extract_embeddings(
                image_path, description
            )

            embeddings[flower_id] = {
                'image_embedding': img_embedding,
                'text_embedding': text_embedding,
                'metadata': flower
            }

            progress_bar.set_postfix({'flower': flower['name']})
            
        except Exception as e:
            logger.error(f"Error extracting embeddings for {flower_id}: {e}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    logger.info(f"Extracted and saved CLIP embeddings for {len(embeddings)} flowers to {output_path}")


def main():
    logger.info("Starting CLIP embeddings extraction")

    metadata_path = "data/metadata/flowers_metadata.json"
    image_dir = "data/flowers"
    output_path = "data/cache/flower_embeddings.pkl"

    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return

    logger.info(f"Loading flower metadata from {metadata_path}")
    metadata = load_flower_metadata(metadata_path)
    logger.info(f"Loaded metadata for {len(metadata)} flowers")

    if not os.path.exists(image_dir):
        logger.error(f"Image directory not found: {image_dir}")
        return    

    extract_embeddings(metadata, image_dir, output_path)
    
    logger.info("CLIP embeddings extraction completed successfully")

if __name__ == "__main__":
    main()