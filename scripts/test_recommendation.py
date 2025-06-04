import os
import pickle
import json
import logging
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Any, Tuple, Optional

from models.embedding.fusion import EmbeddingFusion
from models.recommendation.cross_attention import CrossAttentionRecommender
from models.recommendation.color_harmony import evaluate_flower_harmony
from models.generation.diffusion_generator import FlowerArrangementGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_models(fusion_model_path: str, recommender_model_path: str, device: torch.device) -> Tuple:
    fusion_model = None

    if os.path.exists(fusion_model_path):
        fusion_model = EmbeddingFusion(
            img_dim=512,
            text_dim=512,
            output_dim=512
        ).to(device)
        
        fusion_model.load_state_dict(torch.load(fusion_model_path, map_location=device))
        fusion_model.eval()
        logger.info(f"Loaded fusion model from {fusion_model_path}")
    else:
        logger.warning(f"Fusion model not found: {fusion_model_path}")

    recommender_model = None
    if os.path.exists(recommender_model_path):
        recommender_model = CrossAttentionRecommender(
            dim=512,
            heads=8,
            dropout=0.1,
            layers=2
        ).to(device)
        
        recommender_model.load_state_dict(torch.load(recommender_model_path, map_location=device))
        recommender_model.eval()
        logger.info(f"Loaded recommender model from {recommender_model_path}")
    else:
        logger.warning(f"Recommender model not found: {recommender_model_path}")
    
    return fusion_model, recommender_model


def get_flower_embeddings(flower_id: str, embeddings: Dict) -> Tuple[np.ndarray, np.ndarray]:
    if flower_id not in embeddings:
        raise ValueError(f"Flower ID {flower_id} not found in embeddings")
    
    embedding_data = embeddings[flower_id]
    return embedding_data['image_embedding'], embedding_data['text_embedding']


def get_fused_embedding(flower_id: str, embeddings: Dict, fusion_model: torch.nn.Module, 
                         device: torch.device) -> np.ndarray:

    img_emb, txt_emb = get_flower_embeddings(flower_id, embeddings)

    img_emb = torch.tensor(img_emb, dtype=torch.float32).unsqueeze(0).to(device)
    txt_emb = torch.tensor(txt_emb, dtype=torch.float32).unsqueeze(0).to(device)
 
    with torch.no_grad():
        fused_emb = fusion_model(img_emb, txt_emb).cpu().numpy().squeeze()
    
    return fused_emb


def predict_harmony_score(main_emb: np.ndarray, other_emb: np.ndarray, 
                          recommender_model: torch.nn.Module, device: torch.device) -> float:

    main_emb = torch.tensor(main_emb, dtype=torch.float32).unsqueeze(0).to(device)
    other_emb = torch.tensor(other_emb, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        score = recommender_model(main_emb, other_emb).cpu().item()
    
    return score


def recommend_flowers(main_flower_id: str, metadata: List[Dict], embeddings: Dict,
                     fusion_model: torch.nn.Module, recommender_model: torch.nn.Module,
                     device: torch.device, num_medium: int = 2, num_small: int = 3) -> Dict:
   
    flowers_map = {flower['id']: flower for flower in metadata}

    main_flower = flowers_map.get(main_flower_id)
    if not main_flower:
        raise ValueError(f"Main flower ID {main_flower_id} not found in metadata")

    main_fused_emb = get_fused_embedding(main_flower_id, embeddings, fusion_model, device)
    
    medium_flowers = [f for f in metadata if f['size'].lower() == 'medium']
    small_flowers = [f for f in metadata if f['size'].lower() == 'small']
    
    recommendations = {
        'main_flower': main_flower,
        'medium_flowers': [],
        'small_flowers': [],
        'harmony_score': 0.0
    }
    
    medium_scores = []
    for flower in medium_flowers:
        flower_id = flower['id']
        # 자기 자신이면 건너뛰기
        if flower_id == main_flower_id:
            continue
        
        try:
            flower_fused_emb = get_fused_embedding(flower_id, embeddings, fusion_model, device)
            score = predict_harmony_score(main_fused_emb, flower_fused_emb, recommender_model, device) 
            medium_scores.append((flower, score))

        except Exception as e:
            logger.warning(f"Error processing flower {flower_id}: {e}")

    medium_scores.sort(key=lambda x: x[1], reverse=True)
    top_medium = medium_scores[:num_medium]
    recommendations['medium_flowers'] = [item[0] for item in top_medium]


    small_scores = []
    for flower in small_flowers:
        flower_id = flower['id']
        # 자기 자신이면 건너뛰기
        if flower_id == main_flower_id:
            continue
        
        try:
            flower_fused_emb = get_fused_embedding(flower_id, embeddings, fusion_model, device)
            score = predict_harmony_score(main_fused_emb, flower_fused_emb, recommender_model, device)
            
            small_scores.append((flower, score))
        except Exception as e:
            logger.warning(f"Error processing flower {flower_id}: {e}")
    
    small_scores.sort(key=lambda x: x[1], reverse=True)
    top_small = small_scores[:num_small]
    recommendations['small_flowers'] = [item[0] for item in top_small]

    color_harmony_score = evaluate_flower_harmony(
        main_flower, 
        recommendations['medium_flowers'],
        recommendations['small_flowers']
    )
    
    # NN 점수와 색상 조화 점수를 결합
    neural_scores = [score for _, score in top_medium] + [score for _, score in top_small]
    neural_harmony_score = sum(neural_scores) / len(neural_scores) if neural_scores else 0.5
    
    
    # 최종 조화 점수 (신경망 70%, rule based 30%)
    recommendations['harmony_score'] = 0.7 * neural_harmony_score + 0.3 * color_harmony_score
    
    return recommendations

def generate_arrangement_image(recommendation: Dict, generator: FlowerArrangementGenerator, 
                              style: str = "natural", seed: Optional[int] = None) -> Tuple[Image.Image, str]:

    main_flower = recommendation['main_flower']
    medium_flowers = recommendation['medium_flowers']
    small_flowers = recommendation['small_flowers']

    image, prompt = generator.generate_image(
        main_flower=main_flower,
        medium_flowers=medium_flowers,
        small_flowers=small_flowers,
        style=style,
        seed=seed
    )
    
    return image, prompt

def display_recommendation(recommendation: Dict, image: Optional[Image.Image] = None, 
                           prompt: Optional[str] = None):

    main_flower = recommendation['main_flower']
    print(f"메인 꽃: {main_flower['name']} ({main_flower['color']})")

    print("중형 꽃:")
    for i, flower in enumerate(recommendation['medium_flowers'], 1):
        print(f"  {i}. {flower['name']} ({flower['color']})")

    print("소형 꽃:")
    for i, flower in enumerate(recommendation['small_flowers'], 1):
        print(f"  {i}. {flower['name']} ({flower['color']})")

    print(f"조화 점수: {recommendation['harmony_score']:.2f}")

    if image:
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    
    if prompt:
        print("이미지 생성 프롬프트:")
        print(prompt)

def main():
    logger.info("Starting recommendation system test")

    embeddings_path = "data/cache/flower_embeddings.pkl"
    metadata_path = "data/metadata/flowers_metadata.json"
    fusion_model_path = "models/weights/embedding_fusion.pth"
    recommender_model_path = "models/weights/cross_attention_recommender.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(embeddings_path):
        logger.error(f"Embeddings file not found: {embeddings_path}")
        return
    
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return

    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded embeddings for {len(embeddings)} flowers and metadata for {len(metadata)} flowers")

    fusion_model, recommender_model = load_models(
        fusion_model_path, recommender_model_path, device
    )

    if fusion_model is None or recommender_model is None:
        logger.error("Required models not found. Please train the models first.")
        return

    use_image_generator = True
    generator = None
    
    if use_image_generator:
        try:
            logger.info("Initializing flower arrangement image generator")
            generator = FlowerArrangementGenerator()
        except Exception as e:
            logger.error(f"Error initializing image generator: {e}")
            use_image_generator = False

    large_flowers = [f for f in metadata if f['size'].lower() == 'large']
    if not large_flowers:
        logger.error("No large flowers found in metadata")
        return
    
    main_flower = random.choice(large_flowers)
    main_flower_id = main_flower['id']
    
    logger.info(f"Selected main flower: {main_flower['name']} (ID: {main_flower_id})")

    try:
        recommendation = recommend_flowers(
            main_flower_id=main_flower_id,
            metadata=metadata,
            embeddings=embeddings,
            fusion_model=fusion_model,
            recommender_model=recommender_model,
            device=device,
            num_medium=2,
            num_small=3
        )
        
        logger.info(f"Generated recommendation with harmony score: {recommendation['harmony_score']:.2f}")

        image = None
        prompt = None
        
        if use_image_generator and generator:
            try:
                logger.info("Generating flower arrangement image")
                image, prompt = generate_arrangement_image(
                    recommendation=recommendation,
                    generator=generator,
                    style="natural"
                )

                image_path = f"data/generated/arrangement_{main_flower_id}.png"
                image.save(image_path)
                logger.info(f"Saved generated image to {image_path}")
                
            except Exception as e:
                logger.error(f"Error generating image: {e}")

        display_recommendation(recommendation, image, prompt)
        
    except Exception as e:
        logger.error(f"Error generating recommendation: {e}")

if __name__ == "__main__":
    main()