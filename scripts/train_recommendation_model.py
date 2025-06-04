import os
import pickle
import json
import logging
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional

from models.embedding.fusion import EmbeddingFusion
from models.recommendation.cross_attention import CrossAttentionRecommender

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FlowerRecommendationDataset(Dataset):
    """꽃 추천 모델 훈련을 위한 데이터셋"""
    
    def __init__(self, embeddings_path: str, harmony_groups_path: str, 
                 fusion_model_path: Optional[str] = None, device: Optional[str] = None):
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        with open(embeddings_path, 'rb') as f:
            self.embeddings = pickle.load(f)
        
        with open(harmony_groups_path, 'r', encoding='utf-8') as f:
            self.harmony_groups = json.load(f)
        
        self.fusion_model = None
        if fusion_model_path and os.path.exists(fusion_model_path):
            self.fusion_model = EmbeddingFusion(
                img_dim=512,
                text_dim=512,
                output_dim=512
            ).to(self.device)
            
            self.fusion_model.load_state_dict(torch.load(fusion_model_path, map_location=self.device))
            self.fusion_model.eval()
            logger.info(f"Loaded fusion model from {fusion_model_path}")
        
        self.fused_embeddings = {}
        
        with torch.no_grad():
            for flower_id, embedding_data in self.embeddings.items():
                if self.fusion_model:
                    img_emb = torch.tensor(embedding_data['image_embedding'], dtype=torch.float32).unsqueeze(0).to(self.device)
                    txt_emb = torch.tensor(embedding_data['text_embedding'], dtype=torch.float32).unsqueeze(0).to(self.device)

                    fused_emb = self.fusion_model(img_emb, txt_emb).cpu().numpy().squeeze()
                    self.fused_embeddings[flower_id] = fused_emb
                else:
                    # 퓨전 모델이 없으면 이미지 임베딩 사용
                    self.fused_embeddings[flower_id] = embedding_data['image_embedding']
        
        self.samples = []

        self._create_positive_pairs()
        self._create_negative_pairs()
        
        logger.info(f"Created dataset with {len(self.samples)} samples "
                   f"({sum(s['label'] > 0.5 for s in self.samples)} positive, "
                   f"{sum(s['label'] < 0.5 for s in self.samples)} negative)")
    
    def _create_positive_pairs(self):
        """조화로운 꽃 조합으로부터 positive pair 생성"""
        for group in self.harmony_groups:
            main_id = group['main_flower']
            medium_ids = group['medium_flowers']
            small_ids = group['small_flowers']
            harmony_score = group['harmony_score']
            
            # 메인 꽃 임베딩이 있는지 확인
            if main_id not in self.fused_embeddings:
                continue
            
            # 메인-중형 쌍
            for medium_id in medium_ids:
                if medium_id in self.fused_embeddings:
                    self.samples.append({
                        'main_id': main_id,
                        'other_id': medium_id,
                        'label': harmony_score,  
                        'is_positive': True
                    })
            
            # 메인-소형 쌍
            for small_id in small_ids:
                if small_id in self.fused_embeddings:
                    self.samples.append({
                        'main_id': main_id,
                        'other_id': small_id,
                        'label': harmony_score, 
                        'is_positive': True
                    })
    
    def _create_negative_pairs(self):
        all_flower_ids = list(self.fused_embeddings.keys())

        compatible_flowers = {}
        for group in self.harmony_groups:
            main_id = group['main_flower']
            if main_id not in compatible_flowers:
                compatible_flowers[main_id] = set()

            for flower_id in group['medium_flowers'] + group['small_flowers']:
                compatible_flowers[main_id].add(flower_id)
        
        for main_id in compatible_flowers:
            # 이 메인 꽃과 호환되지 않는 꽃 목록
            incompatible = [fid for fid in all_flower_ids if 
                            fid != main_id and fid not in compatible_flowers[main_id]]
            
            # 최대 3개의 음성 샘플 추가
            num_negatives = min(len(incompatible), 3)
            if num_negatives > 0:
                for other_id in random.sample(incompatible, num_negatives):
                    self.samples.append({
                        'main_id': main_id,
                        'other_id': other_id,
                        'label': 0.0,  # 0점 (호환되지 않음)
                        'is_positive': False
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        main_embedding = self.fused_embeddings[sample['main_id']]
        other_embedding = self.fused_embeddings[sample['other_id']]

        main_embedding = torch.tensor(main_embedding, dtype=torch.float32)
        other_embedding = torch.tensor(other_embedding, dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.float32)
        
        return main_embedding, other_embedding, label

def train_recommender(train_loader: DataLoader, val_loader: DataLoader,
                      model: torch.nn.Module, device: torch.device,
                      epochs: int = 50, lr: float = 0.001,
                      save_path: str = None) -> Dict[str, Any]:

    model = model.to(device)
 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
 
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for main_emb, other_emb, labels in progress_bar:
            main_emb = main_emb.to(device)
            other_emb = other_emb.to(device)
            labels = labels.to(device)

            pred_scores = model(main_emb, other_emb)

            loss = criterion(pred_scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            progress_bar.set_postfix({"loss": loss.item()})

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for main_emb, other_emb, labels in val_loader:
                main_emb = main_emb.to(device)
                other_emb = other_emb.to(device)
                labels = labels.to(device)

                pred_scores = model(main_emb, other_emb)

                loss = criterion(pred_scores, labels)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch + 1
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
 
                torch.save(model.state_dict(), save_path)
                logger.info(f"Saved best model at epoch {epoch+1} with validation loss {val_loss:.4f}")
    
    return history

def main():

    logger.info("Starting recommendation model training")

    embeddings_path = "data/cache/flower_embeddings.pkl"
    harmony_groups_path = "data/metadata/harmony_groups.json"
    fusion_model_path = "models/weights/embedding_fusion.pth"
    recommender_save_path = "models/weights/cross_attention_recommender.pth"

    batch_size = 32
    learning_rate = 0.001
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(embeddings_path):
        logger.error(f"Embeddings file not found: {embeddings_path}")
        return
    
    if not os.path.exists(harmony_groups_path):
        logger.error(f"Harmony groups file not found: {harmony_groups_path}")
        return

    if not os.path.exists(fusion_model_path):
        logger.warning(f"Fusion model not found: {fusion_model_path}")
        logger.warning("Will proceed using original image embeddings instead of fused embeddings")
        fusion_model_path = None

    logger.info("Creating flower recommendation dataset")
    dataset = FlowerRecommendationDataset(
        embeddings_path=embeddings_path,
        harmony_groups_path=harmony_groups_path,
        fusion_model_path=fusion_model_path,
        device=device
    )

    if len(dataset) == 0:
        logger.error("Dataset is empty. Please check your data files.")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    logger.info(f"Created data loaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")

    logger.info("Initializing cross-attention recommender model")
    model = CrossAttentionRecommender(
        dim=512,
        heads=8,
        dropout=0.1,
        layers=2
    )

    logger.info(f"Starting training for {epochs} epochs on {device}")
    history = train_recommender(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device,
        epochs=epochs,
        lr=learning_rate,
        save_path=recommender_save_path
    )
    
    # 훈련 결과 요약
    logger.info(f"Training completed. Best validation loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']}")
    logger.info(f"Model saved to {recommender_save_path}")

if __name__ == "__main__":
    main()