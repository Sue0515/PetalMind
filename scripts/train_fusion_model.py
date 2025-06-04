import os
import pickle
import json
import logging
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from typing import Dict, List, Any, Tuple

from models.embedding.fusion import EmbeddingFusion
from models.metric_learning.trainer import MetricLearningTrainer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FlowerEmbeddingDataset(Dataset):
    """꽃 임베딩 데이터셋"""
    
    def __init__(self, embeddings_path: str, harmony_groups_path: str):

        with open(embeddings_path, 'rb') as f:
            self.embeddings = pickle.load(f)

        with open(harmony_groups_path, 'r', encoding='utf-8') as f:
            self.harmony_groups = json.load(f)

        self.harmony_group_ids = {}
        for idx, group in enumerate(self.harmony_groups):
            # 조화 그룹의 모든 꽃에 그룹 ID 할당
            all_flowers = [group['main_flower']] + group['medium_flowers'] + group['small_flowers']
            for flower_id in all_flowers:
                if flower_id not in self.harmony_group_ids:
                    self.harmony_group_ids[flower_id] = []
                self.harmony_group_ids[flower_id].append(idx)

        self.samples = []

        for flower_id, embedding_data in self.embeddings.items():
            img_embedding = embedding_data['image_embedding']
            text_embedding = embedding_data['text_embedding']

            group_ids = self.harmony_group_ids.get(flower_id, [])

            if group_ids:
                label = group_ids[0]
                
                self.samples.append({
                    'flower_id': flower_id,
                    'img_embedding': img_embedding,
                    'text_embedding': text_embedding,
                    'label': label
                })
        
        logger.info(f"Created dataset with {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_embedding = torch.tensor(sample['img_embedding'], dtype=torch.float32)
        text_embedding = torch.tensor(sample['text_embedding'], dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return img_embedding, text_embedding, label

def prepare_dataloaders(dataset: Dataset, batch_size: int, 
                        train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
   
    train_size = int(train_ratio * len(dataset))
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
    
    return train_loader, val_loader

def main():
    logger.info("Starting embedding fusion model training")

    embeddings_path = "data/cache/flower_embeddings.pkl"
    harmony_groups_path = "data/metadata/harmony_groups.json"
    model_save_path = "models/weights/embedding_fusion.pth"

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

    logger.info("Creating flower embedding dataset")
    dataset = FlowerEmbeddingDataset(embeddings_path, harmony_groups_path)

    if len(dataset) == 0:
        logger.error("Dataset is empty. Please check your data files.")
        return

    logger.info("Preparing data loaders")
    train_loader, val_loader = prepare_dataloaders(dataset, batch_size)
    logger.info(f"Created data loaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")
 
    logger.info("Initializing metric learning trainer")
    trainer = MetricLearningTrainer(
        model_path=None, 
        img_dim=512,     
        text_dim=512,    
        output_dim=512,   
        device=device
    )

    logger.info(f"Starting training for {epochs} epochs on {device}")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        save_path=model_save_path
    )

    logger.info(f"Training completed. Best validation loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']}")
    logger.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()