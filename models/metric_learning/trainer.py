import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple

from models.embedding.fusion import EmbeddingFusion
from models.metric_learning.triplet_loss import BatchHardTripletLoss

class MetricLearningTrainer:
    """Train embedding fusion model w metric learning """
    
    def __init__(self, model_path: Optional[str] = None, img_dim: int = 512, 
                 text_dim: int = 512, output_dim: int = 512, device: str = None):
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = EmbeddingFusion(
            img_dim=img_dim,
            text_dim=text_dim,
            output_dim=output_dim
        ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded pre-trained model from {model_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 50, lr: float = 0.001, save_path: str = None) -> Dict[str, Any]:

        triplet_loss = BatchHardTripletLoss(margin=1.0).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_batches = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

            for img_batch, txt_batch, labels in progress_bar:
                img_batch = img_batch.to(self.device)
                txt_batch = txt_batch.to(self.device)
                labels = labels.to(self.device)
                
                fused_embeddings = self.model(img_batch, txt_batch)
                loss = triplet_loss(fused_embeddings, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1
                
                progress_bar.set_postfix({"loss": loss.item()})

            train_loss /= train_batches
            history['train_loss'].append(train_loss)

            val_loss, val_metrics = self._validate(val_loader, triplet_loss)
            history['val_loss'].append(val_loss)

            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Metrics: Active Triplets: {val_metrics['active_triplets']}, "
                  f"Pos Dist: {val_metrics['pos_dist_mean']:.4f}, "
                  f"Neg Dist: {val_metrics['neg_dist_mean']:.4f}")

            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['best_epoch'] = epoch + 1
                
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Saved best model at epoch {epoch+1} with validation loss {val_loss:.4f}")

        print("Training completed!")
        print(f"Best validation loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']}")
        
        return history


    def _validate(self, val_loader: DataLoader, 
                  criterion: nn.Module) -> Tuple[float, Dict[str, float]]:

        self.model.eval()
        val_loss = 0.0
        val_batches = 0

        metrics = {
            'pos_dist_sum': 0.0,
            'neg_dist_sum': 0.0,
            'active_triplets': 0
        }

        progress_bar = tqdm(val_loader, desc="Validation")
        
        with torch.no_grad():
            for img_batch, txt_batch, labels in progress_bar:
                img_batch = img_batch.to(self.device)
                txt_batch = txt_batch.to(self.device)
                labels = labels.to(self.device)

                fused_embeddings = self.model(img_batch, txt_batch)
                distance_matrix = torch.cdist(fused_embeddings, fused_embeddings, p=2)

                same_label_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
                identity_mask = torch.eye(len(labels), device=self.device).bool()
                
                positive_mask = same_label_mask & (~identity_mask)
                negative_mask = ~same_label_mask
                
                pos_dist_mean = distance_matrix[positive_mask].mean().item() if positive_mask.sum() > 0 else 0
                neg_dist_mean = distance_matrix[negative_mask].mean().item() if negative_mask.sum() > 0 else 0
                
                metrics['pos_dist_sum'] += pos_dist_mean
                metrics['neg_dist_sum'] += neg_dist_mean
                
                loss = criterion(fused_embeddings, labels)
                
                with torch.no_grad():
                    batch_size = img_batch.size(0)
                    max_dist = torch.max(distance_matrix).item()
                    
                    positive_dist = distance_matrix * positive_mask.float() + (1.0 - positive_mask.float()) * 1e-9
                    hardest_positive_dist, _ = torch.max(positive_dist, dim=1)
                    
                    negative_dist = distance_matrix * negative_mask.float() + (1.0 - negative_mask.float()) * max_dist
                    hardest_negative_dist, _ = torch.min(negative_dist, dim=1)
                    
                    active_triplets = ((hardest_positive_dist - hardest_negative_dist + 1.0) > 0).sum().item()
                    metrics['active_triplets'] += active_triplets

                val_loss += loss.item()
                val_batches += 1
                progress_bar.set_postfix({"loss": loss.item()})
        
        val_loss /= val_batches
        
        final_metrics = {
            'pos_dist_mean': metrics['pos_dist_sum'] / val_batches,
            'neg_dist_mean': metrics['neg_dist_sum'] / val_batches,
            'active_triplets': metrics['active_triplets']
        }
        
        return val_loss, final_metrics
    
    
    def predict(self, img_embedding: torch.Tensor, txt_embedding: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        
        if isinstance(img_embedding, np.ndarray):
            img_embedding = torch.tensor(img_embedding, dtype=torch.float32)
        if isinstance(txt_embedding, np.ndarray):
            txt_embedding = torch.tensor(txt_embedding, dtype=torch.float32)
        
        if img_embedding.dim() == 1:
            img_embedding = img_embedding.unsqueeze(0)
        if txt_embedding.dim() == 1:
            txt_embedding = txt_embedding.unsqueeze(0)
        
        img_embedding = img_embedding.to(self.device)
        txt_embedding = txt_embedding.to(self.device)

        with torch.no_grad():
            fused_embedding = self.model(img_embedding, txt_embedding)
        
        return fused_embedding