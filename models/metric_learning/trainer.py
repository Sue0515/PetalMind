import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json

from models.metric_learning.triplet_loss import TripletLoss, BatchHardTripletLoss
from models.embedding.fusion import EmbeddingFusion

class MetricLearningTrainer:
    def __init__(
            self,
            model_path=None,
            img_dim=512,
            text_dim=512,
            output_dim=512,
            device=None
        ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmbeddingFusion(img_dim, text_dim, output_dim).to(self.device)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path,  map_location=self.device))
            print(f"Loaded pre-trained model from {model_path}")

        self.triplet_loss = TripletLoss(margin=1.0)
        self.batch_hard_loss = BatchHardTripletLoss(margin=1.0)

    def train(
        self,
        train_loader,
        val_loader=None,
        epochs=10,
        lr=0.001,
        save_path="models/weights/embedding_fusion.pth"
        ):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        best_val_loss = float('inf')

        for epoch in range(epochs):
            self.model.train() 
            train_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in progress_bar:
                img_anchor, text_anchor, labels = batch 
                img_anchor = img_anchor.to(self.device)
                text_anchor = text_anchor.to(self.device)
                labels = labels.to(self.device)

                embeddings = self.model(img_anchor, text_anchor)
                loss = self.batch_hard_loss(embeddings, labels)

                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step() 

                train_loss += loss.item() 
                progress_bar.set_postfix({"loss": loss.item()})

            train_loss /= len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

            if val_loader:
                val_loss = self._validate(val_loader)
                print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")
                
                scheduler.step(val_loss) # 학습률 조정
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Saved best model with validation loss: {val_loss:.4f}")
            
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)

    
    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                img_anchor, txt_anchor, labels = batch
                img_anchor = img_anchor.to(self.device)
                txt_anchor = txt_anchor.to(self.device)
                labels = labels.to(self.device)
                
                embeddings = self.model(img_anchor, txt_anchor)
                loss = self.batch_hard_loss(embeddings, labels)
                
                val_loss += loss.item()
        
        return val_loss / len(val_loader)