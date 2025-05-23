import os
import torch
import numpy as np
from PIL import Image
import clip
from typing import Tuple, List, Dict, Any

class CLIPExtractor:
    def __init__(self, model_name="ViT-B/32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval() 

    def extract_image_embedding(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_embedding = self.model.encode_image(processed_image)
            
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        return image_embedding.cpu().numpy()[0]
    
    def extract_text_embedding(self, text: str) -> np.ndarray:
        text_tokens = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
            
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        return text_embedding.cpu().numpy()[0]
    
    def extract_embeddings(self, image_path: str, text: str) -> Tuple[np.ndarray, np.ndarray]:
        img_emb = self.extract_image_embedding(image_path)
        txt_emb = self.extract_text_embedding(text)
        
        return img_emb, txt_emb
    
    def batch_extract_embeddings(self, 
                                image_paths: List[str], 
                                texts: List[str]) -> Dict[str, List[np.ndarray]]:
        
        batch_size = len(image_paths)
        assert batch_size == len(texts), "이미지와 텍스트 수가 일치해야 합니다"
        
        img_embeddings = []
        txt_embeddings = []
        
        for i in range(batch_size):
            img_emb, txt_emb = self.extract_embeddings(image_paths[i], texts[i])
            img_embeddings.append(img_emb)
            txt_embeddings.append(txt_emb)
        
        return {
            'image_embeddings': img_embeddings,
            'text_embeddings': txt_embeddings
        } 