import torch
import clip
import numpy as np

from PIL import Image
from typing import Tuple

class CLIPExtractor:
    def __init__(self, model_name="ViT-B/32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval() 

    def extract_embedding(self, image_path: str, text_description: str) -> Tuple[np.ndarray, np.ndarray]:
        
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise Exception(f"Error processing image at {image_path}: {str(e)}")

        text_input = clip.tokenize([text_description]).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)

            image_features = image_features / image_features.norm(dim=-1, keeptim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_embedding = image_features.cpu().numpy().squeeze() # change to numpy array 
        text_embedding = text_features.cpu().numpy().squeeze() 

        return image_embedding, text_embedding 