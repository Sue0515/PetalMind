import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingFusion(nn.Module):

    def __init__(self, img_dim: int = 512, text_dim: int = 512, output_dim: int = 512, 
                 hidden_dim: int = 1024, dropout_rate: float = 0.1):
        
        super(EmbeddingFusion, self).__init__()
        
        self.img_projection = nn.Linear(img_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.output_norm = nn.LayerNorm(output_dim)
        self._init_weights()
    

    def _init_weights(self):
        # initializiing weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    

    def forward(self, img_embedding, text_embedding):
        # image embedding projection 
        img_proj = self.img_projection(img_embedding)
        img_proj = F.relu(img_proj)
        
        # text embedding projection
        text_proj = self.text_projection(text_embedding)
        text_proj = F.relu(text_proj)
        
        combined = torch.cat([img_proj, text_proj], dim=1)
        fused = self.fusion_mlp(combined)
        fused = self.output_norm(fused)
        
        # L2 Normalization
        fused = F.normalize(fused, p=2, dim=1)
        
        return fused