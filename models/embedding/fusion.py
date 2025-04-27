import torch
import torch.nn as nn

class EmbeddingFusion(nn.Module):
    def __init__(self, img_dim=512, text_dim=512, output_dim=512):
        super().__init__() 
        self.img_proj = nn.Linear(img_dim, 512)
        self.text_proj = nn.Linear(text_dim, 512)

        self.fusion = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, output_dim)
        )

    def forward(self, img_emb, text_emb):
        img_feat = self.img_proj(img_emb) 
        text_feat = self.text_proj(text_emb)

        concat_feat = torch.cat([img_feat, text_feat], dim=-1)
        fused_emb = self.fusion(concat_feat)

        return fused_emb
