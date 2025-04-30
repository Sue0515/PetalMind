import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionRecommender(nn.Module):
    def __init__(self, dim=512, heads=8, dropout=0.1):
        super().__init__() 
        self.dim = dim 
        self.heads = heads 
        self.head_dim = dim // heads 

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        self.score = nn.Sequential(
            nn.Linear(dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid() 
        ) 

    def forward(self, query, key):
        """
        Args:
            query: 쿼리 임베딩 - 메인 꽃 (B, D)
            key: 키 임베딩 - 후보 꽃 (B, D)
            
        Returns:
            scores: 추천 점수 (B)
        """
        batch_size = query.shape[0]

        q = self.q_proj(query).view(batch_size, self.heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, self.heads, self.head_dim)
        v = self.v_proj(key).view(batch_size, self.heads, self.head_dim)

        attn_scores = torch.einsum('bhd,bhd->bh', q, k) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.einsum('bh,bhd->bhd', attn_weights, v)
        attn_output = attn_output.reshape(batch_size, self.dim)
        attn_output = self.out_proj(attn_output)

        scores = self.score(attn_output).squeeze(-1)

        return scores  