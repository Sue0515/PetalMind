import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CrossAttentionLayer(nn.Module):
    
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
    
        super(CrossAttentionLayer, self).__init__()

        self.head_dim = dim // heads
        self.heads = heads
        self.scale = self.head_dim ** -0.5

        self.to_queries = nn.Linear(dim, dim)
        self.to_keys = nn.Linear(dim, dim)
        self.to_values = nn.Linear(dim, dim)

        self.to_out = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
       
        batch_size, seq_len_q, _ = x.shape
        _, seq_len_kv, _ = context.shape
        
        queries = self.to_queries(x)
        keys = self.to_keys(context)
        values = self.to_values(context)

        queries = queries.view(batch_size, seq_len_q, self.heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len_kv, self.heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len_kv, self.heads, self.head_dim).transpose(1, 2)
        
        dots = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale

        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, values)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
 
        return self.to_out(out)


class FeedForward(nn.Module):
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super(CrossAttentionBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.cross_attn = CrossAttentionLayer(dim, heads, dropout)
        self.ff = FeedForward(dim, dim * 4, dropout)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.cross_attn(self.norm1(x), self.norm2(context))
        x = x + self.ff(self.norm3(x))
        
        return x


class CrossAttentionRecommender(nn.Module):
    
    def __init__(self, dim: int = 512, heads: int = 8, dropout: float = 0.1, layers: int = 2):
        super(CrossAttentionRecommender, self).__init__()
        
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(dim, heads, dropout)
            for _ in range(layers)
        ])
        
        self.score_predictor = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, main_embedding: torch.Tensor, other_embedding: torch.Tensor) -> torch.Tensor:
        """
        꽃 조합 조화 점수 예측
        
        Args:
            main_embedding: 메인 꽃 임베딩
            other_embedding: 다른 꽃 임베딩
            
        Returns:
            조화 점수 (bs, 1)
        """

        batch_size = main_embedding.size(0)

        main_emb = main_embedding.unsqueeze(1)  # (batch, 1, dim)
        other_emb = other_embedding.unsqueeze(1)  # (batch, 1, dim)
        
        for block in self.cross_attn_blocks:
            # 메인 -> 다른 꽃 
            other_emb = block(other_emb, main_emb)
            # 다른 꽃 -> 메인 
            main_emb = block(main_emb, other_emb)
        
        main_emb = main_emb.squeeze(1)  # (batch, dim)
        other_emb = other_emb.squeeze(1)  # (batch, dim)

        combined = torch.cat([main_emb, other_emb], dim=1)  # (batch, dim*2)

        harmony_score = self.score_predictor(combined)
        
        return harmony_score.squeeze(-1)  # (batch)