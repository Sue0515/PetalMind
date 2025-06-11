import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        # triplet loss 
        losses = F.relu(pos_dist - neg_dist + self.margin)

        return torch.mean(losses)


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=1.0, distance_metric='euclidean'):
    
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
    
    def compute_distance_matrix(self, embeddings):
        if self.distance_metric == 'cosine':
            embeddings = F.normalize(embeddings, p=2, dim=1)
            sim_matrix = torch.matmul(embeddings, embeddings.t()) # cosine similarity 
            
            return 1 - sim_matrix
        
        else:  # 'euclidean'
            return torch.cdist(embeddings, embeddings, p=2)
    
    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        distance_matrix = self.compute_distance_matrix(embeddings)
        
        # 같은 레이블을 가진 샘플들의 마스크
        same_label_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))   
        # 자기 자신 마스크
        identity_mask = torch.eye(batch_size, device=embeddings.device).bool()
        # 양성 쌍 마스크 (같은 레이블, 자기 자신 제외)
        positive_mask = same_label_mask & (~identity_mask)
        # 음성 쌍 마스크 (다른 레이블)
        negative_mask = ~same_label_mask
        
        max_dist = torch.max(distance_matrix).item()
        
        # 같은 클래스 중 가장 먼 것
        positive_dist = distance_matrix * positive_mask.float() + (1.0 - positive_mask.float()) * 1e-9
        hardest_positive_dist, _ = torch.max(positive_dist, dim=1)
        
        # 다른 클래스 중 가장 가까운 것
        negative_dist = distance_matrix * negative_mask.float() + (1.0 - negative_mask.float()) * max_dist
        hardest_negative_dist, _ = torch.min(negative_dist, dim=1)
        
        # triplet loss 
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        active_triplets = (triplet_loss > 0).float().sum()
        
        if active_triplets > 0:
            return triplet_loss.sum() / active_triplets
        else:
            # return torch.tensor(0.0, device=embeddings.device)
            return triplet_loss.sum() * 0.0