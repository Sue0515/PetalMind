import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, distance_metric='euclidean', normalize_embeddings=True, weight_fn=None):
        super(TripletLoss, self).__init__() 
        self.margin = margin 
        self.distance_metric = distance_metric 
        self.normalize_embeddings = normalize_embeddings
        self.weight_fn = weight_fn 

    def compute_distance(self, x1, x2):
        if self.distance_metric == "cosine":
            cos_sim = F.cosine_similarity(x1, x2, dim=1, eps=1e-8)
            return 1.0 - cos_sim 
        
        else: # euclidean
            return torch.norm(x1 - x2, p=2, dim=1)

    def forward(self, anchor, positive, negative, weights=None):
        if self.normalize_embeddings:
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)

        pos_dist = self.compute_distance(anchor, positive) 
        neg_dist = self.compute_distance(anchor, negative)

        losses = F.relu(pos_dist - neg_dist + self.margin)

        if weights is not None:
            losses = losses * weights 
        elif self.weight_fn is not None:
            weights = self.weight_fn(pos_dist, neg_dist)
            losses = losses * weights 

        non_zero = (losses > 0).float().sum() + 1e-8
        return losses.sum() / non_zero, {
            'pos_dist_mean': pos_dist.mean().item(),
            'neg_dist_mean': neg_dist.mean().item(),
            'active_triplets': (losses > 0).float().sum().item() 
        }

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=1.0, distance_metric='euclidean', 
                 mining_strategy='hard', normalize_embeddings=True,
                 epsilon=1e-16, semi_hard_margin=0.1):
        """
        Args:
            margin: 트리플렛 마진
            distance_metric: 'euclidean' 또는 'cosine'
            mining_strategy: 'hard', 'semi_hard', 'random'
            normalize_embeddings: 임베딩을 L2 정규화할지 여부
            epsilon: 수치 안정성을 위한 값
            semi_hard_margin: semi-hard 네거티브 필터링을 위한 마진
        """
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.mining_strategy = mining_strategy 
        self.normalize_embeddings = normalize_embeddings  
        self.epsilon = epsilon 
        self.semi_hard_margin = semi_hard_margin 
    
    def compute_pairwise_distances(self, embeddings):
        if self.distance_metric == "cosine":
            embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
            sim_matrix = torch.matmul(embeddings_normalized, embeddings_normalized.t())
            return 1.0 - sim_matrix 
        else:
            return torch.cdist(embeddings, embeddings, p=2)

    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=embeddings.device), {
                'error': 'batch_size must be greater than 1'
            }
        
        unique_labels = torch.unique(labels)
        if len(unique_labels) <= 1: 
            return torch.tensor(0.0, device=embeddings.device), {
                'error': 'batch must contain at least 2 different labels'
            }
        
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        

        pairwise_dist = self.compute_pairwise_distances(embeddings) # 거리 행렬 계산 
        label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)) # 같은 label 을 가진 샘플들의 마스크 

        # 양성 마스크 
        i_not_equal_j = ~torch.eye(batch_size, device=embeddings.device).bool() 
        positive_mask = label_equal & i_not_equal_j
        negative_mask = ~label_equal # 음성마스크 

        if positive_mask.sum() == 0: 
            return torch.tensor(0.0, device=embeddings.device), {
                'error': 'no positive pairs found in batch'
            }

        # 같은 클래스 중 가장 먼 것 
        hardest_positive_dist, _ = torch.max(
            pairwise_dist * positive_mask.float() + (1.0 - positive_mask.float()) * self.epsilon, dim=1 
        ) 

        if negative_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device), {
                'error': 'no negative pairs found in batch'
            }
        
        if self.mining_strategy == 'hard':
            # 다른 클래스 중 가장 가까운 것
            hardest_negative_dist, _ = torch.min(
                pairwise_dist * negative_mask.float() + 
                (1.0 - negative_mask.float()) * (torch.max(pairwise_dist) + 1.0), 
                dim=1
            )
            
            relevant_negative_dist = hardest_negative_dist
            
        elif self.mining_strategy == 'semi_hard':
            semi_hard_mask = negative_mask & (pairwise_dist > hardest_positive_dist.unsqueeze(1)) & \
                             (pairwise_dist < hardest_positive_dist.unsqueeze(1) + self.margin)
            
            if semi_hard_mask.sum() > 0:
                # semi-hard 네거티브 중 가장 가까운 것
                semi_hard_dist = pairwise_dist * semi_hard_mask.float() + \
                                 (1.0 - semi_hard_mask.float()) * (torch.max(pairwise_dist) + 1.0)
                relevant_negative_dist, _ = torch.min(semi_hard_dist, dim=1)
            else:
                # semi-hard가 없으면 일반 hardest 사용
                hardest_negative_dist, _ = torch.min(
                    pairwise_dist * negative_mask.float() + 
                    (1.0 - negative_mask.float()) * (torch.max(pairwise_dist) + 1.0), 
                    dim=1
                )
                relevant_negative_dist = hardest_negative_dist
            
        else:  # random
            random_negative_dist = []
            for i in range(batch_size):
                negative_indices = torch.where(negative_mask[i])[0]
                if len(negative_indices) > 0:
                    random_idx = negative_indices[torch.randint(0, len(negative_indices), (1,))]
                    random_negative_dist.append(pairwise_dist[i, random_idx])
                else:

                    random_negative_dist.append(torch.tensor(0.0, device=embeddings.device))
            
            relevant_negative_dist = torch.stack(random_negative_dist)
        
        triplet_loss = F.relu(hardest_positive_dist - relevant_negative_dist + self.margin)
        active_triplets = (triplet_loss > 0).float().sum()
        
        loss = triplet_loss.mean() if active_triplets > 0 else torch.tensor(0.0, device=embeddings.device)
        
        metrics = {
            'pos_dist_mean': hardest_positive_dist.mean().item(),
            'neg_dist_mean': relevant_negative_dist.mean().item(),
            'active_triplets': active_triplets.item(),
            'active_ratio': (active_triplets / batch_size).item()
        }
        
        return loss, metrics