import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from multi_task_estimator import MultiTaskEstimator


class TopItemSelector(nn.Module):
    def __init__(
        self,
        num_tasks: int,
        user_id_hash_size: int,
        user_id_embedding_dim: int,
        user_features_size: int, 
        item_id_hash_size: int,
        item_id_embedding_dim: int,
        user_value_weights: List[float]
    ) -> None:
        super(TopItemSelector, self).__init__()

        # Instantiate the MultiTaskEstimator
        self.estimator = MultiTaskEstimator(num_tasks, user_id_hash_size, user_id_embedding_dim,
                                            user_features_size, item_id_hash_size, item_id_embedding_dim)

        # User value weights
        self.user_value_weights = nn.Parameter(torch.Tensor(user_value_weights), requires_grad=False)

    def forward(
        self,
        user_id,  # [B, K]
        user_features,  # [B, K, UFS]
        item_id  # [B, K]
    ) -> torch.Tensor :
        # Call forward of the estimator to get task_estimates
        task_estimates = self.estimator.forward(user_id, user_features, item_id)

        # Combine task estimates with user value weights
        value_estimate = torch.matmul(task_estimates, self.user_value_weights)

        # Pick max item based on value_estimate
        max_item = torch.argmax(value_estimate, dim=1)

        return max_item

# Example usage:
# Replace the placeholder values with your actual data dimensions
num_tasks = 3
user_id_hash_size = 100
user_id_embedding_dim = 50
user_features_size = 10
item_id_hash_size = 200
item_id_embedding_dim = 30
user_value_weights = [0.5, 0.3, 0.2]  # Replace with your actual user_value_weights

# Instantiate the TopItemSelector
top_selector = TopItemSelector(num_tasks, user_id_hash_size, user_id_embedding_dim,
                               user_features_size, item_id_hash_size, item_id_embedding_dim,
                               user_value_weights)

# Example input data
B = 5  # Batch size
K = 8  # Number of items per user
user_id = torch.randint(0, user_id_hash_size, (B, K))  # [B, K]
user_features = torch.randn(B, K, user_features_size)  # [B, K, UFS]
item_id = torch.randint(0, item_id_hash_size, (B, K))  # [B, K]

# Example forward pass
max_item = top_selector(user_id, user_features, item_id)
print("Max Item:", max_item)
