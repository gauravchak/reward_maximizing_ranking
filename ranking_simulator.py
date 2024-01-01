"""
Ranking Simulator reads the training data collected by the behavior policy
It calls the MultiTaskEstimator to compute estimates.
Based on the estimate it computes the probability that the given item 
would have been selected at top position. The net reward is computed as
Sum over data points (prob(top) * reward) / Sum over data points (prob(top))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from multi_task_estimator import MultiTaskEstimator


class RankingSimulator(nn.Module):
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
        super(RankingSimulator, self).__init__()

        # Instantiate the MultiTaskEstimator
        self.estimator = MultiTaskEstimator(
            num_tasks, user_id_hash_size, user_id_embedding_dim,
            user_features_size, item_id_hash_size, item_id_embedding_dim,
            user_value_weights
        )

        # User value weights
        self.user_value_weights = nn.Parameter(torch.Tensor(user_value_weights), requires_grad=False)

    def forward(
        self,
        user_id,  # [B]
        user_features,  # [B, UFS]
        item_id  # [B]
    ) -> torch.Tensor:
        """
        Returns expected reward in the batch
        """
        # Call forward of the estimator to get task_estimates
        task_estimates = self.estimator.forward(
            user_id, user_features, item_id
        )  # [B, T]

        # Combine task estimates with user value weights
        value_estimates = torch.matmul(
            task_estimates, self.user_value_weights
        )  # [B]

        # Compute item probability based on value estimate
        item_probs = F.softmax(value_estimates)  # [B]

        return torch.sum(item_probs * value_estimates) / torch.sum(item_probs)

# Example usage:
# Replace the placeholder values with your actual data dimensions
num_tasks = 3
user_id_hash_size = 100
user_id_embedding_dim = 50
user_features_size = 10
item_id_hash_size = 200
item_id_embedding_dim = 30

# This is based on a separate assessment of what linear combination of 
# point-wise immediate rewards is the best predictor of long term user
# satisfaction.
user_value_weights = [0.5, 0.3, 0.2]

ranking_simulator = RankingSimulator(
    num_tasks, user_id_hash_size, user_id_embedding_dim,
    user_features_size, item_id_hash_size, item_id_embedding_dim,
    user_value_weights
)

# Example input data
B = 5  # Batch size
user_id = torch.randint(0, user_id_hash_size, (B))  # [B]
user_features = torch.randn(B, user_features_size)  # [B, UFS]
item_id = torch.randint(0, item_id_hash_size, (B))  # [B]

# Example forward pass
max_item = ranking_simulator(user_id, user_features, item_id)
print("Max Item:", max_item)
