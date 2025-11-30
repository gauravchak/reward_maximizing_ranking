"""
This is a specific instance of a final ranker in a recommender system.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MultiTaskEstimator(nn.Module):
    def __init__(
        self,
        num_tasks: int,
        user_id_hash_size: int,
        user_id_embedding_dim: int,
        user_features_size: int,
        item_id_hash_size: int,
        item_id_embedding_dim: int,
        user_value_weights: List[float],
    ) -> None:
        """
        params:
            num_tasks (T): The tasks to compute estimates of
            user_id_hash_size: the size of the embedding table for users
            user_id_embedding_dim (DU): internal dimension
            user_features_size (IU): input feature size for users
            item_id_hash_size: the size of the embedding table for items
            item_id_embedding_dim (DI): internal dimension
            user_value_weights: T dimensional weights, such that a linear
            combination of point-wise immediate rewards is the best predictor
            of long term user satisfaction.
        """
        super(MultiTaskEstimator, self).__init__()
        self.register_buffer(
            "user_value_weights",
            torch.tensor(user_value_weights, dtype=torch.float32),
        )

        # Embedding layers for user and item ids
        self.user_embedding = nn.Embedding(
            user_id_hash_size, user_id_embedding_dim
        )
        self.item_embedding = nn.Embedding(
            item_id_hash_size, item_id_embedding_dim
        )

        # Linear projection layer for user features
        self.user_features_layer = nn.Linear(
            in_features=user_features_size, out_features=user_id_embedding_dim
        )  # noqa

        # Linear layer for final prediction
        self.task_arch = nn.Linear(
            2 * user_id_embedding_dim + item_id_embedding_dim, num_tasks
        )  # noqa

    def forward(
        self, user_id, user_features, item_id  # [B]  # [B, IU]  # [B]
    ) -> torch.Tensor:
        # Embedding lookup for user and item ids
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)

        # Linear transformation for user features
        user_features_transformed = self.user_features_layer(user_features)

        # Concatenate user embedding, user features, and item embedding
        combined_features = torch.cat(
            [user_embedding, user_features_transformed, item_embedding], dim=1
        )

        # Compute per-task scores/logits
        task_logits = self.task_arch(combined_features)  # [B, T]

        return task_logits

    def train_forward(self, user_id, user_features, item_id, labels) -> float:
        """Compute the loss during training"""
        # Get task logits using forward method
        task_logits = self.forward(user_id, user_features, item_id)

        # Compute binary cross-entropy loss
        cross_entropy_loss = F.binary_cross_entropy_with_logits(
            input=task_logits, target=labels.float(), reduction="sum"
        )

        return cross_entropy_loss
