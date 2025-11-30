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
        """Initializes the MultiTaskEstimator.

        Args:
            num_tasks: The number of tasks to compute estimates for.
            user_id_hash_size: The size of the embedding table for users.
            user_id_embedding_dim: The internal dimension for user embeddings.
            user_features_size: The input feature size for users.
            item_id_hash_size: The size of the embedding table for items.
            item_id_embedding_dim: The internal dimension for item embeddings.
            user_value_weights: T-dimensional weights for combining rewards.
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
        self,
        user_id: torch.Tensor,
        user_features: torch.Tensor,
        item_id: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the forward pass to get task logits.

        Args:
            user_id: Tensor of user IDs with shape [B].
            user_features: Tensor of user features with shape [B, IU].
            item_id: Tensor of item IDs with shape [B].

        Returns:
            A tensor of task logits with shape [B, T].
        """
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

    def train_forward(
        self,
        user_id: torch.Tensor,
        user_features: torch.Tensor,
        item_id: torch.Tensor,
        labels: torch.Tensor,
        model_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss during training.

        Args:
            user_id: Tensor of user IDs.
            user_features: Tensor of user features.
            item_id: Tensor of item IDs.
            labels: Tensor of labels for each task.
            model_scores: Optional tensor of scores from a behavior policy.
                This is unused in the base class but needed for subclasses.

        Returns:
            The computed loss as a tensor.
        """
        # Get task logits using forward method
        task_logits = self.forward(user_id, user_features, item_id)

        # Compute binary cross-entropy loss
        return F.binary_cross_entropy_with_logits(
            input=task_logits, target=labels.float(), reduction="sum"
        )
