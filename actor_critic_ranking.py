from typing import List
import torch
import torch.nn as nn

from multi_task_estimator import MultiTaskEstimator


class ActorCriticRanking(MultiTaskEstimator):
    """Inference is same as superclass. Training uses Actor-Critic"""

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
        super(ActorCriticRanking, self).__init__(
            num_tasks=num_tasks,
            user_id_hash_size=user_id_hash_size,
            user_id_embedding_dim=user_id_embedding_dim,
            user_features_size=user_features_size,
            item_id_hash_size=item_id_hash_size,
            item_id_embedding_dim=item_id_embedding_dim,
            user_value_weights=user_value_weights,
        )
        # Set up critic
        self.critic = nn.Linear(2 * user_id_embedding_dim, num_tasks)  # noqa

    def evaluate_critic(
        self,
        user_id,
        user_features,    
    ) -> torch.Tensor:
        """Computes the T dimensional task scores from critic"""
        # Embedding lookup for user is
        user_embedding = self.user_embedding(user_id)

        # Linear transformation for user features
        user_features_transformed = self.user_features_layer(user_features)

        # Concatenate user embedding, user features, and item embedding
        combined_features = torch.cat(
            [
                user_embedding,
                user_features_transformed, 
            ],
            dim=1
        )

        # Compute per-task scores/logits
        critic_task_scores = self.critic(combined_features)  # [B, T]
        return critic_task_scores

    def train_forward(
        self,
        user_id,
        user_features,
        item_id,
        labels
    ) -> float:
        """Compute the loss during training"""
        # Get task logits using forward method
        task_scores = self.forward(user_id, user_features, item_id)  # [B, T]
        task_weighted_scores = task_scores @ self.user_value_weights  # [B]
        observed_reward = labels.float() @ self.user_value_weights  # [B]

        critic_task_scores = self.evaluate_critic(
            user_id, user_features)  # [B, T]
        # The expected reward by critic
        critic_reward = critic_task_scores @ self.user_value_weights  # [B]
        # The probability of the item being shown is assumed to be exp(score)
        incremental_reward = (
            torch.exp(task_weighted_scores) *
            (observed_reward - critic_reward)
        )  # [B]

        criterion = nn.MSELoss()
        critic_mse_loss = criterion(critic_reward, observed_reward)

        loss = critic_mse_loss - incremental_reward

        return loss


# Example usage:
# Replace the placeholder values with your actual data dimensions
num_tasks = 3
user_id_hash_size = 100
user_id_embedding_dim = 50
user_features_size = 10
item_id_hash_size = 200
item_id_embedding_dim = 30

# unused in the above implementation
user_value_weights = [0.5, 0.3, 0.2]


# Instantiate the MultiTaskEstimator
model = RewardMaximizer(
    num_tasks, user_id_hash_size, user_id_embedding_dim,
    user_features_size, item_id_hash_size, item_id_embedding_dim,
    user_value_weights
)

# Example input data
user_id = torch.tensor([1, 2, 3])  # Replace with your actual user_id data
user_features = torch.randn(3, user_features_size)  # Replace with your actual user_features data
item_id = torch.tensor([4, 5, 6])  # Replace with your actual item_id data
labels = torch.tensor([0, 1, 2])  # Replace with your actual labels data

# Example forward pass
output = model(user_id, user_features, item_id)
print("Forward Pass Output:", output)

# Example train_forward pass
loss = model.train_forward(user_id, user_features, item_id, labels)
print("Training Loss:", loss.item())
