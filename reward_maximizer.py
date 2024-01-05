import torch

from multi_task_estimator import MultiTaskEstimator


class RewardMaximizer(MultiTaskEstimator):
    """inference is same. Training uses REINFORCE"""

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
        weighted_label = labels.float() @ self.user_value_weights  # [B]

        # The probability of the item being shown is assumed to be exp(score)
        # The observed reward = weighted_label
        reward = torch.exp(task_weighted_scores) * weighted_label  # [B]

        return -reward


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
