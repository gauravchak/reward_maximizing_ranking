import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskEstimator(nn.Module):
    def __init__(
        self,
        num_tasks: int,
        user_id_hash_size: int,
        user_id_embedding_dim: int,
        user_features_size: int, 
        item_id_hash_size: int,
        item_id_embedding_dim: int,
    ) -> None:
        """
        params:
        num_tasks (T): The tasks to compute estimates of
        user_id_hash_size: the size of the embedding table for users
        user_id_embedding_dim (DU): internal dimension
        user_features_size (IU): input feature size for users
        item_id_hash_size: the size of the embedding table for items
        item_id_embedding_dim (DI): internal dimension
        """
        super(MultiTaskEstimator, self).__init__()

        # Embedding layers for user and item ids
        self.user_embedding = nn.Embedding(user_id_hash_size, user_id_embedding_dim)  # noqa
        self.item_embedding = nn.Embedding(item_id_hash_size, item_id_embedding_dim)  # noqa

        # Linear projection layer for user features
        self.user_features_layer = nn.Linear(user_features_size, user_id_embedding_dim)

        # Linear layer for final prediction
        self.task_arch = nn.Linear(2 * user_id_embedding_dim + item_id_embedding_dim, num_tasks)

    def forward(
        self, 
        user_id,  # [B]
        user_features,  # [B, IU]
        item_id  # [B]
    ) -> torch.Tensor:
        # Embedding lookup for user and item ids
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)

        # Linear transformation for user features
        user_features_transformed = self.user_features_layer(user_features)

        # Concatenate user embedding, user features, and item embedding
        combined_features = torch.cat(
            [
                user_embedding,
                user_features_transformed, 
                item_embedding
            ],
            dim=1
        )

        # Final prediction using linear layer
        task_logits = self.task_arch(combined_features)  # [B, T]

        return task_logits

    def train_forward(
        self,
        user_id,
        user_features,
        item_id,
        labels
    ) -> None:
        # Get task logits using forward method
        task_logits = self.forward(user_id, user_features, item_id)

        # Compute binary cross-entropy loss
        cross_entropy_loss = F.binary_cross_entropy_with_logits(
            input=task_logits, target=labels.float(), reduction='sum'
        )

        return cross_entropy_loss

# Example usage:
# Replace the placeholder values with your actual data dimensions
num_tasks = 3
user_id_hash_size = 100
user_id_embedding_dim = 50
user_features_size = 10
item_id_hash_size = 200
item_id_embedding_dim = 30

# Instantiate the MultiTaskEstimator
model = MultiTaskEstimator(
    num_tasks, user_id_hash_size, user_id_embedding_dim,
    user_features_size, item_id_hash_size, item_id_embedding_dim)

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
