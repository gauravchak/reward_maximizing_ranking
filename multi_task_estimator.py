import torch
import torch.nn as nn


class MultiTaskEstimator(nn.Module):
    def __init__(self, num_tasks, user_id_hash_size, user_id_embedding_dim,
                 user_features_size, item_id_hash_size, item_id_embedding_dim):
        super(MultiTaskEstimator, self).__init__()

        # Embedding layers for user and item ids
        self.user_embedding = nn.Embedding(user_id_hash_size, user_id_embedding_dim)
        self.item_embedding = nn.Embedding(item_id_hash_size, item_id_embedding_dim)

        # Linear layer for user features
        self.user_features_layer = nn.Linear(user_features_size, user_id_embedding_dim)

        # Linear layer for final prediction
        self.final_layer = nn.Linear(user_id_embedding_dim + item_id_embedding_dim, num_tasks)

    def forward(self, user_id, user_features, item_id):
        # Embedding lookup for user and item ids
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)

        # Linear transformation for user features
        user_features_transformed = self.user_features_layer(user_features)

        # Concatenate user embedding, user features, and item embedding
        combined_features = torch.cat([user_embedding, user_features_transformed, item_embedding], dim=1)

        # Final prediction using linear layer
        output = self.final_layer(combined_features)

        return output

    def train_forward(self, user_id, user_features, item_id, labels):
        # Forward pass
        predictions = self.forward(user_id, user_features, item_id)

        # Compute cross-entropy loss with logits
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(predictions, labels)

        return loss

# Example usage:
# Replace the placeholder values with your actual data dimensions
num_tasks = 3
user_id_hash_size = 100
user_id_embedding_dim = 50
user_features_size = 10
item_id_hash_size = 200
item_id_embedding_dim = 30

# Instantiate the MultiTaskEstimator
model = MultiTaskEstimator(num_tasks, user_id_hash_size, user_id_embedding_dim,
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
