import torch

from src.multi_task_estimator import MultiTaskEstimator


def test_multi_task_estimator():
    # Replace the placeholder values with your actual data dimensions
    num_tasks = 3
    user_id_hash_size = 100
    user_id_embedding_dim = 50
    user_features_size = 10
    item_id_hash_size = 200
    item_id_embedding_dim = 30

    user_value_weights = [0.5, 0.3, 0.2]

    # Instantiate the MultiTaskEstimator
    model = MultiTaskEstimator(
        num_tasks,
        user_id_hash_size,
        user_id_embedding_dim,
        user_features_size,
        item_id_hash_size,
        item_id_embedding_dim,
        user_value_weights,
    )

    # Example input data
    B = 4
    user_id = torch.randint(0, user_id_hash_size, (B,))
    user_features = torch.randn(B, user_features_size)
    item_id = torch.randint(0, item_id_hash_size, (B,))
    labels = torch.randint(0, 2, (B, num_tasks)).float()

    # Example forward pass
    output = model(user_id, user_features, item_id)
    print("Forward Pass Output:", output)
    assert output.shape == (B, num_tasks)

    # Example train_forward pass
    # model_scores is unused in the base estimator but required for signature consistency
    model_scores = torch.rand(B, num_tasks)
    loss = model.train_forward(
        user_id, user_features, item_id, labels, model_scores
    )
    print("Training Loss:", loss.item())
    assert loss.item() > 0
