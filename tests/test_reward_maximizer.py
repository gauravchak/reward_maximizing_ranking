import torch

from src.reward_maximizer import RewardMaximizer


def test_reward_maximizer():
    # Replace the placeholder values with your actual data dimensions
    num_tasks = 3
    user_id_hash_size = 100
    user_id_embedding_dim = 50
    user_features_size = 10
    item_id_hash_size = 200
    item_id_embedding_dim = 30

    user_value_weights = [0.5, 0.3, 0.2]

    # Instantiate the RewardMaximizer
    model = RewardMaximizer(
        num_tasks,
        user_id_hash_size,
        user_id_embedding_dim,
        user_features_size,
        item_id_hash_size,
        item_id_embedding_dim,
        user_value_weights,
        lambda_pg=1.0,
    )

    # Example input data
    B = 4
    user_id = torch.randint(0, user_id_hash_size, (B,))
    user_features = torch.randn(B, user_features_size)
    item_id = torch.randint(0, item_id_hash_size, (B,))
    labels = torch.randint(0, 2, (B, num_tasks)).float()

    # model_scores from behavior policy (as probabilities)
    model_scores = torch.rand(B, num_tasks)

    # Example forward pass
    output = model(user_id, user_features, item_id)
    print("Forward Pass Output:", output)
    assert output.shape == (B, num_tasks)

    # Example train_forward pass
    loss, ope_reward = model.train_forward(
        user_id, user_features, item_id, labels, model_scores
    )
    print("Training Loss:", loss.item())
    print("OPE Reward:", ope_reward.item())
    # Loss can be positive or negative
    assert isinstance(loss.item(), float)
    assert isinstance(ope_reward.item(), float)
    # Check that ope_reward is detached and has no grad_fn
    assert ope_reward.grad_fn is None
