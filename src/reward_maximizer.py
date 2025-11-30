import torch
import torch.nn.functional as F
from typing import List, Tuple
from .multi_task_estimator import MultiTaskEstimator


class RewardMaximizer(MultiTaskEstimator):
    """
    A RewardMaximizer model that uses a PPO-style loss for training.

    Inference behavior is inherited from MultiTaskEstimator.
    """

    def __init__(
        self,
        num_tasks: int,
        user_id_hash_size: int,
        user_id_embedding_dim: int,
        user_features_size: int,
        item_id_hash_size: int,
        item_id_embedding_dim: int,
        user_value_weights: List[float],
        lambda_pg: float = 1.0,
    ) -> None:
        """Initializes the RewardMaximizer."""
        super().__init__(
            num_tasks,
            user_id_hash_size,
            user_id_embedding_dim,
            user_features_size,
            item_id_hash_size,
            item_id_embedding_dim,
            user_value_weights,
        )
        self.lambda_pg = lambda_pg

    def train_forward(
        self,
        user_id: torch.Tensor,
        user_features: torch.Tensor,
        item_id: torch.Tensor,
        labels: torch.Tensor,
        model_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss during training using importance sampling.

        Args:
            user_id: Tensor of user IDs.
            user_features: Tensor of user features.
            item_id: Tensor of item IDs.
            labels: Tensor of labels for each task.
            model_scores: Tensor of scores (probabilities) from the behavior policy.

        Returns:
            A tuple containing:
            - loss (torch.Tensor): The training loss.
            - ope_reward (torch.Tensor): The off-policy estimate of the reward.
        """
        # Get task logits using forward method
        task_logits = self.forward(user_id, user_features, item_id)  # [B, T]

        # Estimated probabilities from the current model (theta)
        estimated_probs = torch.sigmoid(task_logits)  # [B, T]
        s_theta = estimated_probs @ self.user_value_weights  # [B]
        pi_theta = F.softmax(s_theta, dim=0)  # [B]

        # Probabilities from the behavior policy (beta)
        # model_scores are probabilities from the behavior policy.
        s_beta = model_scores @ self.user_value_weights  # [B]
        pi_beta = F.softmax(s_beta, dim=0)  # [B]

        # Ensure pi_beta is not zero to avoid division by zero.
        pi_beta = torch.clamp(pi_beta, min=1e-8)

        # Importance sampling ratio
        rho = pi_theta / pi_beta  # [B]

        # Calculate reward
        # labels are [B, T], user_value_weights is [T]
        reward = torch.sum(
            labels.float() * self.user_value_weights, dim=1
        )  # [B]

        # Policy gradient loss component
        l_policy_gradient = -torch.sum(rho * reward)

        # BCE loss component, calculated directly to avoid a second forward pass
        l_bce = F.binary_cross_entropy_with_logits(
            input=task_logits, target=labels.float(), reduction="sum"
        )

        # Combined loss
        loss = l_bce + self.lambda_pg * l_policy_gradient

        # The off-policy estimate of the reward for the batch
        ope_reward = torch.sum(rho * reward)

        return loss, ope_reward.detach()
