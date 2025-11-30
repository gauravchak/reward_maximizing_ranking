import torch
import torch.nn.functional as F

from .multi_task_estimator import MultiTaskEstimator


class RewardMaximizer(MultiTaskEstimator):
    """inference is same. Training uses REINFORCE"""

    def train_forward(
        self, user_id, user_features, item_id, labels, model_scores
    ) -> float:
        """Compute the loss during training"""
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

        # The loss is the negative of the expected reward, estimated with importance sampling.
        # We take the sum over the batch.
        loss = -torch.sum(rho * reward)

        return loss
