# Reward Maximizing Ranking for Recommender Systems

A PyTorch implementation for training a recommender system ranker to directly optimize for a weighted combination of user rewards, using off-policy policy gradient methods.

This repository explores moving beyond standard engagement prediction (e.g., clicks) to directly optimize for long-term user value.

For a detailed explanation, please read the accompanying blog post: _[Link to blog post to be added]_

## Core Idea

Instead of just predicting the probability of various user interactions (like click, like, share), this model learns a ranking policy `π_θ` that maximizes an expected long-term reward. The reward is defined as a weighted sum of the different interaction probabilities.

The training uses an off-policy approach, learning from data logged by a production policy `π_β`. The loss function combines a standard multi-task prediction loss with a policy gradient loss, adjusted by importance sampling.

`Loss = L_BCE + λ * L_PolicyGradient`

## Models

-   `src/multi_task_estimator.py`: A baseline model that predicts probabilities for multiple tasks using a standard Binary Cross-Entropy loss.
-   `src/reward_maximizer.py`: An advanced model that uses a combined loss function to directly optimize for the weighted reward, while also being trained on the base prediction tasks. It returns an Off-Policy Estimate (OPE) of the reward for evaluation.

## Project Structure

```
├── src
│   ├── __init__.py
│   ├── multi_task_estimator.py
│   └── reward_maximizer.py
├── tests
│   ├── conftest.py
│   ├── test_multi_task_estimator.py
│   └── test_reward_maximizer.py
└── README.md
```

## Getting Started

1.  **Install dependencies:**
    ```bash
    pip install torch pytest
    ```

2.  **Run tests:**
    ```bash
    pytest
    ```