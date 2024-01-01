# reward_maximizing_ranking
Adding REINFORCE based reward maximization to pointwise ranking

As elaborated in [Does your model get better at task T when you rank by estimated probability p(T) ?](https://recsysml.substack.com/p/does-your-model-get-better-at-task) the gradient update from logistic / cross-entropy loss of predicting a task is different than the gradient update of maximizing the reward of the task.

In this experiment, we seek to train a ranking estimator model to maximize a reward in addition to getting better at prediction accuracy of the binary tasks.

## Control = Estimator based on accuracy of prediction
Let $f_{\theta}(x)$ represent the predictions (note predictions not logits) of your model, and $y$ be the true labels. If $f_{\theta}(x)$ and $y$ are T-dimensional vectors, the binary cross-entropy loss for each task can be computed as

$$
L(\theta)_{t} = -\frac{1}{N} 
$$

$$
L(\theta)_{t} = -\frac{1}{N} \sum_{i=1}^{N} [ y_{i,t} \log(f_{\theta}(x_i)_{t}) + (1 - y_{i,t}) \log(1 - f_{\theta}(x_i)_{t}) ]
$$

And summing over all tasks:

$$L(\theta) = \sum_{t=1}^{T} L(\theta)_{t}$$

## Test = Reward maximization in addition to prediction accuracy
In test, we will also compute a reward as a function of our model and hence seek to maximize it.

$$
\text{Expected Reward}(\theta) = \sum_{i=1}^{N} \frac{\exp\left(\frac{\langle f_\theta(x_i), UVW \rangle}{\tau}\right) \cdot \langle y_i, UVW \rangle}{\sum_{j=1}^{N} \exp\left(\frac{\langle f_\theta(x_j), UVW \rangle}{\tau}\right)}
$$

Here we have used a Bradley-Terry approach:

$$
P(\text{item at top}) = \frac{\exp\left(\frac{\langle f_\theta(x), UVW \rangle}{\tau}\right)}{\sum_{i=1}^{N} \exp\left(\frac{\langle f_\theta(x_i), UVW \rangle}{\tau}\right)}
$$

