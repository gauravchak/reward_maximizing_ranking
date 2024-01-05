# reward_maximizing_ranking
Adding REINFORCE based reward maximization to pointwise ranking

As elaborated in [Does your model get better at task T when you rank by estimated probability p(T) ?](https://recsysml.substack.com/p/does-your-model-get-better-at-task) the gradient update from logistic / cross-entropy loss of predicting a task is different than the gradient update of maximizing the reward of the task.

In this experiment, we seek to compare two approaches to multi-task scoring
1. rank with scores that are trained to maximize accuracy of prediction of observed labels
2. rank with scores that when combined according to prespecified weights leads to maximal reward

## Control = Estimator based on accuracy of prediction
Let $f_{\theta}(x)$ represent the predictions (note predictions not logits) of your model, and $y$ be the true labels. If $f_{\theta}(x)$ and $y$ are T-dimensional vectors, the binary cross-entropy loss for $t_{th}$ task can be computed as

$$

L(\theta)_{t} = -\frac{1}{N} \sum_{i=1}^{N} [ y_{i,t} \log(f_{\theta}(x_i)_{t}) + (1 - y_{i,t}) \log(1 - f_{\theta}(x_i)_{t}) ]

$$

And summing the loss over all tasks:

$$L(\theta) = \sum_{t=1}^{T} L(\theta)_{t}$$

## Test = Reward maximization in addition to prediction accuracy
In test, we will also compute a reward as a function of our model and hence seek to maximize it. UVW are T weights that have been separately found to be optimal linear combination of observed labels.

$$
\text{Expected Reward}(\theta) = \sum_{i=1}^{N} \frac{\exp\left(\frac{\langle f_\theta(x_i), UVW \rangle}{\tau}\right) \cdot \langle y_i, UVW \rangle}{\sum_{j=1}^{N} \exp\left(\frac{\langle f_\theta(x_j), UVW \rangle}{\tau}\right)}
$$

Here we have used a Bradley-Terry / Gumbel approach to computing the probability of an item being shown to the user.

$$
P(\text{item at top}) = \frac{\exp\left(\frac{\langle f_\theta(x), UVW \rangle}{\tau}\right)}{\sum_{i=1}^{N} \exp\left(\frac{\langle f_\theta(x_i), UVW \rangle}{\tau}\right)}
$$

The problem is that while training we only have access to what was logged and people typically log the impressed items and not the items that were considered. 

### Options
1. Learn a float parameter z such that 
$$
P(\text{item at top}) = \frac{\exp\left(\frac{\langle f_\theta(x), UVW \rangle}{\tau}\right)}{z}
$$

2. Learn $\tau$ and $z$ as a function of $x_u$.
$$
P(\text{item at top}) = \frac{\exp\left(\frac{\langle f_\theta(x), UVW \rangle}{g_\theta(x_u)}\right)}{h_\theta(x_u)}
$$

h and g are in a way learning the scale of the scores of the behavior policy.
