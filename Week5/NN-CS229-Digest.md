### Stochastic Gradient Descent (SGD)

- Hyperparameters: learning rate $\alpha$. number of total iterations $n_{iter}$
- Initialized $\theta$ randomly
- for $i = 1$ to $n_{iter}$ do:
  - Sample $j$ uniformly from $\{1, ..., n \}$ and update $\theta$ by:

$$
\theta \coloneqq \theta - \alpha \nabla_\theta J^{(j)}(\theta)
$$

Comparing with the Batch gradient descent:

$$
\theta \coloneqq \theta - \alpha \nabla_\theta J(\theta)
$$

we are only updating based on a sampled set. For faster optimization, we use hardware parallelization and uses a mini-batch for sampling:

- In addition to above algorithm, we specify batch size $B$.
- In the loop, we do:
  - Sample $B$ examples $j_1, ..., j_B$ without replacement uniformly from ${1, ..., n}$ and update $\theta$ by:

$$
\theta \coloneqq \theta - \frac{\alpha}{B} \sum_{k=1}^B \nabla_\theta J^{(j_k)}(\theta)
$$

### Vectorized Representation of NN

For a two-layer fully-connected NN with $m$ hidden units and $d$ dimensional input $x \in \mathbb{R}^d$, it is defined mathematically as:

$$
\forall j \in [1, ..., m], \qquad z_j = (w_j^{[1]})^T x + b_j^{[1]} \qquad \text{where} w_j^{[1]} \in \mathbb{R}^d, b_j^{[1]} \in \mathbb{R}
$$

this means we assume that for each layer, the dimension of weight/parameter vector $w$ is $d$.

We'd like to write the notation in vector form, becausevectorized representation makes use of GPUs and ensures better performance.

Define the weight matrix of an entire layer as:

$$
W^{[i]} = \begin{bmatrix}
{w_i^{[1]}}^T \\
{w_i^{[2]}}^T \\
... \\
{w_i^{[m]}}^T
\end{bmatrix} \in \mathbb{R}^{m \times d}
$$

we can write $z^{[i]}$ as:

$$
z_i = W^{[i]} x^{[i]} + b^{[i]}
$$

where the dimension of $x^{[i]} = [x_1^{[1]}, ..., x_d^{[i]}]^T$ has dimension $d \times 1$ and both $z$ and $b$ has dimension $m \times 1$. The activation function doesn't change the dimension so we have

$$
a_i = f(z_i) \in \mathbb{R}^{m \times 1}
$$

The neural network is then defined by the following recursion:

$$
a^{[k]} = f(W^{[k]}a^{[k-1]} + b^{[k]}), \forall k = 1,..., r-1
$$

### Neaty-Gritty of BackProp

Backprop is basically a mathematical trick to compute $\nabla J^{(i)}(\theta)$ efficiently.

#### Chain Rule

If

$$
\begin{aligned}
g_j = g_j(\theta_1, ..., \theta_p), \forall j \in \{ 1, ..., k \} \\
J = J(g_1, ..., g_k)
\end{aligned}
$$

then

$$
\frac{\partial J}{\partial \theta_i} = \sum_{j=1}^k \frac{\partial J}{\partial g_j} \frac{\partial g_j}{\partial \theta_i}
$$

#### Multi-Layer NN

define

$$
\delta^{[k]} = \frac{\partial J}{\partial z^{[k]}}
$$

where $z$ is the result of the linear combination, i.e. $a^{[r]} = z^{[r]} = W^{[r]}a^{[r-1]} + b^{[r]}$.

The algorithm:

- Compute and store the values of $a^{[k]}$ and $z^{[k]}$ with forward propagation, for $k = 1, ..., r-1$
- Compute $\delta^{[r]} = \frac{\partial J}{\partial z^{[r]}}$. i.e. compute the loss at output
- For $k = r - 1$ to $1$ do:
  - Compute

$$
\delta^{[k]} = \frac{\partial J}{\partial z^{[k]}} = \Big( {W^{[k+1]}}^T \delta^{[k+1]}\Big) f'(z^{[k]})
$$

- Compute

$$
\begin{aligned}
$$
