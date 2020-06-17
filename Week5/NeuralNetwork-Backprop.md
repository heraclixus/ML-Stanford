### NN Cost Function

- $L$ as the total number of layers in the network
- $s_l$ as the number of units (not counting bias term) in layer $l$.
- K as the number of output units/classes
- $h_\Theta(x)_k$ as the hypothesis that results in the $k$th output.
- $m$ as the number of nodes in a layer

The cost function for neural network is then:

$$
J(\theta) = -\frac{1}{m}\sum_{i=1}^m \sum_{k=1}^K \Big[ y_k^{(i)} \log((h_\Theta(x^{(i})_k)) + (1-y_k^{(i)})\log(1-(h_\Theta (x^{(i)})_k))    \Big] + \frac{\lambda}{2m} \sum_{l=1}^{L-1} \sum_{i=1}^{s_l}\sum_{j=1}^{s_l + 1} (\Theta_{j,i}^{l})^2
$$

- double sum adds up logistic regression cost for each cell in the output layer
- triple sum adds up the squares of all individual $\Theta$ in the entire network
- $i$ in the triple sum refers to the number of nodes in the layer $l$.

### Back Propagation

With gradient descent, we need to compute

$$
\frac{\partial}{\partial \Theta^{l}_{i,j}} J(\Theta) \qquad \forall i,j,l
$$

We use back propagation algorithm for it: Given training set $\{ (x^{(1)}, y^{(i)}), ..., (x^{(m)}, y^{(m)})\}$. Let $\Delta_{i,j}^{l} \coloneqq 0$, $\forall l,i,j$, i.e, initialize the paramters with all zeros.

For each training sample $t= 1$ to $m$:

1. Set $a^{(1)} \coloneqq x^{(t)}$
2. Perform forward propagation to compute $a^{(l)}$ for $l=1,2,...,L$.
3. Using $y^{(t)}$, compute $\delta^{(L)} = a^{(L)} - y^{(t)}$, where $a^{(L)}$ is the vector of outputs of the activation units for the last yaer. The error is the difference between the last layer and the correct outputs in $y$.
4. Compute $\delta^{(L-1)}, ..., \delta^{(2)}$ using

$$
\delta^{(l)} = [(\Delta^{(l)})^T \delta^{(l+1)}] \cdot * a^{(l)} \cdot * (1-a^{(l)})
$$

the detal values of layer $l$ is calcualted by multiplying delta value of next layer with the theta matrix of layer $l$. Then element-wise multiplication is performed with the derivative of the activation function (here we are using the sigmoid, hence the derivative is of the form above).

5. $\Delta_{i,j}^{(l)} \coloneqq \Delta_{i,j}^{(l)} + a_j^{(l)}\Delta_i^{(l+1)}$, or with vectorization, $\Delta^{(l)} \coloneqq \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T$.
6. Compute the accumulator matrix:

$$
\begin{aligned}
& D_{i,j}^{(l)} \coloneqq \frac{1}{m} \Big( \Delta{i,j}^{(l)} + \lambda \Theta_{i,j}^{(l)} \Big)  && \text{if  } j \neq 0 \\
& D_{i,j}^{(l)} \coloneqq \frac{1}{m} \Delta_{i,j}^{(l)}  && \text{if  } j = 0
\end{aligned}
$$

We have that $\frac{\partial}{\partial \Theta_{i,j}^{(i)}} J(\Theta) = D_{i,j}^{(l)}$.
