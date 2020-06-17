### Neural Network Representation

when using linear regression or logistic regression, we can use the basis function to apply nonlinearities. However, with high dimension data (lots of features), the computations involved can be very expensive. 

The Difficulties of computer vision:
- images are represented by pixel matrices. 
- The data has very high dimension and can be extremely difficult to compute if we use a simple linear model with basis function (especially in the case of least square, in which case we need to compute Eucleadean distance in a high dimensional space) -- curse of dimensionality.
- If greyscale image ($50 \times 50$ pixel) consists of the training data and we are using a logistic regression model with quadratic terms $(x_i, x_j)$, then each input's dimension becomes $C(2500,2) = 3125000$, which is close to 3 million features.

Each neuron in the neural network is a logistic unit. The $\sigma$ here can be the sigmoid function or other nonlinear functions. In general it is called __activation function__ and must be nonlinear. 

The first layer is the __input layer__ and the final layer the __output layer__; anything in between are __hidden layers__. 

__Notation__:
- $a_i^{(j)}$ as the "activation" of unit $i$ in layer $j$.
- $\Theta^{(j)}$ as the matrix of weights controlling function mapping from layer $j$ to layer $(j+1)$
  - If layer $j$ has $s_j$ units and layer $(j+1)$ has $s_{j+1}$ units, then dimension of $\Theta^{(j)}$ is $s_{j+1} \times (s_j + 1)$.
- In general the hyperscript represents layer number and the subscript the $i-th$ elemnt in the layer.
- Let $m$ represents the number of neurons in a layer.


From one layer to another, we have: 
$$
a_i^{(j)} = g(\Theta_{i0}^{(j-1)} x_0 + \Theta_{i1}^{(j-1)}x_1 + ... + \Theta_{im}^{(j-1)}x_m) = g((\Theta^{(j)})^T x)
$$

This transition applied everytime we have layer progression. The variable $x_0$ is usually treated as a bias term. This process is called __forward propagation__.

Vectorized representation of the forward propagation:
$$
\begin{aligned}
z_i^{(j+1)} = (\Theta^{(j)})^T x \\
a_i^{(j+1)} = g(z_i^{(j+1)}) 
\end{aligned}
$$

The output from previous layer ($a$) are inputs for the next layer. In terms of representation learning, each layer learns some different features.

When we want to do multiclass classification, the output layer uses softmax instead of sigmoid function. The result labels can be one-hot encoded vectors; for $n$-class classification problem, the output would be a n-dimensional vector.