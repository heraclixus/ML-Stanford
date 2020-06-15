### Notation
* $x = \{ x^{(1)}, ..., x^{(i)}, ..., x^{(m)} \}$ the training dataset
* $n$ as the number of features.
* $m$ as the number of samples.
* $x^{(i)}$: input features of the ith training example.
* $x^{(i)}_j$ the jth feature of the ith training example. 

__Hypothesis__:

Define $x_0 = 1$, and define a feature vector (the ith feature vector) $x^{(i)}$ as:
$$
x^{(i)} = \begin{bmatrix}
x_0^{(i)} \\
x_1^{(i)} \\
... \\
x_n^{(i)} 
\end{bmatrix}
$$

and define the parameter vector as:
$$
\theta = \begin{bmatrix}
\theta_0 \\
\theta_1 \\
... \\
\theta_n
\end{bmatrix}
$$
The hypothesis for an input feature is therefore:
$$
h_{\theta}(x^{(i)})  = \sum_{j=0}^n \theta_j x^{(i)}_j = \theta^T x^{(i)}
$$

### Gradient Descent

Suppose we have $m$ input vectors, then the cost function for the hypothesis (MLE) is: 
$$
J(\theta) = J(\theta_0, ..., \theta_n) = \frac{1}{2m} \sum_{i=1}^m (\theta^T x^{(i)} - y_i)^2
$$

Gradient Descent algorithm (batch):

Repeat:
$$
\theta_j \coloneqq \theta_j - \alpha \frac{\partial}{\partial \theta_j} J (\theta) = \alpha \frac{1}{m}\sum_{i=1}^m (\theta^T x^{(i)})x_j^{(i)} \quad \forall j \in 0...n
$$

### Practical Tricks

__Feature Scaling__:

This speeds up gradient descent.

1. Make sure that features are on the same scale, by normalization or standardization.
2. Get feature into range between -1 and 1 for linear analysis, and best centered around zero.

standardization:

$$
x_i \coloneqq \frac{x_i - \mu_i}{\sigma_1}
$$

This uses the assumption that the data follows a Gaussian distribution, and so we find the mean and the variance to use for standardization.


__Learning Rate__:

Fine-tune the learning rate by tricks such as early stopping; we want to have a learning rate that steadily keeps the loss function to decrease. We can set up a threshold (i.e. $10^{-3}$) and see if the amount of drecreasing is less than this threshold; if so we abort. Larger learning rate could be imprecise and sometimes jump over the minimum entirely (sometimes resulting in an increasing value of cost function) and may not converge, and very small learning rate makes it slow to converge.


#### Feature and Polynomial Regression

Sometimes we can create new features, and sometimes we would like to fit a quadratic model. using a quadratic function. i.e.

$$
h(x^{(i)}) = \sum_{j=0}^n \theta_j \phi(x^{(i)}_j)
$$

where the $\phi$ is a __basis function__ which can turn the feature into nonlinear basis. With basis functions and in the case of polynomial regression, feature scaling becomes more intricate than before.

### Analytical Linear Regression

__Design matrix__ for the feature inputs: 

$$
X = \begin{bmatrix}
(x^{(1)})^T \\
(x^{(2)})^T \\
... \\
(x^{(m)})^T 
\end{bmatrix}
$$

where $x^{(i)}$ is a $(n+1) \times 1$ matrix, making the design matrix of dimension $m \times (n+1)$. Since  hypothesis for each input features is $h(x^{(i)}) = \theta^T x^{(i)}$, the overall notation for the cost function of the entire training data is:
$$
J(X, \theta) = \frac{1}{2m}(X \theta - y)^2 = \frac{1}{2m}(X \theta - y)^T (X \theta  - y)
$$
where $y$ is the target vector of dimension $m \times 1$. This cost function can be analyzed analytically because it is convex, and the minimum can be computed using the normal equation:
$$
\Theta = (X^T X)^{-1} X^T y
$$
Compared to gradient descent, normal equation is not scalable despite its exactness. In terms of complexity theory, gradient descent is $O(kn^2)$ while normal equation is $O(n^3)$.

Scenarios where the normal equations might not be invertible: 
- Redundant features, where two features are very closely related (i.e. they are linearly dependent)
- Too many features (e.g. m ≤ n). In this case, delete some features or use "regularization" (to be explained in a later lesson).

Solutions to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.

### Appendix: Normal Equation Derivation

The cost function is written as 
$$
J(X, \theta) = \frac{1}{2m}(X \theta - y)^T (X \theta -y)
$$
To get the value for $\theta$ we solve for $\frac{\partial J}{\partial \theta}$, in which case we can throw out the constant product term, and after openning the bracket we can get the following:
$$
J(X, \theta) = \Theta^T X^T X \theta - 2(X\theta)^T y + y^T y
$$

solving for the partial derivative using the following theorems of matrix calculus: 

$$
\begin{aligned}
& \frac{\partial Ax}{\partial x} = A^T \\ 
& \frac{\partial x^T A}{\partial x} = A \\
& \frac{\partial A^T x A}{\partial x} = Ax + A^T x
\end{aligned}
$$

we get the following result:

$$
\begin{aligned}
& \frac{\partial J}{\partial \theta} = 2X^T X\theta - 2X^T y = 0
& X^TX \theta = X^T y 
\end{aligned}
$$

where the last equation gives us the normal equation,

$$
\theta = (X^T X)^{-1} X^T y
$$

* link for the theorems of matrix algebra: http://cs229.stanford.edu/section/cs229-linalg.pdf

