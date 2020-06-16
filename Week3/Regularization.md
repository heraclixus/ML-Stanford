### Overfitting

Happens when the model fits the training data too well (proportional to the model complexity) so that it doesn't generalize well on the test set.

The tradeoff between fitting training set and validation set is called __Bias Variance Tradeoff__.
* When model complexity is low, then model has strong Bias and low Variance; this is when __underfitting__ happens.
* When model complexity is high, then model has low Bias and Strong vairance. This is when __overfitting__ happens.


#### Addressing Overfitting

1. Reduce the number of features.
    - Manually Select features.
    - Model selection algorithms.
2. Regularization
    - Reduce magnitude of the parameters $\theta$
    - add penalty terms to the cost function: higher penalty applies to higher complexity terms.

In the case of Least square cost function using L-2 norm: 
$$
\begin{aligned}
J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)})^2 + \lambda \sum_{i=1}^n \theta_j^2 \\
J(\theta) = \frac{1}{2m} (h_\theta(x)-y)^T (h_\theta(x)-y) + \lambda || \theta ||^2
\end{aligned}
$$

the regularization paramter $\lambda$ needs to be picked with not too large (this leads to underfitting) and not too small (leading to overfitting).

#### Regularized Linear Regression

With the new cost function above, the gradient descent rule changes for $\theta_j, j \neq 0$ ($\theta_0$ is not penalized):
$$
\begin{aligned}
\theta_j \coloneqq \theta_j - \alpha \Big[ \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})- y_i)x^{(i)}_j + \frac{\lambda}{m} \theta_j \Big] \\
\theta_j \coloneqq \theta_j (1-\alpha \frac{\lambda}{m}) -\alpha\frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})- y_i)x^{(i)}_j
\end{aligned}
$$

The term $(1-\alpha \frac{\lambda}{m}) < 1$, and it shrinks the paramter $\theta$ in each iteration.

With regularization term, the analytical solution (i.e. the Normal equation) becomes:
$$
\theta = (X^T X + \lambda I)^{-1} X^T y
$$

If $m < n$, or the number of samples fewer than number of features, then the original matrix $(X^T X)$ is singular; However, if $\lambda > 0$, $(X^T X + \lambda I)$ is always invertible, because it is a positive definite matrix.

#### Regularized Logistic Regression

The regularization is similar, using the L-2 norm.
$$
\begin{aligned}
& J(\theta) = -\frac{1}{m} \sum_{i=1}^m y_i \log (h_\theta(x^{(i)})) + (1-y_i)\log(1-h_\theta(x^{(i)})) + \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2 \\
& J(\theta) = \frac{1}{m} \cdot [-y^T \log(h)) - (1-y)^T \log(1-h)] + \frac{\lambda}{2m} || \theta ||^2
\end{aligned}
$$

Gradient descent update rule for $\theta_j$ (still no penalizing $\theta_0$): 
$$
\theta_j \coloneqq \theta_j - \alpha \Big[ \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})- y_i)x^{(i)}_j + \frac{\lambda}{m} \theta_j \Big]
$$

which looks just like the linear regression update rule--difference is that the hypothesis function now is $\sigma$.