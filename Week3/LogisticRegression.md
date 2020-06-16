### Classification

Problem of using linear regression for classification: the output of the linear regression covers the whole $\mathbb{R}$, while for classification problems, we assign labels from a finite set, and in this case, 0 or 1. This means we want the output of the classification algorithm to have values between 0 and 1. This problem motivates the use of discriminant function to limit the regression outcome to be a number between 0 and 1, and that gives rise to __logistic regression__.

### Logistic Regression

#### Hypothesis Representation

in order to make $0 \leq h_{\theta}(x) \leq 1$, apply discriminant function (in this case, the sigmoid function) to its output:
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
so that we get:
$$
h_\theta (x) = \frac{1}{1 + e^{-\theta^T x}}
$$

The output of the logisitic regression can be interpreted as likelihood for our target $P(y = 1| x; \theta)$. If $P(y = 1 | x; \theta) = x$, then $P(y=0 | x; \theta) = 1-x$. 

#### Probabilistic Representation

From the properties above, we can clearly see that from a frequentist view, the samples follow a Bernoulli distribution. Where the bernoulli parameter $p = \sigma(\theta^T x)$.

Recall that bernoulli probability mass function is:
$$
f(k; p) = p^k (1-p)^{1-k}, k \in \{ 0 ,1 \}
$$

So that the pmf for our case is:
$$
\begin{aligned}
& P(y_i | x^{(i)}, \theta) = Ber(y_i ; h_\theta (x^{(i)})) = Ber(y_i; \sigma(\theta^T x^{(i)})) \\
& P(y = y_i | x = x^{(i)}) = \sigma (\theta^T x^{(i)})^{y^{(i)}} [1-\sigma(\theta^T x^{(i)})]^{(1-y_i)}
\end{aligned}
$$

We can get the likelihood function:

$$
L(\theta) = \prod_{i=1}^m P(y = y^{(i)} | x = x^{(i)}) = \prod_{i=1}^m \sigma(\theta^T x^{(i)})^{y^{(i)}} [1-\sigma(\theta^T x^{(i)})]^{1-y^{(i)}}
$$

and the negative log-likelihood:
$$
NLL(\theta) = -\sum_{i=1}^m  y^{(i)} \log \sigma(\theta^T x^{(i)}) + (1-y^{(i)})\log[1-\sigma(\theta^T x^{(i)})]
$$

This negative log-likelihood represents the __cross entropy__ function, and can only be optimized using iterative methods, such as gradient descent.

#### Decision Boundary

Here for simplicity, I'm using $x$ for any sample from our training set. suppose that we predict "y = 1" if $h_\theta (x) \geq 0.5$, then this is equivalent to say that $\theta^T x \geq 0$. The hyperplane for $\theta x =0$ represents the logistic decision boundary.

Problem with the crude logistic regression doesn't work when we have a non-linear decision boundary, so in the same form we can apply basis function to the linear regression model inside: 

$$
h_\theta (x)  = \sigma(\theta^T \phi(x))
$$

where the nonlinear basis function could transform the decision boundary to nonlinear ones. 


#### Cost Function

Using the cost function of linear regression results in non-convex cost function; the cost function for logistic regression is then:
$$
\text{Cost}(h_\theta (x), y) = \begin{cases}
-\log (h_\theta(x)) & \text{if } y = 1 \\
-\log (1- h_\theta(x)) &\text{if } y = 0
\end{cases}
$$

Intuitively it penalizes the wrong predictions exponentially, because:
$$
\begin{aligned}
&\text{Cost}(h_\theta(x), y) = 0 && \text{ if } h_\theta(x) = y \\
&\text{Cost}(h_\theta(x), y) \rightarrow \infty && \text{if } y = 0 \text{ and } h_\theta(x) \rightarrow 1 \\
&\text{Cost}(h_\theta(x), y) \rightarrow \infty && \text{if } y = 1 \text{ and } h_\theta(x) \rightarrow 0
\end{aligned}
$$

The whole dataset therefore have the cost function as:
$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \text{Cost}(h_\theta(x^{(i)}, y^{(i)}))
$$


The cost function can be rewritten as:
$$
\begin{aligned}
& \text{Cost}(h_\theta(x), y) = -y \log (h_\theta(x)) - (1-y)\log(1-h_\theta(x)) \\
& J(\theta) = -\frac{1}{m} \sum_{i=1}^m y_i \log (h_\theta(x^{(i)})) + (1-y_i)\log(1-h_\theta(x^{(i)})) \\
& J(\theta) = \frac{1}{m} \cdot [-y^T \log(h)) - (1-y)^T \log(1-h)]
\end{aligned}
$$

and it just looke like the NNL we derived ealier. Therefore the cross entropy cost function follows from the assumption that data we have are generated i.i.d using a Bernoulli distribution.

The cost function doesn't have a closed form solution, so we use gradient descent to solve it. The key formula is:

$$
\theta_j \coloneqq \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
$$
where 
$$
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)})x_j^{(i)}
$$

again for each paramter we are doing batch update.

The vectorized form:

$$
\theta \coloneqq \theta - \frac{\alpha}{m} X^T (g(X\theta) - y)
$$

#### Advanced Optimization

Example of optimization function given in the lecture: Conjugate gradient, BFGS, L-BFGS. They usually converge faster than gradient descent.

To use these functions, provide function that evalutes the cost function and its gradient in respect to each $\theta$, i.e.: $J(\theta)$ and $\frac{\partial}{\partial \theta_j} J(\theta)$.

example of using optimization function in Octave:

```octave
function [jVal, gradient] = costFunction(theta)
    jVal = [...code to compute J(theta)...]
    gradient = [ ...code to compute derivative of J(theta)...];
end

options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

#### Multiclass Logistic Regression

- Option 1: run one vs. all using binary logistic regression. With a new sample, run all classifiers and use the one that returns the highest value.
- Option 2: simply chage the discriminant function from sigmoid to softmax:

$$
\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$