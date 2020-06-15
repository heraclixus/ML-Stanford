## From PRML

#### Linear Regression with Basis Function

The PRML book dives deeper into the mathematical background of linear regression. Here I will keep using the same notation as the Stanford ML course instead of the PRML one, so as to avoid confusion and maintain consistency. 

The notations are:
* $x = \{ x^{(1)}, ..., x^{(i)}, ..., x^{(m)} \}$ the training dataset
* $n$ as the number of features
* $m$ as the number of samples.
* $x^{(i)}$: input features of the ith training example.
* $x^{(i)}_j$ the jth feature of the ith training example. 
* $h$ is the hypothesis.
* $y$ is the target vector.

Here with basis function, a new value is introduced to denote the number of basis functions we use: $k$, and with the restriction $k \leq m$, since at most we have basis function for each sample. For simplicity of notation, $\phi_0(x^{(i)}) = 1 \forall i$.

$$
h(x^{(i)}, \theta) = \theta^T \phi(x^{(i)}) = \sum_{i=0}^{k-1} \theta_j \phi_j(x^{(i)}) 
$$

Dimension of $\theta^T$ has the same dimension as the basis function vector $\phi$. Specifically $\theta = (\theta_0, \theta_1, ..., \theta_{k-1})$ and $\phi = (\phi_0, ..., \phi_{k-1}).$

The basis function allows flexibility to the regression. For example, it can be a polynomial function that turns the model into a polynomial regression, or it can be a Gaussian function that turns it into linear kernel model. Despite the flexibility of basis functions, the model is still a linear model, because the function form is linear in regard to parameter $\theta$.

#### Frequentist Probabilistic Perspective: Maximum Likelihood

Here is the frequentist view of linear regression. In short, it says that "each data generated is a linear function with a Gaussian noise centered around zero":

$$
\begin{aligned}
& y_i = h(x^{(i)}, \theta) + \epsilon \\
& \epsilon \in \mathcal{N}(0, \beta^{-1})
\end{aligned}
$$

The gaussian noise is centered around zero, with precision $\beta$, and the above form gives the likelihood function for the whole distribution:

$$
p (y_i| x^{(i)}, \theta, \beta) = N(y_i| h(x^{(i)},\theta), \beta^{-1})
$$

The interpretation is basically that each data point we observe is from Gaussian distribution centered around the modeled target value, $h(x, \theta)$

The frequentist assumption is that the observed data is generated from the true distribution, and so we need to maximize the likelihood that the data we observed is generated from it--this means that observed model is the most probable given the hypothesis. The likelihood function has the form $\prod_{i=1}^m p(y_i|x^{(i)}, \beta)$

The log-likelihood function (the normal likelihood function is a product form, and logarithm turns it into summation form--much easier to deal with) is:
$$
\ln p(y|\theta, \beta) = \sum_{n=1}^m \ln \mathcal{N}(y_i | h(x^{(i)}, \theta), \beta^{-1}) = \frac{m}{2} \ln \beta - \frac{m}{2} \{ y_i - \theta^T \phi(x^{(i)}) \}^2 
$$
Setting its gradient to zero:
$$
\nabla_\theta \ln p(y|\theta, \beta) = \sum_{i=1}^m \{ y_i - \theta^T \phi(x^{(i)}) \} \phi(x^{(i)})^T = 0
$$

which gives us:
$$
\sum_{i=1}^m y_i \phi(x^{(i)})^T = \theta^T \Big( \sum_{i=1}^m \phi(x^{(i)})\phi(x^{(i)})^T   \Big)
$$

This can be translated into matrix form as:
$$
\begin{aligned}
& \theta^T (\phi \phi^T) = y^T \phi \\
& (\phi^T \phi) \theta = \phi^T y
\end{aligned}
$$

where $\phi$ is a $m \times k$ matrix with entry $\phi_{ij} = \phi_j(x^{(i)})$. It is called also the __design matrix__, now with basis function. Its form is:
$$
\begin{bmatrix}
\phi_0(x^{(1)}), & \phi_1(x^{(1)}), & ... & \phi_{k-1}(x^{(1)}) \\
\phi_0(x^{(2)}), & \phi_1(x^{(2)}), & ... & \phi_{k-1}(x^{(2)}) \\
...
\phi_0(x^{(m)}), & \phi_1(x^{(m)}), & ... & \phi_{k-1}(x^{(m)})
\end{bmatrix}
$$


Solving the equation gives the result: 
$$
\theta  = (\phi^T \phi)^{-1} \phi^T y
$$

This is the normal equation under the basis function.

Another property that the book proved is that maximum likelihood estimation on linear regression is equal to the use of least square cost function, which is used in the lecture.

The Bayesian approach to linear regression deserves a note of its own due to its complexity.