### Anomaly Detection

Define a threshold and a standard for "normal", i.e. a probability distribution or a threshold value.

The most frequently used probability distirbution here is the Gaussian:
$$
\begin{aligned}
& p(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \\
& \mu = \frac{1}{n}\sum_{i=1}^n x_i \\
& \sigma^2 = \frac{1}{m} \sum_{i=1}^n (x_i - \mu)^2  
\end{aligned}
$$

For anomaly detection, we assume each sample i.i.d, following a Gaussian $N(x_i; \mu_i, \sigma_i)$.
$$
P(X) = \prod_{i=1}^n P(x_i; \mu_i, \sigma_i)
$$

For $X = \{ x^{(1)}, ..., x^{(m)}\}, x^{(i)} \in \mathbb{R}^n$, we have the following algorithm:
1. Choose feature $x_i$ that might be indicative of anomalous behavior.
2. Fit paramters:

$$
\begin{aligned}
\mu_j = \frac{1}{m} \sum_{i=1}^m x_j^{(i)} \\
\sigma_j = \frac{1}{m}\sum_{i=1}^m (x_j^{(i)} - \mu_j)^2
\end{aligned}
$$

3. Given new example $x$, compute 

$$
P(x) = \prod_{j=1}^n p(x_j; \mu_j, \sigma_j)
$$

Output anomaly if $p(x) < \epsilon$, for a threshold $\epsilon$.

#### Anomaly Detection vs Supervised Learning

- Anomaly detection needs to train on large number of negative samples and only a small number of positive samples.
- In supervised learning we need a large number of both.
- In anomaly detection there are many different types of anomalies, some not even discovered before 
- Future anomalies may look nothing like what the model sees before.
- In supervised learning we have enough samples to give an idea of what positive samples look like, and we assume future samples to be similar.

#### Choosing what features to use

We want probability to be large for common examples and small for anomalies. So a problem would rise if anormal samples also have high probability. In this case we can try to find other features that more exemplifies this anormaly.
- Find values that could take unusually large or small valuess

