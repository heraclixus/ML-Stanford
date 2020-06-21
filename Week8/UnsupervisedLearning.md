### K-Means Clustering

Clustering is an example of unsupervised learning, in which we are trying to find patterns/structures in unlabeled data.

__Input__:
- K: number of clusters
- $(x^{(1)}, ..., x^{(m)})$, unlabeled training data
- $x^{(n)} \in \mathbb{R}^n$

Algorithm:
- Randomly assign $K$ centroids, $\mu_1, ..., \mu_K$.
- Repeat:
  - For $i$ = 1 to $m$
    - $c^{(i)} =$ index of cluster centroids closest to x^{(i)}; this is the cluster that $x^{(i)}$ is currently assigned to
  - For $k$ = 1 to $K$
    - $\mu_k$ = mean of points assigned to cluster $k$

#### Optimization Objective

One additional paramter to keep track of: $\mu_{c^{(i)}}$: cluster centroid of cluster to which $x^{(i)}$ has been assigned.

Optimization objective:
$$
\min_{c^{(i)}, ..., c^{(m)}, \mu_1, ..., \mu_{K}} J(c^{(i)}, ..., c^{(m)}, \mu_1, ..., \mu_{K}) = \frac{1}{m}\sum_{i=1}^m || x^{(i)} - \mu_{c^{(i)}} ||^2
$$

The algorithm itself is actually minimizing this cost function, using EM algorithm.

#### Random Initialization 

- Have $K < m$
- Randomly pick $K$ training examples
- set $\mu_1, ..., \mu_k$ equal to these training examples. 

#### Choosing K

General hyperparameter tuning. Plot cost vs. number of clusters and look for the saddle point in the trend.
In general, more $k$ gives less cost. When the reverse is true, we may need to re-randomize the initial points and run more experiments.

Sometimes K-means is run for some downstream purposes, and we need to evalute k-means for how well it performs for that downstream purpose.


