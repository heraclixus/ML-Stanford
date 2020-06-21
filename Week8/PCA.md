Application of dimensionality reduction:
- Data compression
- Data visualization 


__Main idea__: (for n to k dimensional case) Find $k$ vectors $\{ \mu^{(1)}, ..., \mu^{(k)} \}$ onto which to project data, so as to minimize projection error.

#### Algorithm

__Data Preprocessing__:
Feature scaling/mean normalization: 
$$
\mu_j = \frac{1}{n}\sum_{i=1}^m x_j^{(i)}
$$
Replace each $x_j^{(i)}$ with $x_j - \mu_j$. If different features on different scales, scale features to comparable range of values.

__Find Principal Components__:

Compute covariance matrix:
$$
\Sigma = \frac{1}{m}\sum_{i=1}^n x^{(i)}{x^{(i)}}^T
$$
Compute eigenvectors of $\Sigma$, i.e. `[U,S,V] = svd(sigma)`. Where $U$ is the square matrix whose columns are eigenvectors:

The first $k$ columns of $U$, $\mu_1, ...,\mu_k$ would then be chosen as the principal components. Let the result matrix be $U_k$ of shape $n \times k$, then we can compute the new features from old feature $x \in \mathbb{R}^n$ as:
$$
z = U_k^T x
$$
which is of dimension $k \times 1$.

```
sigma = (1/m) * X' * X;
[U,S,V] = svd(Sigma);
Ureduce = U(:, 1:k);
z = Ureduce' * x;
```

#### Application

__Reconstruction from Compressed Representation__: The PCA algorithm is invertible-given a compressed object using PCA, we can retrieve the original object.

__Choosing the Number of PCs__:

choose $k$ to be smallest value so that:
$$
\frac{\frac{1}{m}\sum_{i=1^m} ||x^{(i)} - x^{(i)}_{\text{approx}}||^2} {\frac{1}{m}\sum_{i=1}^m || x^{(i)}||^2} \leq 0.01
$$
where the numerator is the average squared projection error and the denominator is the total variation in the data.

Iteratively we can start with $k=1$ and keep increasing it to meet the above requirement. But this is expensive; intead use the $S$ matrix from the svd decomposition, and then use the following theorem: For given $k$, we can compute the quantity as:
$$
1- \frac{\sum_{i=1}^k S_{ii}}{\sum_{i=1}^n S_{ii}} \leq 0.01
$$
this makes the computation easier.

#### Advices for Applying PCA

PCA Speedsup supervised learning problems with high dimensional inputs: First, extract the inputs from unlabeled dataset and apply PCA; with the new dataset with fewer dimensions, assign labels and start supervised learning.
- The PCA mapping $x^{(i)} \rightarrow z^{(i)}$ should be established on the training set.
- This mapping can be applied as well to the examples in cross validation and test set.

__Bad use: prevent overfitting__: The argument is that fewer features means less likely to overfit. This is a bad practice, because it doesn't address overfitting; use regularization instead. The reason is because PCA doesn't have information about the target, but when solving overfitting, we need to have information about the target value.

__PCA when misused__:
- people shouldn't just start with always using PCA
- Instead, better to consider the case where we don't use PCA first, and if that doesn't work, try PCA.