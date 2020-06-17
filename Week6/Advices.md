### Train-Validation-Test

#### Debugging a learning algorithm

Given a linear model:

$$
J(\theta) = \frac{1}{2m} (h(x) - y)^T (h(x) - y) + \lambda || \theta||^2
$$

when the model performs badly on novel data, what do we do?

- Get more training examples (however sometimes it doesn't help)
- Try smaller set of features (feature selection)
- Try getting additional features
- Try adding polynomial features (in linear models)
- Try tuning the regularization learning rate

It is a good practice to do algorithm diagnosis, which can give guidance to what might be more fruitful things to try to improve learning algorithms.

#### Evaluating a Hypothesis

use training set for training, test set for evaluation. Overfitted models in general have high training accuracy but low testing accuracy. Evaluation metrics include:

- Accuracy
- Precision, Recall, F-1 Score
- AUC

#### Model Selection

- First split the data into training, validation, and test set.
- Create multiple different models, use the same training data to train them.
- Evaluate the models on the validation set; choose the best model
- Don't use the test set for model selection, because this will make the parameters overly optimistic:
  - In order to reduce the validation error, we introduced hidden parameters to lower the cost function.
  - The whole training process and model selection process shouldn't have any knowledge of the test set.

The validation set is usually picked with __cross-validation__. the split ratio of all data can be:

- 60% as training set
- 20% as validation set
- 20% as test set

After model selection, use test set to estimate the model's generalization.

### Bias Variance Tradeoff

It is helpful to plot both the validation and the training error. The ideal stopping point is to find the point where validation error reaches an inflection point/saddle point. After the inflection point overfitting occurs.

- When training accuracy is high but validation is low: then we have a variance problem (stop training earlier).
- When training accuracy is high and validation also high: we have a bias problem (keep on training).

Effect of the regularization hyperparamter:

- When it has a large value, all paramters are penalized, and we have underfitting/bias problem.
- When it has a very small value, we have a variance/overfitting problem.

How to choose a good regularization parameter? - __Hyperparamter tuning__

- Set a candidate list of hyperparamters and try each combination separately. Use validation set for model selection as well as hyperparameter tuning.
- Large neural network needs regularization to address overfitting


### Design Example: Spam Classifier


#### Prioritizing What to Work On

- Collect lots of data
- Develop rich and sophisticated feature
- Develop algorithms to process input in different ways
- start with simple algorithm, plot the learning curve, decide whether we need more complex features/algorithms or not.

Error analysis: manually examine the examples in cross validation to spot where algorithm made mistakes. 


### Handling Skewed Data
This is when the class distribution in the training set is very disproportional: i.e. number of positive samples much less than number of negative samples. No need for a classifier.
- Instead of using accuracy: use precision and recall.

Confusion matrix:

|                 | 1 (Actual)          | 0 (Actual)          |
|-----------------|---------------------|---------------------|
| __1 (Predict)__ | TP (True Positive)  | FP (False Positive) |
| __0 (Predict)__ | FN (False Negative) | TN (True Negative)  |

- Precision: of all of patients where we predicted 1, what fraction has cancer.
- Recall: Of all patients that have cancer, what fraction did we correctly predict.
- F-1 score: harmonic mean of the above, gives a more balanced metrics.

$$
\begin{aligned}
\text{Precision} = \frac{TP}{TP + FP} \\
\text{Recall} = \frac{TP}{TP + FN} \\
\text{F-1 Score} = 2\frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\end{aligned}
$$

__Precision-Recall curve__ is a good visualization for the behavior of precisoin vs. recall, as most of the times we need to make trade offs between these two.

When choosing a decision threshold for classificatio; it is good practice to __choose threshold that maximizes the F-1 score on validation set__.


### Using Large Dataset

When is it true that the more data the better? 
- The feature of the data must be informative; otherwise mere accumulation of data doesn't help at all.
- Given the information available, can a human expert confidently solve the problem? 

Large training set fits perfectly with deep neural networks, which could have millions of paramters to tune (provided that features are informative).
