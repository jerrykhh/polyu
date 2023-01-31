# COMP5541 Quiz 1

### 1. KNN algorithm is always more suitable for regression tasks than classification tasks. (5 points)

A. True

B. False

### 2. To determine the parameters W and b for linear regression algorithms, all the training/validation/test data points together with their ground truth labels must be used in gradient descent. (5 points)

A. True

B. False

### 3. To label three categories of fruits, 'apple', 'pineapple', "banana', which of the following encoding methods is correct? (5 points)

A. apple: -1, pineapple: 0, banana: 1

B. apple: [0 1 0], pineapple: [0 0 1], banana: [1 0 0]

### 4. Linear regression models are designed to classify data points with non-linear relationships. (5 points)

A. True

B. False

### 5. Once the parameters of linear regression models are well learned, the training data can be discarded. This is also true for KNN algorithm. (5 points)

A. True

B. False

### 6. KNN algorithm cannot classify data points with non-linear relationships, because it only has a hyperparameterK. (5 points)

A. True

B. False

### 7. Logistic regression models are designed for linear regression tasks. (5 points)
   
A. True

B. False

### 8. If there are 50 training data points, each point has 4 features, how many parameters do we need to learn for a simple linear regression model? (Note: you need to add the bias term separately) (5 points)

A. $50\times (4+1)=250$

B. $50\times 4+1=201$

C. $5$

### 9. We use gradient descent to optimize the parameters of a linear regression model. In order to ensure that the moving direction of the parameters is correct, we need to manually check whether the gradient is positive or negative at every step, and then decide the updating mechanism for each parameter. (5 points)

A. True

B. False

### 10. Since the gradient descent method can guarantee the loss value to be smaller and smaller, the optimal values of all parameters will be obtained given enough training steps. (5 points)

A. True

B. False

### 11. Because basis expansion strategy is able to model complex relationships between data points X and labels y, we should always choose higher order of expansion strategy to capture nonlinear relationships. (5 points)

A. True

B. False

### 12. Cross-validation is only useful to overcome the overfitting issue. (5 points)

A. True

B. False

### 13. A more complex model is more likely to suffer from underfitting issue. (5 points)

A. True

B. False

### 14. Because logistic regression model has a non-linear function Sigma, it can capture the non-linear relationships between X and y for binary classification. (5 points)

A. True

B. False

### 15. There are four digits: 1, 2, 3, 4. Logistic regression can classify these points into two groups: 1) even and 2) odd numbers. (5 points)
    
A. True

B. False

### 16. Naive Bayes classifier can classify data points with non-linear relationships between X and y. (5 points)
    
A. True

B. False

### 17. Similar to logistic regression, Naive Bayes classifier will have better performance if we use feature expansion strategy. (5 points)
    
A. True

B. False

### 18. If some feature values of some training data points are missing, which method can be used without modifying the training set? (5 points)

A. Logistic Regression

B. Naive Bayes Classifier

### 19. Both Naive Bayes classifier and logistic regression method assume that the features are independent. (5 points)
    
A. True

B. False

### 20. Because a logistic regression model has much more parameters than a Naive Bayes classifier, the former will have much better accuracy on a classification task. (5 points)

A. True

B. False

## Solution

1. False
2. False
3. apple: [0 1 0], pineapple: [0 0 1], banana: [1 0 0]
4. False
5. False
6. False
7. False
8. 5
9. False
10. False
11. False
12. False
13. False
14. False
15. False
16. True
17. False
18. Naive Bayes Classifier
19. False
20. False