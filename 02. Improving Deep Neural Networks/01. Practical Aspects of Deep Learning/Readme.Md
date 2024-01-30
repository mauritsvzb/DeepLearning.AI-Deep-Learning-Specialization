# Practical Aspects of Deep Learning

## Train / Dev / Test sets
* It's impossible to get all your hyperparameters right on a new application from the first time
* So the idea is you go through the loop: Idea ==> Code ==> Experiment
* You have to go through the loop many times to figure out your hyperparameters
* Your data will be split into three parts:
  * Training set (Has to be the largest set)
  * Cross validation set / Development or "dev" set
  * Testing set
* You will try to build a model usingg the training set, then try to optimize hyperparameters on the dev set as much as possible. Then after your model is ready, you try and evaluate the testing set
*  So the trend on the ratio of splitting the models (what matters is that the absolute number of examples in the test set is high enough):
  * If size of the dataset is 100 to 1.000.000 ==> 60/20/20
  * If size of the dataset is over 1.000.000 ==> 98/1/1 or 99.5/0.25/0.25
* The dev/test set just needs to be big enough for evaluating different algorithm choices and quickly decides which one is better => does not need the whole 20% dataset for that (in a large dataset case)
* The trend now gives the training data the biggest sets
* Make sure the dev and test set are coming from the same distribution (i.e., acquired in the same way e.g., in  binary classification, use camereras that yield identical or highly simular results).
  * For example, if cat training pictures is from the web and the dev/test pictures are from users cell phone they will mismatch. It is better to make sure that dev and test set are from the same distribution.
* The dev set rule is to try them on some of the good models you've created.
* It's OK to only have a dev set without a testing set. But a lot of people in this case call the dev set as the test set. A better terminology is to call it a dev set as its used in the development.

## Bias / Variance
* Bias / variance techniques are easy to learn, but difficult to master
* So here the explanation of Bias / Variance:
  * If your model is underfitting (logistic regression of non linear data) it has a "high bias"
  * If your model is overfitting then it has a "high variance"
  * Your model will be alright if you balance the bias / variance

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/a5117d29-3f14-4652-8252-48dada6a6ba8.png" width="700" />

* Another idea to get the bias / variance if you don't have a 2D plotting mechanism:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/c1317dcb-b358-47cc-beaa-24c504c10d18.png" width="700" />

* These assumptions based on the idea that a human has 0% error in classifying images. If the problem that you're trying to solve isn't like that e.g., could be dealing with blurry images that even a human has difficulties with classifying, you'll need to use a different human error as baseline.

## Basic Recipe for Machine Learning
* If your algorithm has a high bias:
  * Try to make your NN bigger (size of hidden units, number of layers)
  * Try a different model that is suitable for your data
  * Try to run it longer
  * Different (advanced) optimization algorithms
* If your algorithm has a high variance:
  * More data
  * Try regularization
  * Try a different model that is suitable for your data
* You should try the previous two points until you have a low bias and low variance
* In the older days before deep learning, there was a "Bias/variance tradeoff". But because now you have more options/tools for solving the bias and variance problem its really helpful to use deep learning
* Training a bigger neural network never hurts

## Regularization
* Adding regularization to NN will help it reduce variance (overfitting)
* L1 matrix norm:
  * `||W|| = Sum(|w[i,j]|)  # sum of absolute values of all w`
* L2 matrix norm because of arcane technical math reasons is called Frobenius norm:
  * `||W||^2 = Sum(|w[i,j]|^2)  # sum of all w squared`
  * Also can be calculated as `||W||^2 = W.T * W` if W is a vector
* Regularization for logistic regression:
  * The normal cost function that we want to minimize is: `J(w,b) = (1/m) * Sum(L(y(i),y'(i)))`
  * The L2 regularization version: `J(w,b) = (1/m) * Sum(L(y(i),y'(i))) + (lambda/2m) * Sum(|w[i]|^2)`
  * The L1 regularization version: `J(w,b) = (1/m) * Sum(L(y(i),y'(i))) + (lambda/2m) * Sum(|w[i]|)`
  * The L1 regularization version makes a lot of w values zero, which makes the model more sparse i.e., model size smaller
  * L2 regularization is being used much more often.
  * lambda here is the regularization parameter (hyperparameter)
* Regularization for NN:
  * The normal cost function that we want to minimize is: 
    `J(W1,b1...,WL,bL) = (1/m) * Sum(L(y(i),y'(i)))`
* The L2 regularization version:
  * `J(w,b) = (1/m) * Sum(L(y(i),y'(i))) + (lambda/2m) * Sum((||W[l]||^2)`
* We stack the matrix as one vector `(mn,1)` and then we apply `sqrt(w1^2 + w2^2.....)`
* To do back propagation (old way):
  * `dw[l] = (from back propagation)`
* The new way:
  * `dw[l] = (from back propagation) + lambda/m * w[l]`
* So plugging it in weight update step:
```
w[l] = w[l] - learning_rate * dw[l]         % original
     = w[l] - learning_rate * [(from back propagation) + w[l] * lambda/m ]  % 
     = w[l] - (learning_rate*lambda/m) * w[l] - learning_rate * (from back propagation)
     = [1 - (learning_rate*lambda)/m] * w[l] - learning_rate * (from back propagation)
```
* In practice this penalizes large weights and effectively limits the freedom in your model.
* The new term (learning_rate*lambda)/m is little bit less than 1 and causes the weight to decay in proportion to its size.
