# Neural Networks Basics
Learn to set up a machine learning problem with a neural network mindset. Learn to use vectorization to speed up your models

## Binary classification
* Mainly he is talking about how to do a logistic regression to make a binary classifier.

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/4ec78558-30b1-4ec6-9a04-827db7518ed8.png" width="600" />

* He talked about an example of knowing if the current image contains a cat or not.
* Here are some notations:
    * M is the number of training vectors
    * N<sub>x</sub> is the size of the input vector
    * N<sub>y</sub> is the size of the output vector
    * X(1) is the first input vector
    * Y(1) is the first output vector
    * X = [x(1) x(2).. x(M)]
    * Y = (y(1) y(2).. y(M))
* We will use python in this course.
* In NumPy we can make matrices and make operations on them in a fast and reliable time.

## Logistic regression
* Algorithm is used for classification algorithm of 2 classes.
* Equations:
   * Simple equation: y = w * x + b
   * If x is a vector: y = w<sup>T</sup> * x + b
   * If we need y to be in between 0 and 1 (probability): y = sigmoid(w<sup>T</sup> * x + b)
   * In some notations this might be used: y = sigmoid(w<sup>T</sup> * x)
      * While b is w<sub>0</sub> of w and we add x<sub>0</sub> = 1. But we won't use this notation in the course (Andrew said that the first notation is better).
* In binary classification Y has to be between 0 and 1.
* In the last equation, w is a vector of N<sub>x</sub> and b is a real number

## Logistic regression cost function
* First loss function would be the square root error: L(y',y&#x0302) = 1/2 (y' - y&#x0302)<sup>2</sup>
* But we won't use this notation because it leads us to optimization problem which is non convex, means it contains local optimum points, and not a global optimum.
* This is the function that we will use: L(y',y) = - (y*log(y^) + (1-y)*log(1-y^))
To explain the last function lets see:
if y = 1 ==> L(y',1) = -log(y') ==> we want y' to be the largest ==> y' biggest value is 1
if y = 0 ==> L(y',0) = -log(1-y') ==> we want 1-y' to be the largest ==> y' to be smaller as possible because it can only has 1 value.
Then the Cost function will be: J(w,b) = (1/m) * Sum(L(y'[i],y[i]))
The loss function computes the error for a single training example; the cost function is the average of the loss functions of the entire training set.


y&#x0302
