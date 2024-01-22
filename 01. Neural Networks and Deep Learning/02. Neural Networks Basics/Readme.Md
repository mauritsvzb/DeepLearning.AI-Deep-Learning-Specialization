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
* Algorithm is used for classification algorithm of 2 classes
* Equations:
   * Simple equation: y = w * x + b
   * If x is a vector: y = w<sup>T</sup> * x + b
   * If we need y to be in between 0 and 1 (probability): y = sigmoid(w<sup>T</sup> * x + b)
   * In some notations this might be used: y = sigmoid(w<sup>T</sup> * x)
      * While b is w<sub>0</sub> of w and we add x<sub>0</sub> = 1. But we won't use this notation in the course (Andrew said that the first notation is better)
* In binary classification Y has to be between 0 and 1
* In the last equation, w is a vector of N<sub>x</sub> and b is a real number

## Logistic regression cost function
* First loss function would be the square root error: L(ŷ, y) = 1/2 (ŷ - y)<sup>2</sup>
   * But we won't use this notation because it leads us to optimization problem which is non convex, means it contains multiple local optima instead of 1 global optimum
* This is the function that we will use: L(ŷ, y) = - (y * log(ŷ) + (1 - y) * log(1 - ŷ))
* To explain the last function lets see:
   * if y = 1 ==> L(ŷ, 1) = - log(ŷ) ==> we want ŷ to be the largest ==> ŷ biggest value is 1
   * if y = 0 ==> L(ŷ, 0) = - log(1 - ŷ) ==> we want 1 - ŷ to be the largest ==> ŷ to be smaller as possible because it can only has 1 value
* Then the Cost function will be: J(w,b) = (1/m) * Sum( L (ŷ[i], y[i]))
* The loss function computes the error for a single training example; the cost function is the average of the loss functions of the entire training set

## Gradient Descent
* We want to predict w and b that minimize the cost function
* Our cost function is convex ie 1 global optimum (= minimum)
* First we initialize `w` and `b` to 0,0 or initialize them to a random value in the convex function and then try to improve the values the reach minimum value
* In Logistic regression people always use 0,0 instead of random  (it doesn't matter where  you start, you'll always arrive at the global optimium or very close to it)
* The gradient decent algorithm repeats: `w = w - alpha * dw`, where `alpha` is the learning rate (i.e., how big of a step in horizontal plane) and `dw` is the derivative of `w` (Change to w OR slope of 'w' in vertical plane).
* Looks like greedy algorithms. the derivative give us the direction to improve our parameters (i.e., direction of largest descent)
* The actual equations we will implement:
   * `w = w - alpha * d(J(w,b) / dw)` (how much the function slopes in the w direction)
   * `b = b - alpha * d(J(w,b) / db)` (how much the function slopes in the b direction)
<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/9c5e09d1-6a5d-4a38-a319-b37d2e262d60.png" width="600" /><img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/b03206bb-07dd-451e-a423-dbcd831984ea.png" width="600" />

## Derivatives
* We will talk about some of required calculus
* You don't need to be a calculus geek to master deep learning but you'll need some skills from it
* Derivative of a linear line is its slope
   * ex. f(a) = 3a d( f(a) ) / d(a) = 3
   * if a = 2 then f(a) = 6
   * if we move a a little bit a = 2.001 then f(a) = 6.003 means that we multiplied the derivative (slope) to the moved area and added it to the last result

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/67e31110-f7f8-4044-920c-72fe6def61c9.png" width="600" />

## More Derivatives examples
* f(a) = a<sup>2</sup> ==> d(f(a))/d(a) = 2a
* a = 2 ==> f(a) = 4
* a = 2.0001` ==> f(a) = 4.0002 approx
* f(a) = a<sup>3</sup> ==> d( f(a) ) / d(a) = 3 * a<sup>2</sup>
* f(a) = log(a) ==> d( f(a) ) /d (a) = 1/a
* To conclude, Derivative is the slope and slope is different in different points in the function thats why the derivative is a function

## Computation graph
* Its a graph that organizes the computation from left to right
   * <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/3e924bac-01af-45bd-9617-a81d20e04063.png" width="600" />

## Derivatives with a Computation Graph
* Calculus chain rule says: If `x -> y -> z` (x effect y and y effects z) Then `d(z) / d(x) = d(z) / d(y) * d(y) / d(x)`
* The video illustrates a big example
   * <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/e8f480cf-33b3-46d4-94c6-7e2891714fdc.png" width="600" />
* We compute the derivatives on a graph from right to left and it will be a lot more easier
* `dvar` means the derivatives of a final output variable with respect to various intermediate quantities

## Logistic Regression Gradient Descent
In the video he discussed the derivatives of gradient decent example for one sample with two features `x1` and `x2`
* <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/fa9f581b-981c-4be4-9377-19280f4c9fe5.png" width="600" />


