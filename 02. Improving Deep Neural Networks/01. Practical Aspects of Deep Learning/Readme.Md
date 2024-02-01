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

## Why Regularization Reduces Overfitting?

Explanation:
* If we adapt regularization, then Loss increases, leading W decreases (through back propagation & gradient descent) through training
* As 'W' parameter decreases, some w's will be close to zero
* With w's close to zeros, the Neural Network becomes simpler, thus reducing overfitting
* The below intuitions will discuss this matter in some specific cases

Here are some intuitions:
* Intuition 1:
  * If lambda is very large, then a lot of w's will be close to zero (via subtracting the weight decay), which will make the NN simpler (you can think of it as it would behave closer to a logistic regression)
  * If lambda is good enough it will just reduce some weights that makes the neural network not overfit but also not underfit.
* Intuition 2 (with tanh activation function):
  * If lambda is very large, w's will be small (close to zero) => then z will be small => e.g., will use the linear part of the tanh activation function, so we will go from non-linear activation to roughly linear which would make the NN a roughly linear classifier.
  * If lambda is good enough, it will just make some of tanh activations roughly linear which will reduce overfitting.

Implementation tip: if you implement gradient descent, one of the steps to debug gradient descent is to plot the cost function J as a function of the number of iterations of gradient descent and you want to see that the cost function J decreases monotonically after every elevation of gradient descent with regularization. If you plot the old definition of J (no regularization) then you might not see it decrease monotonically.

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/00bcc658-acc1-4f01-ae9d-0ecb80e1ecc0.png" width="700" />

## Dropout Regularization
* Andrew Ng tells that in most cases he uses the L2 regularization
* The dropout regularization eliminates some neurons/weights on each iteration based on a probability

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/9315a1a8-62d2-4f4f-aee3-c33abbb98815.png" width="700" />

* A most common technique to implement dropout is called "Inverted dropout"
* Code for Inverted dropout:

```
keep_prob = 0.8   # 0 <= keep_prob <= 1
l = 3  # this code is only for layer 3

# The generated number that are less than 0.8 will be dropped. 80% stay, 20% dropped
d3 = np.random.rand(a[l].shape[0], a[l].shape[1]) < keep_prob # This is a boolean that returns True (think: 1) if a random number is below the threshold, otherwise return False (think: 0)

# Keep only the values in d3; Think multiplying the old a3 activation parameters with either 1's or 0's from the d3 vector. Obviously multiplications with 0 return 0, hence 'dropping out those neurons/weights
a3 = np.multiply(a3,d3)

# Increase a3 to not reduce the expected value of output; because dividing by the keep_prob wil reduce a3 parameters. So first increase the values by an amount so that the net result of change will be 0 after division
# (ensures that the expected value of a3 remains the same) - to solve the scaling problem
a3 = a3 / keep_prob
```

* Vector d[l] is used for forward and back propagation and is the same for them, but it is different for each iteration (pass) or training example when you have m training examples.
* At test time we don't use dropout; if you implement dropout at test time - it would add noise to predictions. However, you could do this and then take the average, but this would be computationally inefficient as the results are likely going to be similar.

* Take home message: dropout regularization reduces the complexity of a NN (i.e., makes it smaller), which should have a regularizing effect

## Understanding Dropout
* Another intuition: the NN can't rely on any one feature, so have to spread out weights
* It's possible to show that dropout has a similar effect to L2 regularization
* Dropout can have different keep_probs for each layer
* The input layer dropout has to be near 1 (or 1 - no dropout) because you don't want to eliminate a lot of features.
* If you're more worried about some layers overfitting than others, you can set a lower keep_prob for some layers than others 
   * The downside is that this gives you even more hyperparameters to search for using cross-validation. 
* One other alternative might be to have some layers where you do apply dropout and some where you don't, and then you just have one hyperparameter, which is a keep_prob for the layers for which you do apply dropouts
* A lot of researchers are using dropout with Computer Vision (CV) because they have a very big input size and almost never have enough data, so overfitting is the usual problem. And dropout is a regularization technique to prevents overfitting
* A downside of dropout is that the cost function J is not well defined and so it will be hard to debug (plot J by iteration)
    * To solve that you'll need to turn off dropout, set all the keep_probs to 1, and then run the code and check that it monotonically decreases J and then turn on the dropouts again

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/e9dc8f03-1276-43a7-a288-2acf3a88e785.png" width="700" />

## Other Regularization Methods
* Data augmentation:
  * For example in a computer vision:
    * You can flip all your pictures horizontally, which will give you m more data instances
    * You could also apply a random position and rotation to an image to get more data
    * For example in OCR, you can impose random rotations and distortions to digits/letters
  * New data obtained using this technique isn't as good as the real independent data, but still can be used as a regularization technique

* Early stopping:
  * In this technique we plot the training set and the dev set cost together for each iteration. At some iteration the dev set cost will stop decreasing and will start increasing (convex shape)
  * We will pick the point at which the training set error and dev set error are best (i.e., lowest training cost with lowest dev cost)
  * We will take these parameters as the best parameters.

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/5f67060b-9842-44e2-9643-5b402d9c655e.png" width="700" />

* Andrew prefers to use L2 regularization instead of early stopping because early stopping simultaneously tries to minimize the cost function and reduce/prevent overfitting (opposite of orthogonalization, which enables you to work on each of these 2 problems independently
* But its advantage is that you don't need to search a hyperparameter like in other regularization approaches (like lambda in L2 regularization)

## Normalizing inputs
* If you normalize your inputs, then this will speed up the training process a lot
* Normalization steps are as follows:
  * Get the mean of the training set: `mean = (1/m) * sum(x(i))`
  * Subtract the mean from each input: `X = X - mean`
    * This makes your inputs centered around 0.
  * Get the variance of the training set: `variance = (1/m) * sum(x(i)^2)`
  * Normalize the variance: `X /= sqrt(variance)`

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/93e3badd-fc3b-4598-a009-3d2a1bfa53f4.png" width="700" />

* These steps should be applied to training, dev, and testing sets (but using mean and variance of the train set)
* Why normalize?
  * If we don't normalize the inputs our cost function will be deep and its shape will be inconsistent (elongated) then optimizing it will take a long time.
  * But if we normalize it the opposite will occur. The shape of the cost function will be consistent (look more symmetric like circle in 2D example) and we can use a larger learning rate alpha - the optimization will be 
faster

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/3749a2ad-60b8-48d3-892d-2c076bd75bbc.png" width="700" />

## Vanishing / Exploding Gradients
* The Vanishing / Exploding gradient phenomenon occurs when your derivatives become very small or very big
* To understand the problem, suppose that we have a deep neural network with a number of layers L, and all the activation functions are linear and each b = 0
Then:

`Y' = W[L]W[L-1].....W[2]W[1]X`

* Then, if we have 2 hidden units per layer and x1 = x2 = 1, we result in:
```
if W[l] = [1.5   0] 
          [0   1.5] (l != L because of different dimensions in the output layer)
Y' = W[L] [1.5  0]^(L-1) X = 1.5^L  # which will be very large (exponentially large)
          [0  1.5]
if W[l] = [0.5  0]
          [0  0.5]
Y' = W[L] [0.5  0]^(L-1) X = 0.5^L  # which will be very small (exponentially small)
          [0  0.5]
```

* The above example explains that the activations (and similarly derivatives) will decrease/increase exponentially as a function of number of layers.
* So If W > I (Identity matrix), the activation and gradients will explode
* And If W < I (Identity matrix) the activation and gradients will vanish
* Recently Microsoft trained 152 layers (ResNet)! which is a really big number. With such a deep neural network, if your activations or gradients increase or decrease exponentially as a function of L, then these values could get really big or really small. And this makes training difficult, especially if your gradients are exponentially smaller than L, then gradient descent will take tiny little steps. It will take a long time for gradient descent to learn anything
* There is a partial solution that doesn't completely solve this problem but it helps a lot - careful choice of how you initialize the weights (next video)

## Weight Initialization for Deep Networks
* A partial solution to the Vanishing / Exploding gradients in NN is a better or more careful choice of the random initialization of weights
* In a single neuron (Perceptron model): Z = w<sup>1</sup>x<sup>1</sup> + w<sup>2</sup>x<sup>2</sup> + ... + w<sup>n</sup>x<sup>n</sup>
  * So if N<sub>x</sub> is large, we want W's to be smaller to not explode the cost
* So it turns out that we need the variance which equals 1/n<sub>x</sub> to be the range of W's
* So let's say when we initialize W's like this (better to use with tanh activation):
  * `np.random.rand(shape) * np.sqrt(1/n[l-1])`

or variation of this (Bengio et al.):

`np.random.rand(shape) * np.sqrt(2/(n[l-1] + n[l]))`

* Setting initialization part inside sqrt to 2/n<sup>[l-1]</sup> for ReLU is better:
`np.random.rand(shape) * np.sqrt(2/n[l-1])`
* Number 1 or 2 in the numerator can also be a hyperparameter to tune (but shouldn't be the first hyperparameter to start with)
* This is one of the best partial solutions to the Vanishing / Exploding gradients (ReLU + Weight Initialization with variance), which will help gradients not to vanish/explode too quickly
* The initialization in this video is called "He Initialization / Xavier Initialization" and has been published in 2015 paper

## Numerical Approximation of Gradients
* There is a technique called gradient checking which tells you if your implementation of backpropagation is correct
* There's a numerical way to calculate the derivative:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/5779c7ea-366e-4070-841a-7e22655d8cf1.png" width="700" />

* Gradient checking approximates the gradients and is very helpful for finding any errors in your backpropagation implementation. But it's slower than gradient descent (so use only for debugging)
* Implementation of this is very simple
* Gradient checking:
   * First take `W[1],b[1],...,W[L],b[L]`, reshape and concatenate it into one big vector `theta`
   * The cost function will be `J(theta)`
   * Then take `dW[1],db[1],...,dW[L],db[L]`, reshape and concatenate it into one big vector `d_theta`
   * Algorithm:
    ```
    eps = 10^-7   # small number`
    for i in len(theta):
      d_theta_approx[i] = (J(theta[1],...,theta[i] + eps) -  J(theta[1],...,theta[i] - eps)) / 2*eps
    ```
* Finally we evaluate this formula `(||d_theta_approx - d_theta||) / (||d_theta_approx||+||d_theta||)` (|| - Euclidean vector norm) and check (with eps = 10<sup>-7</sup>):
   * if it is < 10^-7 - great, very likely the backpropagation implementation is correct
   * if around 10^-5 - can be OK, but need to inspect if there are no particularly big values in d_theta_approx - d_theta vector
   * if it is >= 10^-3 - bad, probably there is a bug in backpropagation implementation

## Gradient checking implementation notes
* Don't use the gradient checking algorithm at training time because it's very slow
* Use gradient checking only for debugging
* If algorithm fails grad check, look at components to try to identify the bug
* Don't forget to add `lamda/(2m) * sum(W[l])` to J if you are using L1 or L2 regularization
* Gradient checking doesn't work with dropout because J is not consistent i.e., some neurons are randomly deactivated during each iteration
  * You can first turn off dropout (set keep_prob = 1.0), run gradient checking and then turn on dropout again
* Run gradient checking at random initialization and train the network for a while maybe there's a bug which can be seen when w's and b's become larger (further from 0) and can't be seen on the first iteration (when w's and b's are very small)


