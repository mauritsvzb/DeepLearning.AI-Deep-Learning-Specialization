# Goals and Learning Objectives
Explore TensorFlow, a deep learning framework that allows you to build neural networks quickly and easily, then train a neural network on a TensorFlow dataset
* Master the process of hyperparameter tuning
* Describe softmax classification for multiple classes
* Apply batch normalization to make your neural network more robust
* Build a neural network in TensorFlow and train it on a TensorFlow dataset
* Describe the purpose and operation of GradientTape
* Use tf.Variable to modify the state of a variable
* Apply TensorFlow decorators to speed up code
* Explain the difference between a variable and a constant

# Hyperparameter Tuning, Batch Normalization, and Programming Frameworks
## Tuning Process

* Hyperparameter tuning is important to optimize NN performance
* Hyperparameters importance ranked (according to Andrew Ng (might differ per case)):
  * most:
    * Learning rate
  * medium:
    * Momentum beta (~ 0.9)
    * Mini-batch size
    * No. of hidden units
  * least:   
    * No. of layers
    * Learning rate decay
    * (Regularization lambda)
    * (Activation functions)
  * not:
    * Adam beta1, beta2 & epsilon ~0.9, 0.999, 10<sup>-8</sup>

* It's challenging to decide which hyperparameter is the most important in a problem, as it depends a lot on your problem
* One of the ways to tune is to sample a grid with N hyperparameter settings and then try all settings combinations on your problem
  * Try random values: don't use a grid
* You can use Coarse to fine sampling scheme:

* When you find some hyperparameters values that give you a better performance, zoom into a smaller region around these values and sample more densely within this parameter space
* These methods can be automated

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/3e4f3076-e03e-40a2-b1d9-37a5cf82b75f.png" width="700" />

## Using an appropriate scale to pick hyperparameters
* Let's say you have a specific range for a hyperparameter from "a" to "b". It's better to search using the logarithmic scale rather then using a linear scale (log scale makes more use of the parameter space at boundaries):
  * Calculate: `a_log = log(a)`  # e.g. `a = 0.0001` then `a_log = -4`
  * Calculate: `b_log = log(b)`  # e.g. `b = 1`  then `b_log = 0`
  * Then:
    * ```
       r = (a_log - b_log) * np.random.rand() + b_log
       # In the example the range would be from [-4, 0] because rand range [0, 1]
       result = 10^r
      ```
  * It uniformly samples values in log scale from [a, b]
* If we want to use the last method on exploring on the "momentum beta":
  * Beta best range is from 0.9 to 0.999.
  * You should search for `1 - beta` in range 0.001 to 0.1 (1 - 0.9 and 1 - 0.999) and the use `a = 0.001` and `b = 0.1`. Then:
    * ```
      a_log = -3
      b_log = -1
      r = (a_log - b_log) * np.random.rand() + b_log
      beta = 1 - 10^r   # because 1 - beta = 10^r
      ```

## Hyperparameters tuning in practice: Pandas vs. Caviar
* Intuitions about hyperparameter settings from one application area may or may not transfer to a different one
* If you don't have much computational resources you can use the "babysitting model":
  * Day 0 you might initialize your parameter as random and then start training
  * Then you watch your learning curve gradually decrease over the day
  * And each day you nudge your parameters a little during training.
  * Called 'panda approach'
* If you have enough computational resources, you can run some models in parallel and at the end of the day(s) you check the results
  * Called 'caviar approach'

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/6a467010-1932-4a08-86ea-a747667f2f43.png" width="700" />

## Normalizing activations in a network
* With the rise of deep learning, one of the most important ideas has been an algorithm called 'batch normalization'
* Batch Normalization speeds up learning
* Before, we normalized input by subtracting the mean and dividing by square root of variance; this helped a lot for the shape of the cost function and for reaching the minimum point faster
* The question is: <i>for any hidden layer, can we normalize A<sup>[l]</sup> to train W<sup>[l+1]</sup>, b<sup>[l+1]</sup> faster</i>? This is what batch normalization is about
* There are some debates in the deep learning literature about whether you should normalize values before applying the activation function Z<sup>[l]</sup> or after applying the activation function A[l]. In practice, normalizing Z<si[>[l]</sup> is done much more often and that is what Andrew Ng presents
* Algorithm:
  * Given `Z[l] = [z(1), ..., z(m)]`, i = 1 to m (for each input)
  * Compute `mean = 1/m * sum(z[i])`
  * Compute `variance = 1/m * sum((z[i] - mean)^2)`
  * Then `Z_norm[i] = (z[i] - mean) / np.sqrt(variance + epsilon)` (add epsilon for numerical stability if variance = 0)
    * Forcing the inputs to a distribution with zero mean and variance of 1.
  * Then `Z_tilde[i] = gamma * Z_norm[i] + beta`
    * To make inputs belong to other distribution (with other mean and variance).
    * gamma and beta are learnable parameters of the model.
    * Making the NN learn the distribution of the outputs.
  * Note: if `gamma = sqrt(variance + epsilon)` and `beta = mean then Z_tilde[i] = z[i]`

## Fitting Batch Normalization into a neural network
* Using batch norm in 3 hidden layers NN: 

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/06d18cc8-05e2-4eb2-80a7-2681bea2a8f3.png" width="900" />

* Our NN parameters will be:
  * W<sup>[1]</sup>, b<sup>[1]</sup>, ..., W<sup>[L]</sup>, b<sup>[L]</sup>, beta<sup>[1]</sup>, gamma<sup>[1]</sup>, ..., beta<sup>[L]</sup>, gamma<sup>[L]</sup>
  * beta<sup>[1]</sup>, gamma<sup>[1]</sup>, ..., beta<sup>[L]</sup>, gamma<sup>[L]</sup> are updated using any optimization algorithms (like GD, RMSprop, Adam)
* If you are using a deep learning framework, you won't have to implement batch norm yourself:
  * E.g., in Tensorflow you can add this line: `tf.nn.batch-normalization()`

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/50364d22-44e8-48dd-895c-0001a1674383.png" width="700" />

* Batch normalization is usually applied with mini-batches
* If we are using batch normalization parameters b<sup>[1]</sup>, ..., b<sup>[L]</sup> doesn't count because they will be eliminated after mean subtraction step, so:
  * ```
    Z[l] = W[l]A[l-1] + b[l] => Z[l] = W[l]A[l-1]
    1Z_norm[l] = ...
    Z_tilde[l] = gamma[l] * Z_norm[l] + beta[l]
    ```
  * Taking the mean of a constant b<sup>[l]</sup> will eliminate the b<sup>[l]</sup>
* So if you are using batch normalization, you can remove b<sup>[l]</sup> or make it always zero
* So the parameters will be W<sup>[l]</sup>, beta<sup>[l]</sup>, and alpha<sup>[l]</sup>
* Shapes:
  ```
  Z[l]       - (n[l], m)
  beta[l]    - (n[l], m)
  gamma[l]   - (n[l], m)
  ```

## Why does Batch normalization work?
* The first reason is the same reason as why we normalize X
* The second reason is that batch normalization reduces the problem of input values changing (shifting)

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/1c0b7d12-b64c-4b70-b43e-a6eda2dbbe5d.png" width="700" />

* Why is this a problem with NN?

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/ab8bc300-23ed-4d55-9321-571f0b695527.png" width="700" />

* Batch normalization does some regularization:
  * Each mini batch is scaled by the mean/variance computed of that mini-batch.
  * This adds some noise to the values Z[l] within that mini batch. So similar to dropout it adds some noise to each hidden layer's activations.
  * This has a slight regularization effect: 
  * Using bigger size of the mini-batch you are reducing noise and therefore regularization effect.
* Don't rely on batch normalization as a regularization. It's intended for normalization of hidden units, activations and therefore speeding up learning. For regularization use other regularization techniques (L2 or dropout).

## Batch Norm at Test Time
* When we train a NN with Batch normalization, we compute the mean and the variance of the mini-batch
* During testing, we might need to process examples one at a time; the mean and the variance of one example won't make sense
* We have to compute an estimated value of the mean and variance to use it for testing
* We can use the weighted average across the mini-batches
* We will use the estimated mean and variance to test
* This method is also sometimes called "Running average"
* In practice most often you will use a deep learning framework and it will contain some default implementation of doing such a thing

## Softmax Regression
* In every example we have used so far we were talking about binary classification
* There is a generalization of logistic regression called <i>Softmax regression</i> that is used for multiclass classification/regression
* For example if we are classifying by classes dog, cat, baby chick and none of that e.g., dog: class = 1, cat: class = 2, baby chick: class = 3, none: class = 0 to represent a dog vector y = [0 1 0 0], to represent a cat vector y = [0 0 1 0], to represent a baby chick vector y = [0 0 0 1], or to represent a none vector y = [1 0 0 0]
* Notations:
  * C = no. of classes
  * Range of classes is (0, ..., C-1) e.g., (0, ..., 3)
  * In output layer N<sub>y</sub> = C
* Each of C values in the output layer will contain a probability of the example to belong to each of the classes
* In the last layer we will have to activate the Softmax activation function instead of the sigmoid activation
  * Softmax activation equations:
    ```
    t = e^(Z[L])                      # shape(C, m)
    A[L] = e^(Z[L]) / sum(t)          # shape(C, m), sum(t) - sum of t's for each example (shape (1, m))
    ````
<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/303f476f-9c07-4775-a027-8740ba6bf7d2.png" width="700" />

## Training a Softmax classifier
* There's an activation which is called hard max, which gets 1 for the maximum value and zeros for the others; if you are using NumPy, its np.max over the vertical axis
* The Softmax name came from softening the values and not harding them like hard max
* Softmax is a generalization of logistic activation function to C classes. If C = 2 softmax reduces to logistic regression
* The loss function used with softmax:
  * `L(y, y_hat) = - sum(y[j] * log(y_hat[j])) # j = 0 to C-1`
* The cost function used with softmax:
  * `J(w[1], b[1], ...) = - 1 / m * (sum(L(y[i], y_hat[i]))) # i = 0 to m`
* Back propagation with softmax:
  * `dZ[L] = Y_hat - Y`
* Note that programming frameworks like TensorFlow will implement the back propagation for you (note that in order for the back prop to be correct, the forward prop needs to be correct too)

## Deep Learning Frameworks
* It's impractical to implement everything from scratch; the numpy manual implementations provided a good mathematical basis on how NNs work
* Here are some of the many leading deep learning frameworks:
  * Caffe/ Caffe2
  * CNTK
  * DL4j
  * Keras
  * Lasagne
  * mxnet
  * PaddlePaddle
  * TensorFlow
  * Theano
  * Torch/Pytorch
* These frameworks are getting better month by month. Comparison between them can be found [here](https://en.wikipedia.org/wiki/Comparison_of_deep_learning_software)
* How to choose deep learning framework:
  * Ease of programming (development and deployment)
  * Running speed
  * Truly open (open source with good governance): some open-source packages become closed-source over time
* Programming frameworks can not only shorten your coding time but sometimes also perform optimizations that speed up your code

## TensorFlow
* Lets see how to implement a minimization function:
  * Example function: `J(w) = w^2 - 10w + 25`
  * The result should be `w = 5` as the function is `(w-5)^2 = 0`
  * Code v.1:
    ```python
    import numpy as np
    import tensorflow as tf
    

    w = tf.Variable(0, dtype=tf.float32)                 # creating a variable w
    cost = tf.add(tf.add(w**2, tf.multiply(-10.0, w)), 25.0)        # can be written as this - cost = w**2 - 10*w + 25
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)
    session.run(w)    # Runs the definition of w, if you print this it will print zero
    session.run(train)

    print("W after one iteration:", session.run(w))

    for i in range(1000):
      session.run(train)
    
    print("W after 1000 iterations:", session.run(w))
    ```

  * Code v.2 (we feed the inputs to the algorithm through coefficients):
    ```python
    import numpy as np
    import tensorflow as tf
    
    
    coefficients = np.array([[1.], [-10.], [25.]])
    
    x = tf.placeholder(tf.float32, [3, 1])
    w = tf.Variable(0, dtype=tf.float32)                 # Creating a variable w
    cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]
    
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    
    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)
    session.run(w)    # Runs the definition of w, if you print this it will print zero
    session.run(train, feed_dict={x: coefficients})
    
    print("W after one iteration:", session.run(w))
    
    for i in range(1000):
      session.run(train, feed_dict={x: coefficients})
    
    print("W after 1000 iterations:", session.run(w))
    ```
* In TensorFlow you implement only the forward propagation and TensorFlow will do the backpropagation by itself.
* In TensorFlow a placeholder is a variable you can assign a value to later.
* If you are using a mini-batch training you should change the feed_dict={x: coefficients} to the current mini-batch data.
* Almost all TensorFlow programs use this:
    ```python
       with tf.Session() as session:       # better for cleaning up in case of error/exception
       session.run(init)
       session.run(w)
    ```
* In deep learning frameworks there are a lot of things that you can do with one line of code like changing the optimizer. Side notes:
* Writing and running programs in TensorFlow has the following steps:
  1. Create Tensors (variables) that are not yet executed/evaluated
  2. Write operations between those Tensors
  3. Initialize your Tensors
  4. Create a Session
  5. Run the Session. This will run the operations you'd written above.
* Instead of needing to write code to compute the cost function we know, we can use this line in TensorFlow : tf.nn.sigmoid_cross_entropy_with_logits(logits = ...,  labels = ...)
* To initialize weights in NN using TensorFlow use:
    ```python
    W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    
    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
    ```
* For 3-layer NN, it is important to note that the forward propagation stops at Z3. The reason is that in TensorFlow the last linear layer output is given as input to the function computing the loss. Therefore, you don't need A3!
* To reset the graph use `tf.reset_default_graph()`
    



