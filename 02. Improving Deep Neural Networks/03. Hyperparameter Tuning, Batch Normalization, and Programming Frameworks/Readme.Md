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

