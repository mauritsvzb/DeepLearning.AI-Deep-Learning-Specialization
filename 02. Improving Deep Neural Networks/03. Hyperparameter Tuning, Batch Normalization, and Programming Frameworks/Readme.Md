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


