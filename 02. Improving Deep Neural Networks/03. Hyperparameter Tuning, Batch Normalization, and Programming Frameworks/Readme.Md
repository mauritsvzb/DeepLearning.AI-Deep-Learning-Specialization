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
  * Try random values: don't use a grid.
* You can use Coarse to fine sampling scheme:

* When you find some hyperparameters values that give you a better performance, zoom into a smaller region around these values and sample more densely within this parameter space
* These methods can be automated

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/3e4f3076-e03e-40a2-b1d9-37a5cf82b75f.png" width="700" />

## Using an appropriate scale to pick hyperparameters

