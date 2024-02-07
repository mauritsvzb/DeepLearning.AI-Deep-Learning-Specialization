# ML Strategy I

## Why ML Strategy
* You have a lot of ideas for how to improve the accuracy of your deep learning system:
  * Collect more data
  * Collect more diverse training set
  * Train algorithm longer with gradient descent
  * Try different optimization algorithm (e.g. Adam)
  * Try bigger network
  * Try smaller network
  * Try dropout
  * Add L2 regularization
  * Change network architecture (activation functions, # of hidden units, etc.)
* This course will give you some strategies to help analyze your problem to go in a direction that will help you get better results.

## Orthogonalization
* Some deep learning developers know exactly what hyperparameter to tune in order to try to achieve one effect (called orthogonalization)
* In orthogonalization, you have some controls, but each control does one specific task but doesn't affect other controls
* For a supervised learning system to do well, you usually need to tune the knobs of your system to make sure that four things hold true - chain of assumptions in machine learning:
  * You'll have to fit training set well on cost function (near human level performance if possible)
    * If it's not achieved you could try bigger network, another optimization algorithm (like Adam)...
  * Fit dev set well on cost function
    * If its not achieved you could try regularization, bigger training set...
  * Fit test set well on cost function
    * If its not achieved you could try bigger dev. set...
  * Performs well in real world
    * If its not achieved you could try change dev. set, change cost function...
* Generally, 'early stopping' is not very orthogonal, as it influences multiple parameters at once e.g., the size of the NN set and regularization:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/90ef567d-76c9-45fd-969c-081092eaa325.png" width="700" />

## Single Number Evaluation Metric

