# Optimization Algorithms

## Mini-batch Gradient Descent
* Training NN with a large data is slow. So to find an optimization algorithm that runs faster is a good idea.
* Suppose we have m = 5 million examples. To train these data it will take a huge processing time for one step
    * Because 5 million won't fit in the memory at once we need other processing to make such a thing
* It turns out you can make a faster algorithm to make gradient descent process some of your items even before you finish the 5 million items
* Suppose we have split m to mini batches of size 1000.
    * `X{1} = 0    ...  1000`
    * `X{2} = 1001 ...  2000`
    * `...`
    * 1X{5000} = ...`
* We similarly split `X` & `Y`
* So the definition of mini batches ==> t: X<sup>{t}</sup>, Y<sup>{t}</sup>
* In Batch gradient descent we run the gradient descent on the whole dataset
* In contrast, in Mini-Batch gradient descent we run the gradient descent on a baby dataset (Batch gradient descent)
* Mini-Batch algorithm pseudo code:
```
for t = 1:No_of_batches                         # this whole loop is also called 1 epocch
  AL, caches = forward_prop(X{t}, Y{t})
  cost = compute_cost(AL, Y{t})
  grads = backward_prop(AL, caches)
  update_parameters(grads)
```
* Epoch is a single pass through the training set (going through all the batch gradient descent steps)
* The code inside an epoch should be vectorized
* Mini-batch gradient descent works much faster in the large datasets

## Understanding mini-batch gradient descent
