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
* In mini-batch algorithm, the cost won't go down with each step as it does with batch gradient descent. It could contain some ups and downs but generally the trend is downward

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/bd8e368f-999f-4012-8fe6-b3ec3169e644.png" width="700" />

* Mini-batch size:
   * (mini batch size = m) ==> Batch gradient descent (X{1}, Y{1}) = (X,Y)
   * (mini batch size = 1) ==> Stochastic gradient descent (SGD) (X{1}, (Y{1} = (x1, y1)
   * (mini batch size = between 1 and m) ==> Mini-batch gradient descent
* Batch gradient descent:
   * too long per iteration (epoch)
* Stochastic gradient descent:
   *  extremely noisy regarding cost minimization (can be reduced by using smaller learning rate)
   *  won't ever converge (reach the minimum cost)
   * lose speedup from vectorization
* Mini-batch gradient descent:
   * faster learning:
      * you have the vectorization advantage
      * make progress without waiting to process the entire training set
   * doesn't always exactly converge (oscelates in a very small region, but you can reduce learning rate)

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/41108c1e-519a-491d-aa92-cb804b44fd5e.png" width="700" />

* Guidelines for choosing mini-batch size:
   * If small training set (< 2000 examples) - use batch gradient descent
   * It has to be a power of 2 (because of the way computer memory is layed out and accessed, sometimes your code runs faster if your mini-batch size is a power of 2):
      * 64, 128, 256, 512, 1024, ...
   * Make sure that mini-batch fits in CPU/GPU memory
* Mini-batch size is a hyperparameter

## Exponentially weighted averages
* There are optimization algorithms that are better than gradient descent, but for that you should first learn about Exponentially Weighted Averages
* If we have data like the temperature of day through the year it could be like this:
```
t(1) = 40
t(2) = 49
t(3) = 45
...
t(180) = 60
...
```
* If we plot this data we will find it a bit noisy
* Now lets compute the Exponentially weighted averages:
```
V0 = 0
V1 = 0.9 * V0 + 0.1 * t(1) = 4    # 0.9 and 0.1 are hyperparameters
V2 = 0.9 * V1 + 0.1 * t(2) = 8.5
V3 = 0.9 * V2 + 0.1 * t(3) = 12.15
...
```
* General equation
`V(t) = beta * v(t-1) + (1-beta) * theta(t)`
* If we plot this it will represent averages over ~ (1 / (1 - beta)) entries:
   * beta = 0.9 will average last 10 entries
   * beta = 0.98 will average last 50 entries
   * beta = 0.5 will average last 2 entries
* Best beta average for our case is somewhere between 0.9 and 0.98
* Intuition: The reason why exponentially weighted averages are useful for further optimizing gradient descent algorithm is that it can give different weights to recent data points (theta) based on value of beta. If beta is high (around 0.9), it smoothens out the averages of skewed data points (oscillations w.r.t. Gradient descent terminology). So this reduces oscillations in gradient descent and hence makes faster and smoother path towards minima.

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/aa24bdeb-8f32-4d94-b74c-4ae5d1a4bdf5.png" width="700" />

## Understanding Exponentially Weighted Averages
* Intuitions:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/598017bf-cc8d-4241-a024-276a1a08e32a.png" width="700" />

* We can implement this algorithm with more accurate results using a moving window. But the code is more efficient and faster using the exponentially weighted averages algorithm.
* The algorithm is very simple:
```
v = 0
Repeat
{
  Get theta(t)
  v = beta * v + (1-beta) * theta(t)
}
```

## Bias correction in exponentially weighted averages
* The bias correction helps make the exponentially weighted averages more accurate
* Because v(0) = 0, the bias of the weighted averages is shifted and the accuracy suffers at the start
* To solve the bias issue we have to use this equation:
   * `v(t) = (beta * v(t-1) + (1-beta) * theta(t)) / (1 - beta^t)`
* As t becomes larger, the (1 - beta^t) becomes close to 1
* Note that bias correction usingg exponentially weighted averages is not that common: it's easier to treat the initial period as a 'warmup' rather than part of the analysis

## Gradient descent with momentum
* The momentum algorithm almost always works faster than standard gradient descent
* The simple idea is to calculate the exponentially weighted averages for your gradients and then update your weights with these new values

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/c4a7ad51-f600-4f4f-a03a-bafff151703e.png" width="700" />

* Pseudo code:
```
vdW = 0, vdb = 0
on iteration t:
  # can be mini-batch or batch gradient descent
  compute dw, db on current mini-batch                
      
  vdW = beta * vdW + (1 - beta) * dW
  vdb = beta * vdb + (1 - beta) * db
  W = W - learning_rate * vdW
  b = b - learning_rate * vdb
```
* Momentum helps the cost function to go to the minimum point in a more fast and consistent way
* beta is another hyperparameter. beta = 0.9 is very common and works very well in most cases
* In practice people don't bother implementing bias correction

## RMSprop
* Stands for Root mean square prop
* This algorithm speeds up the gradient descent
```
Pseudo code:
sdW = 0, sdb = 0
on iteration t:
  # can be mini-batch or batch gradient descent
  compute dw, db on current mini-batch
  
  sdW = (beta * sdW) + (1 - beta) * dW^2  # squaring is element-wise
  sdb = (beta * sdb) + (1 - beta) * db^2  # squaring is element-wise
  W = W - learning_rate * dW / sqrt(sdW)
  b = B - learning_rate * db / sqrt(sdb)
```

* RMSprop will make the cost function move slower in the vertical direction and faster on the horizontal direction in the following example

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/7fd1e674-949d-4fd7-a89f-350210486db6.png" width="700" />

* Ensure that `sdW` is not zero by adding a small value epsilon (e.g. epsilon = 10^-8) to it:
   `W = W - learning_rate * dW / (sqrt(sdW) + epsilon)`
* With RMSprop you can increase your learning rate

## Adam optimization algorithm
