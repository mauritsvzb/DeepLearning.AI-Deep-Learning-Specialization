# Shallow Neural Networks
Learn to build a neural network with one hidden layer, using forward propagation and backpropagation.

## Neural Networks Overview
<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/0dc9f2fe-d5fe-4a49-aafa-fe823afe8858.png" width="700" />

The top graphical representation is the logistic regression treated as a simple neural network, whereas the bottom representation represents an actual neural network with 1 hidden layer. (Note that all input features are connected to each hidden node.) Essentially, this is two logistic regressions in sequence. Note the "<sup>[1]</sup>" and "<sup>[2]</sup>" which represent parameters belonging to the first and second layer of the neural network, respectively.

## Neural Network Representation
* We will define the NN that has one hidden layer
* NNs contain 1 input layer, 1 or more hidden layers, and 1 output layers
* 'Hidden layer' comes from the fact that we can't see that layers in the training set, only the input and output
* a<sup>[0]</sup> = x (= input layer)
* a<sup>[1]</sup> will represent the activation of the hidden neurons
* a<sup>[2]</sup> will represent the output layer
* This is a 2-layer NN: the input layer isn't counted

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/bf2e8ae2-5c41-46af-b61f-306f137a2eb9.png" width="700" />

## Computing a Neural Network's Output
Equations of hidden layers

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/62fae21c-741a-4415-910e-5a391163b2ec.png" width="700" />

* Number of hidden neurons = noOfHiddenNeurons = 4
* Number of input features = N<sub>x</sub> = 3

* Shapes of the variables
  * W<sup>[1]</sup> is the matrix of the first hidden layer; it has a shape of (noOfHiddenNeurons,nx)
  * b<sup>[1]</sup> is the matrix of the first hidden layer; it has a shape of (noOfHiddenNeurons,1)
  * z<sup>[1]</sup> is the result of the equation z<sup>[1]</sup> = W<sup>[1]</sup> * X + b<sup>[1]</sup>; it has a shape of (noOfHiddenNeurons,1)
  * a<sup>[1]</sup> is the result of the equation a<sup>[1]</sup> = sigmoid(z<sup>[1]</sup>); it has a shape of (noOfHiddenNeurons,1)
  * W<sup>[2]</sup> is the matrix of the second hidden layer; it has a shape of (1,noOfHiddenNeurons)
  * b<sup>[2]</sup> is the matrix of the second hidden layer; it has a shape of (1,1)
  * z<sup>[2]</sup> is the result of the equation z<sup>[2]</sup> = W<sup>[2]</sup> * a<sup>[1]</sup> + b<sup>[2]</sup>; it has a shape of (1,1)
  * a<sup>[2]</sup> is the result of the equation a<sup>[2]</sup> = sigmoid(z<sup>[2]</sup>); it has a shape of (1,1)

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/abbc2cf8-faa7-4067-b34b-64c2cd64866b.png" width="700" />

## Vectorizing across multiple examples
Pseudo code for forward propagation for the 2 layers NN (unvectorized):

```
for i = 1 to m:
  z[1](i) = W[1] * x[i] + b[1]         # shape of z[1](i) is (noOfHiddenNeurons,1)
  a[1](i) = sigmoid(z[1](i)            # shape of a[1](i) is (noOfHiddenNeurons,1)
  z[2](i) = W[2] * a[1](i) + b[2]      # shape of z[2[(i) is (1,1)
  a[2](i) = sigmoid(z[2](i)            # shape of a[2](i) is (1,1)
```
<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/464cea39-903c-4a27-a2dc-0b3f83b9bdb5.png" width="700" />

Pseudo code for forward propagation for the 2 layers NN (vectorized):
* Lets say we have X on shape (Nx,m). So the new pseudo code:

```
Z[1] = W[1] * X + b[1]        # shape of Z1 (noOfHiddenNeurons,m)
A[1] = sigmoid(Z[1])          # shape of A1 (noOfHiddenNeurons,m)
Z[2] = W[2] * A[1] + b[2]     # shape of Z[2] is (1,m)
A[2] = sigmoid(Z[2])          # shape of  2 is (1,m)
```

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/fb0cee6d-e136-41a7-aaf4-79b4622c4490.png" width="700" />

Note: in the  X matrix, the vertical dimension corresponds to different input features (= different nodes of the input layer)

* If you noticed, m is always the number of columns
In the last example we can called X = A<sup>[0]</sup>. So the previous step can be rewritten as:
```
Z[1] = W[1]A[0] + b[1]         # shape of Z[1] (noOfHiddenNeurons,m)
A[1] = sigmoid(Z[1])           # shape of A[1] (noOfHiddenNeurons,m)
Z[2] = W[2] * A[1] + b[2]      # shape of Z[2] is (1,m)
A[2] = sigmoid(Z[2])           # shape of A[2] is (1,m)
```

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/99adb59d-bec1-4908-ae09-0a1b24ad002c.png" width="700" />

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/02846c2a-0810-44c3-98b7-8bdeb929496a.png" width="700" />

The code on the bottom right is the vectorized version of the code on the top right.

## Activation functions
* So far we have used the sigmoid, but in some cases other activation functions may be a lot better
* Sigmoid can lead us to gradient decent problem when the updates are very low
* Sigmoid activation function range is [0,1] `A = 1 / (1 + np.exp(-z)) # Where z is the input matrix`
* Tangent activation (hyperbolic: tan h) function range is [-1,1] (Shifted version of sigmoid function)
  * In NumPy we can implement Tan h(z) using one of these methods: 
    * `A = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))  # Where z is the input matrix` 
    * `A = np.tanh(z)                                           # Where z is the input matrix`
* It turns out that the tan h activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer
    *  Only real good use of the sigmoid function is for the output layer when using classification where ŷ is between 0 and 1. Otherwise, use the tan h(z) function
* Disadvantage of the sigmoid and tan h function is that if the input is very small or very high, the slope will be near zero which can slow down gradient descent (i.e., cause problems)

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/c079da55-ffce-42ed-acd1-7e21ae15b9b4.png" width="700" />

* One of the popular activation functions that solved the slow gradient decent is the RELU function:
    * `RELU = max(0,z) # so if z is negative the slope is 0 and if z is positive the slope remains linear`
* So here is some basic rule for choosing activation functions, if your classification is between 0 and 1, use the output activation as sigmoid
    * RELU should be the default choice otherwise
* Leaky RELU activation function is different form of RELU in that if the input is negative the slope will be slightly shallow. It works as RELU but most people uses RELU. 
    * `Leaky_RELU = max(0.01z,z)  #the 0.01 can be a parameter for your algorithm`

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/2dc0781b-59a1-45e8-a16c-0c7e1c636cad.png" width="700" />

In NN you will decide a lot of choices like:
* No. hidden layers
* No. neurons in each hidden layer
* Learning rate (the most important parameter)
* Activation functions

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/2161e165-6cd6-4422-8878-2b6eac146788.png" width="700" />

* And others...
* It turns out there are no guide lines for choosing activation functions: You should try all to see which ones gives best results

## Why do you need non-linear activation functions?
* If we removed the activation function from our algorithm then we essentially have a linear activation function that is used
* Linear activation function in the hidden layer(s) will output linear activations
* Therefore, however many hidden layers you add, the activation will be always linear like logistic regression (So its useless in a lot of complex problems)
* You might use linear activation function in one place - in the output layer if the output is real numbers (regression problem). But even in this case if the output value is non-negative you could use RELU instead.

## Derivatives of activation functions
* Derivation of Sigmoid activation function:

```
g(z)  = 1 / (1 + np.exp(-z))
g'(z) = (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
g'(z) = g(z) * (1 - g(z))
```

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/eeda5232-a175-4ba4-be33-7aa8b73592d2.png" width="700" />

* Derivation of Tanh activation function:
```
g(z)  = (e^z - e^-z) / (e^z + e^-z)
g'(z) = 1 - np.tanh(z)^2 = 1 - g(z)^2
```

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/f6ad00e2-58f0-455c-86b5-b6326f4af248.png" width="700" />

* Derivation of RELU activation function:

```
g(z)  = np.maximum(0,z)
g'(z) = { 0  if z < 0
          1  if z >= 0  }
```
* Derivation of leaky RELU activation function:

```
g(z)  = np.maximum(0.01 * z, z)
g'(z) = { 0.01  if z < 0
          1     if z >= 0   }
```

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/f4982ecc-3d45-4040-87d7-58fc0a7ba8c6.png" width="700" />

## Gradient descent for Neural Networks
