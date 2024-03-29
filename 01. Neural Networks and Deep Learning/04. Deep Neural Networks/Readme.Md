# Deep Neural Networks
## Goal and Learning Objectives
Analyze the key computations underlying deep learning, then use them to build and train deep neural networks for computer vision tasks

* Describe the successive block structure of a deep neural network
* Build a deep L-layer neural network
* Analyze matrix and vector dimensions to check neural network implementations
* Use a cache to pass information from forward to back propagation
* Explain the role of hyperparameters in deep learning
* Build a 2-layer neural network

## Deep L-layer Neural Network
* Shallow NN is a NN with one or two layers
* Deep NN is a NN with three or more layers

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/a7285b13-1264-40b1-9686-adbe2239e162.png" width="700" />

Notations:
* L = number of layers in a NN
* n<sup>[l]</sup> = number of neurons in a specific layer l.
* n<sup>[0]</sup> = n<sub>x</sub> = number of neurons input layer. 
* n<sup>[L]</sup> = number of neurons in output layer.
* g<sup>[l]</sup> is the activation function.
* a<sup>[l]</sup> = g<sup>[l]</sup>(z<sup>[l]</sup>)
* w<sup>[l]</sup> weights is used for z<sup>[l]</sup>
x = a<sup>[0]</sup>, 
* a[L] = ŷ

These were the notation we will use for deep neural network.
So we have:
* A vector n of shape (1, NoOfLayers+1)
* A vector g of shape (1, NoOfLayers)
* A list of different shapes w based on the number of neurons on the previous and the current layer.
* A list of different shapes b based on the number of neurons on the current layer.

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/527f79fc-ba25-4020-8f1a-63f242cf5395.png" width="700" />

## Forward Propagation in a Deep Network

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/3472c20d-ff9d-4c10-b1bd-3b1cc0bf6adf.png" width="700" />

* We can't compute forward propagation for all layers without a for loop so its OK to have a for loop here
* It's important to have the dimensions of the matrices right

## Gettiing Your Matrix Dimensions Right
* The best way to debug your matrices dimensions is by a pencil and paper
* Dimension of W is (n<sup>[l]</sup>, n<sup>[l-1]</sup>)
* Dimension of b is (n<sup>[l]</sup>, 1)
* dw has the same shape as W, while db is the same shape as b

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/7bc07611-e844-4595-8f6f-a58777ed5e03.png" width="700" />

* Dimensions of Z<sup>[l]</sup>, A<sup>[l]</sup>, dZ<sup>[l]</sup>, and dA<sup>[l]</sup> is (n<sup>[l]</sup>, m)

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/8d07202f-9a2f-4e64-bb5e-eb62d14552c1.png" width="700" />

## Why Deep representations?
* Deep NNs make relations from data by moving from simple to more complex as data moves through the NN from left to right. In each layer the NN tries to make a relation with the previous layer. 
   * E.g., face recognition application:
     * Image ==> Edges ==> Face parts ==> Faces ==> desired face
   * Audio recognition application:
     * Audio ==> Low level sound features like (sss,bb) ==> Phonemes ==> Words ==> Sentences
   * Neural Researchers think that deep neural networks "think" like brains (simple ==> complex) 

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/ff5eeb80-3ead-48e9-b880-8d13f9f92387.png" width="700" />

* Branding

* Advice: When starting on an application don't start directly by dozens of hidden layers. Try the simplest solutions (e.g. Logistic Regression), then gradually increase complexity until satisfied

## Building Blocks of Deep Neural Networks
* Forward and back propagation for a layer l:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/bb94c552-d039-4ab9-aa94-5bfc225a2f65.png" width="700" />

* Deep NN blocks:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/adbacb9c-755d-47cd-b29b-6d139e458a56.png" width="700" />

## Forward and Backward Propagation
* Pseudo code for forward propagation for layer l:

```
Input  A[l-1]
Z[l] = W[l]A[l-1] + b[l]
A[l] = g[l](Z[l])
Output A[l], cache(Z[l])
```

Pseudo code for back propagation for layer l:
```
Input da[l], Caches
dZ[l] = dA[l] * g'[l](Z[l])
dW[l] = (dZ[l]A[l-1].T) / m
db[l] = sum(dZ[l])/m                # Dont forget axis=1, keepdims=True
dA[l-1] = w[l].T * dZ[l]            # The multiplication here are a dot product.
Output dA[l-1], dW[l], db[l]
```

* If we have used our loss function then:
```
dA[L] = (-(y/a) + ((1-y)/(1-a)))
```

## Parameters vs Hyperparameters
* Main parameters of the NN are W and b
* Hyperparameters (parameters that indirectly control the values of parameters W and b) are like:
  * Learning rate
  * Number of iteration
  * Number of hidden layers L
  * Number of hidden units n
  * Choice of activation functions
* You have to experiment with with values of hyperparameters
