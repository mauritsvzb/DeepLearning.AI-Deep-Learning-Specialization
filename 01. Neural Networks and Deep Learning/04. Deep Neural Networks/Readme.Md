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

* We will use the notation L to denote the number of layers in a NN
* n[l] is the number of neurons in a specific layer l.
n[0] denotes the number of neurons input layer. n[L] denotes the number of neurons in output layer.
g[l] is the activation function.
a[l] = g[l](z[l])
w[l] weights is used for z[l]
x = a[0], a[l] = y'
These were the notation we will use for deep neural network.
So we have:
A vector n of shape (1, NoOfLayers+1)
A vector g of shape (1, NoOfLayers)
A list of different shapes w based on the number of neurons on the previous and the current layer.
A list of different shapes b based on the number of neurons on the current layer.
