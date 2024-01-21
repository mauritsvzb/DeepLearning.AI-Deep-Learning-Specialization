# Introduction to deep learning
## What is a neural network (NN)?
* Single neuron = linear regression without applying activation(perceptron)
* A single neuron will calculate weighted sum of input(W<sup>T</sup> * X) and then we can set a threshold to predict output in a perceptron. If weighted sum of input crosses the threshold, perceptron fires and if not then perceptron doesn't predict
*  Perceptron can take real values input or boolean values
* Actually, when w * x + b = 0, the perceptron outputs 0
* Disadvantage of perceptron is that it only outputs binary values and if we try to give small change in weight and bias then perceptron can flip the output. We need some system which can modify the output slightly according to small change in weight and bias. Enter sigmoid function.
* If we change perceptron with a sigmoid function, then we can make a slight change in output e.g. output in perceptron = 0, you slightly changed weight and bias, output becomes = 1 but actual output is 0.7. In case of sigmoid, output1 = 0, slight change in weight and bias, output = 0.7.
* If we apply sigmoid activation function then a single neuron will act as Logistic Regression.
* we can understand difference between perceptron and sigmoid function by looking at sigmoid function graph.
* Simple NN graph
  * <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/9903d143-3ef8-45ca-86ed-7eb60b42c2e6.png" width="600" />
* ReLU stands for REctified Linear Unit, and is the most popular activation function right now that makes deep NNs train faster now.
* Hidden layers predict connection among inputs and outputs automatically, thats what deep learning is good at.
* Deep NN consists of more hidden layers (deeper layers)
  * <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/be89c057-92a0-4421-bbe5-744138282a44.png" width="600" />
* Each Input will be connected to the hidden layer and the NN will decide the connections.
* Supervised learning means we have the input(s) X and output(s) Y and we need to get the function that maps X to Y.

## Supervised learning with neural networks
* Different types of neural networks for supervised learning which includes:
 * Convolutional neural networks or CNN (Useful in computer vision)
 * Recurrent neural networks or RNN(Useful in Speech recognition or NLP)
 * Standard NN (Useful for Structured data)
 * Hybrid/custom NN or a Collection of NNs types
* Structured data is like the databases and tables.
* Unstructured data is like images, video, audio, and text.
* Structured data gives more money because companies relies on prediction on its big data.

## Why is deep learning taking off?
Deep learning is taking off for 3 reasons:
1. Data:
   * Using this image we can conclude:
    * <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/5466845b-a151-4427-a7a1-feccd8eff7ef.png" width="700" />
   * For small data NN can perform as Linear regression or SVM (Support vector machine)
   * For big data a small NN is better that SVM
   * For big data a big NN is better that a medium NN is better that small NN.
2. Computation:
   * GPUs
   * Powerful CPUs
   * Distributed computing
   * ASICs
3. Algorithm
   * Creative algorithms has appeared that changed the way NN works
   * For example using RELU function is so much better than using SIGMOID function in training a NN because it helps with the vanishing gradient problem

# Neural Networks Basics

