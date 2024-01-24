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
* a<sup>[1]</sup> will represent the activation of the hidden neurons i.e., the parameter values that are passed to the next layer
* a<sup>[2]</sup> will represent the output layer
* This is a 2-layer NN: the input layer isn't counted

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/bf2e8ae2-5c41-46af-b61f-306f137a2eb9.png" width="700" />

## Computing a Neural Network's Output

