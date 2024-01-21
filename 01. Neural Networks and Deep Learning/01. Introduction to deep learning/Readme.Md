# What is a neural network (NN)?
* Single neuron = linear regression without applying activation(perceptron)
* A single neuron will calculate weighted sum of input(W<sup>T * X) and then we can set a threshold to predict output in a perceptron. If weighted sum of input crosses the threshold, perceptron fires and if not then perceptron doesn't predict
*  Perceptron can take real values input or boolean values
* Actually, when w * x + b = 0, the perceptron outputs 0
* Disadvantage of perceptron is that it only outputs binary values and if we try to give small change in weight and bias then perceptron can flip the output. We need some system which can modify the output slightly according to small change in weight and bias. Enter sigmoid function.
* If we change perceptron with a sigmoid function, then we can make a slight change in output e.g. output in perceptron = 0, you slightly changed weight and bias, output becomes = 1 but actual output is 0.7. In case of sigmoid, output1 = 0, slight change in weight and bias, output = 0.7.
* If we apply sigmoid activation function then a single neuron will act as Logistic Regression.
* we can understand difference between perceptron and sigmoid function by looking at sigmoid function graph.
* Simple NN graph
  * <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/9903d143-3ef8-45ca-86ed-7eb60b42c2e6.png" width="600" />
