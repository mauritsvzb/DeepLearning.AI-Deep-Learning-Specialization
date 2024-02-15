# Goal and Learning Objectives
Implement the foundational layers of CNNs (pooling, convolutions) and stack them properly in a deep network to solve multi-class image classification problems.
* Explain the convolution operation
* Apply two different types of pooling operations
* Identify the components used in a convolutional neural network (padding, stride, filter, ...) and their purpose
* Build a convolutional neural network
* Implement convolutional and pooling layers in numpy, including forward propagation
* Implement helper functions to use when implementing a TensorFlow model
* Create a mood classifer using the TF Keras Sequential API
* Build a ConvNet to identify sign language digits using the TF Keras Functional API
* Build and train a ConvNet in TensorFlow for a binary classification problem
* Build and train a ConvNet in TensorFlow for a multiclass classification problem
* Explain different use cases for the Sequential and Functional APIs

# Foundations of Convolutional Neural Networks (CNNs)
## Computer Vision
* Computer vision is one of the applications that is rapidly advancing thanks to deep learning
* Some of the applications of computer vision that are using deep learning includes:
  * Self driving cars
  * Face recognition
* Deep learning is also enabling new types of art to be created
* Rapid changes to computer vision are help create new applications that weren't possible a few years ago
* Computer vision deep leaning techniques are always evolving, making new architectures which can help us in other areas other than computer vision
  * For example, Andrew Ng took some ideas of computer vision and applied it in speech recognition
* Examples of a computer vision problems includes:
  * Image classification
  * Object detection
    * Detect object and localize them
  * Neural style transfer
    * Changes the style of an image using another image
* One of the challenges of computer vision problem that images can be so large and we want a fast and accurate algorithm to work with that
    * For example, a 1000 x 1000 image will represent 3 million feature/input values to the full connected neural network. If the following hidden layer contains 1000, then we will want to learn weights of the shape [1000, 3 million] which is 3 billion parameters only in the first layer and thats so computationally expensive!
* One of the solutions is to build such a NN using convolution layers instead of the fully connected layers

## Edge detection example
* The convolution operation is one of the fundamentals blocks of a CNN; one of the examples about convolution is the image edge detection operation
* Early layers of CNN might detect edges then the middle layers will detect parts of objects and the later layers will put the these parts together to produce an output
* In an image we can detect vertical edges, horizontal edges, or full edge detector
* Vertical edge detection:
  * An example of convolution operation to detect vertical edges:
    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/fc8eed7c-f12b-4ddc-a5c8-febe735e4e1e.png" width="700" />
  * In this example, a 6 x 6 matrix convolved with 3 x 3 filter/kernel gives us a 4 x 4 matrix
  * If you make the convolution operation in TensorFlow you will find the function `tf.nn.conv2d`; in keras you will find `Conv2d` function
  * The vertical edge detection filter will find a 3 x 3 place in an image where there are a bright region followed by a dark region
    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/6f7578c3-9644-49bc-9396-848c04cc8122.png" width="700" />
  * If we applied this filter to a white region followed by a dark region, it should find the edges in between the two colors as a positive value

## More Edge Detection




