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
    * For example, a 1000x1000 image will represent 3 million feature/input values to the full connected neural network. If the following hidden layer contains 1000, then we will want to learn weights of the shape [1000, 3 million] which is 3 billion parameters only in the first layer and thats so computationally expensive!
* One of the solutions is to build such a NN using convolution layers instead of the fully connected layers

## Edge detection example
* The convolution operation is one of the fundamentals blocks of a CNN; one of the examples about convolution is the image edge detection operation
* Early layers of CNN might detect edges then the middle layers will detect parts of objects and the later layers will put the these parts together to produce an output
* In an image we can detect vertical edges, horizontal edges, or full edge detector
* Vertical edge detection:
  * An example of convolution operation to detect vertical edges:
    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/fc8eed7c-f12b-4ddc-a5c8-febe735e4e1e.png" width="700" />
  * In this example, a 6x6 matrix convolved with 3x3 filter/kernel gives us a 4 x 4 matrix
  * If you make the convolution operation in TensorFlow you will find the function `tf.nn.conv2d`; in keras you will find `Conv2d` function
  * The vertical edge detection filter will find a 3x3 place in an image where there are a bright region followed by a dark region
    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/6f7578c3-9644-49bc-9396-848c04cc8122.png" width="700" />
  * If we applied this filter to a white region followed by a dark region, it should find the edges in between the two colors as a positive value

## More Edge Detection
* Consider the previous image: if we applied the same filter to a dark region followed by a white region it will give us negative values. (If we don't care about transition from bright to dark or vise versa, then can use the abs function to always make it positive.)

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/aa69cf0a-fe35-48d5-9931-6e9ad02e5712.png" width="700" />

* Horizontal edge detection

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/286e1b2b-b269-4e6b-9505-7057b7cdf92d.png" width="700" />

* There are a lot of ways we can put numbers inside the horizontal or vertical edge detections. For example, here are the vertical Sobel filter and the Scharr filter (The idea is taking care of the middle row) :

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/1c5574bc-7eae-4422-a795-28c5bea6738a.png" width="700" />

* What we learned in deep learning is that we don't need to hand pick these numbers: we can treat them as weights and then learn them. It can learn horizontal, vertical, angled, or any edge type automatically rather than getting them by hand.




## Padding
* In order to use deep neural networks we really need to use paddings
* In the last section, we saw that a 6x6 matrix convolved with 3x3 filter/kernel gives us a 4x4 matrix
* To provide a general rule: if a matrix n x n is convolved with f x f filter/kernel, we get a `n - f + 1` x `n - f + 1` matrix

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/69dba0ac-7b04-420d-8d12-fa6f845da460.png" width="700" />

* There are 2 problems with this:
  * The convolution operation shrinks the matrix if f > 1
    * We want to apply convolution operation multiple times, but if the image shrinks at each step we will lose a lot of data on this process
  * The edge pixels are used less than those in the center
    * Throws away a lot of information that are in the edges
* To solve these problems, we can pad the input image before convolution by adding some rows and columns to it. We will call the padding amount p the number of layers (row/columns) that we will inserted around the image
* In almost all the cases the padding values are zeros
* The general rule now is that if a matrix n x n is convolved with f x f filter/kernel and padding p give us `n + 2 p - f + 1`, `n + 2 p - f + 1` matrix
  * If n = 6, f = 3, and p = 1, then the output image will have n + 2 p - f + 1 = 6 + 2 - 3 + 1 = 6, and therefore we maintain the initial size of the image

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/9517fcdc-48ac-47aa-b96f-16727e08a421.png" width="700" />

* "Valid convolution": the convolution that was discussed initially i.e., without padding: the image shrinks after each convolution
* "Same convolution": is a convolution with a pad so that output size is the same as the input size. 
  * Given by the equation: `P = (f-1) / 2`
  * In computer vision, f is usually odd, otherwise you get asymmetrical padding
  * Note: the value of p is determined by the size of the filter
    * In order for an image not to shrink when using a 3 x 3 filter on a 6 x 6 image, p should be equal to 1 in order to maintain the same size
    * If either the size of the image or filter changes, so must the value of p

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/e9af1752-8cda-48fe-88f5-30c2b7fda219.png" width="700" />

## Strided Convolutions
* Strided convolution is another piece that are used in CNNs
* We will call stride `S`
* When we are making the convolution operation we used `S` to tell us the number of pixels we will jump when we are convolving filter/kernel. The last examples we described S was 1
* Now the general rule is:
  * if a matrix `n x n` is convolved with `f x f` filter/kernel and padding `p` and stride `s` it give us `(n + 2p - f)/s + 1, (n + 2p - f)/s + 1` matrix
* In case `(n + 2p - f)/s + 1` is fraction we can take the `floor()` of this value

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/d1180d34-82a5-44d4-92e0-b3d8041036fb.png" width="700" />

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/2ed850ae-fee9-4d8c-8cf8-664aa06b8155.png" width="700" />

* In math textbooks the conv operation is filpping the filter before using it. What we were doing is called cross-correlation operation but the state of art of deep learning is using this as conv operation.
"Same Convolution" is a convolution with a padding so that output size is the same as the input size. Its given by the equation:
 `p = (n * s - n + f - s) / 2`
 When `s = 1` ==> `P = (f-1) / 2`

## Convolutions over Volume
* Let's see if we want to convolve 3D images (RGB image)
* We will convolve an image of height, width, # of channels with a filter of a height, width, same # of channels
  * Note that the image number of channels and the filter number of channels are the same
  * We just do the convolutional on the 3D filter similar to the 2D filter
    * Suppose the filter is 3x3x3, which are 27 blocks
    * Do a wise-multiply matrix between the filter and corresponded position of input, then add-up the 27 results numbers together
    * We can call this as stacked filters for each channel

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/801f0c1d-dbb0-4ab7-85a7-8063ed35ecca.png" width="700" />

  * We can use multiple filters to detect multiple features or edges:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/7f908996-6e85-454a-bd9e-fc2a7d55cae8.png" width="700" />

## One Layer of a Convolutional Network
* First we convolve some filters to a given input and then add a bias to each filter output and then get ReLU of the result e.g., 
  * Input image: 6x6x3 # a0
  * 10 Filters: 3x3x3 # W1
  * Result image: 4x4x10 # W1a0
  * Add b (bias) with 10x1 will get us : 4x4x10 image # W1a0 + b
  * Apply ReLU will get us: 4x4x10 image # A1 = ReLU(W1a0 + b)
  * In the last result p=0, s=1
  * Hint number of parameters here are: (3x3x3x10) + 10 = 280
  * The last example forms a layer in the CNN

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/d0da7d69-e0a3-44af-ac02-19a0d7d7b391.png" width="700" />

* Note: no matter the size of the input, the number of the parameters is same if filter size is same, which makes model less prone to overfitting
* Here are some notations we will use. If layer l is a conv layer:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/8f386cfe-67ec-4585-9dc6-bf13b97bc07d.png" width="700" />

## A simple convolution network example

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/6bc159d2-41e1-4209-bc55-76ef7adb10e6.png" width="700" />

* In this cNN, and CNNs more genearlly, the dimensions of the image trends down as you go deeper into the cNN, whereas the number of channels (N<sub>C</sub>) will generally increase
* Types of layer in a convolutional network:
  * Convolution (Conv)
  * Pooling (Pool)
  * Fully connected (FC)

## Pooling Layers
* Other than the conv layers, CNNs often uses pooling layers to:
  1. Reduce the size of the inputs, 
  2. Speed up computation, and 
  3. Make some of the features it detects more robust
* Max pooling example:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/26b3f633-83a3-4b91-bdd8-d8ca4bb3ab20.png" width="700" />

* The above example has f = 2, s = 2, and p = 0 hyperparameters
* The max pooling is saying that if the feature is detected anywhere in this filter then keep a high number (But the main reason why people are using pooling because its works well in practice and reduce computations)
* Max pooling has no parameters to learn
* Max pooling on 3D input: computating max pooling independently on each channels
* Example of Max pooling on 3D input:
  * Input: 4x4x10
  * Max pooling size = 2 and stride = 2
  * Output: 2x2x10
* Average pooling is taking the averages of the values instead of taking the max values
* Max pooling is used more often than average pooling in practice
* Hyperparameters summary

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/775a27c1-8eb3-449a-b445-2bad95d2883d.png" width="700" />

## CNN Example
* This example is something like the LeNet-5 that was invented by Yann Lecun

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/2ab3b795-abd9-45c3-b354-df17bc40ed28.png" width="700" />

* FC stands for: Fully Connected
* 2 conflicting conventions in NN: some count each seperate component as a layer, while others only count only layers weights e.g., a 'max pool' layer doesn't contain weights so is not counted seperately and thus belongs to the same layer as the previous component which does contain weights.
* Note: we follow the second convention here: Conv1 and Pool1 are treated as one layer

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/f0c30db0-cb11-4364-b569-812c3d068ac7.png" width="700" />

* There are a lot of hhyperparameters. For choosing the value of each, you should follow the guideline that we will discuss later or check the literature and takes some ideas and numbers from it
* Usually the input size decreases as you go deeper into the NN while at the same time the number of filters increases
* A CNN usually consists of one or more convolution (Not just one as the shown examples) followed by a pooling.
* Fully connected layers has the most parameters in the network
* To consider using these blocks together you should look at other working examples firsts to get some intuitions
* See https://towardsdatascience.com/understanding-and-calculating-the-number-of-parameters-in-convolution-neural-networks-cnns-fc88790d530d on how to calculate the number of parameters in each layer
  * ((shape of width of filter * shape of height filter * number of channels in the previous layer + 1) * number of filters in  current layer)

## 
* Two main advantages of Convs are:
  * Parameter sharing
    * A feature detector (such as a vertical edge detector) that's useful in one part of the image is probably useful in another part of the image
  * Sparsity of connections
    * In each layer, each output value depends only on a small number of inputs which makes it translation invariance
* These advantages help ConvNet
  * Has a lot fewer parameters => smaller training set => less prone to be overfitting
  * Good at capturing translation invariance
  * Like cat picture shifted a couple of pixels to the right is still recognized clearly a cat picture

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/9818749e-63f7-420c-a889-dc42dee717ae.png" width="700" />

Putting it all together:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/73afa6f7-a8ae-4c3f-b8e5-66cee993aa5d.png" width="700" />

## What you should remember:
* A convolution extracts features from an input image by taking the dot product between the input data and a 3D array of weights (the filter)
* The 2D output of the convolution is called the feature map
* A convolution layer is where the filter slides over the image and computes the dot product
  * This transforms the input volume into an output volume of different size
* Zero padding helps keep more information at the image borders, and is helpful for building deeper networks, because you can build a CONV layer without shrinking the height and width of the volumes
* Pooling layers gradually reduce the height and width of the input by sliding a 2D window over each specified region, then summarizing the features in that region

