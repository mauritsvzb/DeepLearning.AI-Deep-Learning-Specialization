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
* To provide a general rule: if a matrix n x n is convolved with f x f filter/kernel, we get a n - f + 1, n - f + 1 matrix
* There are 2 problems with this:
  * The convolution operation shrinks the matrix if f > 1
    * We want to apply convolution operation multiple times, but if the image shrinks at each step we will lose a lot of data on this process
  * The edge pixels are used less than those in the center
    * Throws away a lot of information that are in the edges
* To solve these problems, we can pad the input image before convolution by adding some rows and columns to it. We will call the padding amount p the number of layers (row/columns) that we will inserted around the image
* In almost all the cases the padding values are zeros
* The general rule now is that if a matrix n x n is convolved with f x f filter/kernel and padding p give us n + 2 p - f + 1, n + 2 p - f + 1 matrix
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








