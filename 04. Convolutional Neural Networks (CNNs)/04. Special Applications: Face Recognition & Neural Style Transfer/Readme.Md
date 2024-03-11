# Goal and Learning Objectives:
Explore how CNNs can be applied to multiple fields, including art generation and face recognition, then implement your own algorithm to generate art and recognize faces!
* Differentiate between face recognition and face verification
* Implement one-shot learning to solve a face recognition problem
* Apply the triplet loss function to learn a network's parameters in the context of face recognition
* Explain how to pose face recognition as a binary classification problem
* Map face images into 128-dimensional encodings using a pretrained model
* Perform face verification and face recognition with these encodings
* Implement the Neural Style Transfer algorithm
* Generate novel artistic images using Neural Style Transfer
* Define the style cost function for Neural Style Transfer
* Define the content cost function for Neural Style Transfer

# Face Recognition
## What is Face Recognition?
* Face recognition systems identify a person's face and can work on both images or videos
* Liveness detection within a video face recognition system prevents the CNN from identifying a face in a picture. (It can be learned by supervised deep learning using a dataset for live human and in-live human and sequence learning.)
* Face verification vs. face recognition:
    * Verification:
        * Input: image, name/ID.
        * Output: whether the input image is that of the claimed person. (1:1 problem)
        * You'd ask yourself "Is this person who he/she claims to be?"
    * Recognition:
        * Has a database of K persons
        * Get an input image
        * Output ID if the image is any of the K persons (or not recognized) (1:k problem)
        * You'd ask yourself: "Who is this person?"
* We can use a face verification system to make a face-recognition system. The accuracy of the verification system has to be high (around 99.9% or more) to be use accurately within a recognition system because the recognition system accuracy will be less than the verification system given K persons.

## One Shot Learning
* One of the face recognition challenges is to solve one shot learning problem
* One Shot Learning: A recognition system is able to recognize a person, learning from one image
* Historically, deep learning doesn't work well with a small number of data
  * Also, when adding a new sample (new face), we would have to train the model all over again => not a good approach
* Instead to make this work, we will learn a **similarity function**:
  * `d(img1, img2)` = degree of difference between images
  * We want `d` result to be low in case of the same faces, high in case of a different face
  * We use `T` (tau) as a threshold for d:
    * If `d(img1, img2) <= T`, then the faces are the same
    * If `d(img1, img2) >  T`, then the faces are different
* Similarity functions help us solving the one shot learning
  * Also this CNN still works fine with new inputs added to the database

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/bc188b88-d5ee-43ae-b181-8dbcbf5bb7b7.png" width="700" />

## Siamese Network
* We will implement the similarity function `d` using a type of NNs called **Siamease Network** in which we can pass multiple inputs to two or more networks with the same architecture and parameters
* Siamese network architecture are as the following:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/a3b4e5c4-e95c-4ca8-9cb4-a535fe9542bd.png" width="700" />

* We make 2 identical conv nets which encode an input image into a vector. In the above image the vector shape is (128, )
*  The loss function will be `d(x1, x2) = || f(x1) - f(x2) ||^2`
   * If X1, X2 are the same person, we want d to be low
   * If X1, X2 are different persons, we want d to be high
* Paper: [Taigman et al., 2014, DeepFace closing the gap to human level performance](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Taigman_DeepFace_Closing_the_2014_CVPR_paper.html)

## Triplet Loss
* Triplet Loss is one of the loss functions we can use to solve the similarity distance in a Siamese Network
* Our learning objective in the triplet loss function is to get the distance between an Anchor image (A) and a positive (P) or a negative (N) image (hence 'triplet')
   * Positive means same person, while negative means different person.
* Formally we want:
   * Positive distance to be less than negative distance, and actually much less (defined by the **margin** parameter 'alpha')

     <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/b7a70315-adb6-40ca-b218-df276877440a.png" width="700" />

   * The margin alpha pushes the anchor-positive pair and the anchor-negative pair further away from each other, forcing a higher confidence that pictures are either the same or different. So either the distance between A and P need to go down, or between A and N to go up.

* Final Loss function:
   * Given 3 images (A, P, N):
      * `L(A, P, N) = max(||f(A) - f(P)||^2  - ||f(A) - f(N)||^2 + alpha , 0)`
      * `J = Sum(L(A[i], P[i], N[i]))` for all triplets of images
      * the max() function above ensures that the distance function minimim value cannot be lower than 0
   * You need multiple images of the same person in your dataset. Then get some triplets out of your dataset. So the dataset should be big enough.
      * i.e. 10k pictures of 1k person for the training set
      * After training, you can use 1 picture per person for one-shot learning
* Choosing the triplets A, P, N:
   * During training if A, P, N are chosen randomly (Subjet to A and P are the same and A and N aren't the same) then one of the problems this constrain is easily satisfied
      * `d(A, P) + alpha <= d(A, N)`
      * So the NN wont learn much
   * What we want to do is choose triplets that are hard to train on
      * So for all the triplets we want this to be satisfied:
         * `d(A, P) + alpha <= d(A, N)`
         * `d(A, P) ~= d(A, N)`
            => increase the computational efficiency of the learning algorithm
      
           <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/73bb4b24-cbef-4d07-83d9-cb3a90cbadd0.png" width="700" />

      * This can be achieved by i.e. choosing people with high similar appearance for P and N
      * Details of choosing triplets: [Schroff et al.,2015, FaceNet: A unified embedding for face recognition and clustering](https://arxiv.org/abs/1503.03832)
* Commercial recognition systems are trained on a large datasets like 10/100 million images that are difficult to acquire
* There are a lot of pretrained models and parameters online for face recognition to circumvent the potential issue of obtaining such large data sets

## Face Verification and Binary Classification
* Triplet loss is one way to learn the parameters of a conv net for face recognition
* There's another way to learn these parameters as a straight binary classification problem
   *  Learning the similarity function another way:

      <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/bda9b9ea-ed57-46cd-8fbd-e4b7804ae892.png" width="700" />

      * The final layer is a sigmoid layer
      * Then Y_hat = Sigmoid( w<sub>i</sub> * | f(x_i) - f(x_j) | + b)
      * Some other similarities can used instead e.g., Chi-square similarity (green formula in  below image)
   * Paper: Taigman et al., 2014, DeepFace closing the gap to human level performance
* A good performance/deployment trick:
   * Pre-compute all the images that you are using as a comparison to the vector f(x_j)
   * When a new image needs to be compared, get its vector f(x_i), then compare it to all the pre-computed vectors in the data base and pass the 'difference' fector to the sigmoid function (as above)
   * This version works as well as the triplet loss function

     <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/dc89c5a3-4949-4f0b-9eb1-1b3f6ec980d1.png" width="700" />

* Face verification supervised learning
   * Create a training set of pair of images where y = 1 when 2 pictures come from the same person and y = 0 otherwise
   * Then, training the Siamese neural network like normal with back progagation

# Neural Style Transfer
## What is Neural Style Transfer?
* Technique in computer vision and image processing that combines the content of one image (C) with the artistic style of another image (S)
* This method uses deep NN to analyze and extract the content and style features of two input images and then generates a new image (G) that merges these features

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/72c4f933-11ca-4076-aee5-eca99ac11829.png" width="700" />

* In order to implement this you need to look at the features extracted by the Conv net at the shallower and deeper layers
* Reminder:  The idea of using a network trained on a different task and applying it to a new task is called transfer learning

## What are deep ConvNets learning?
*  Visualizing what a deep network is learning:
   * Given this AlexNet like Conv net:

     <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/b73a76a1-a564-49d5-8520-15416baf80dd".png" width="700" />

   * Pick a unit in layer l. Find the 9 image patches that maximize that unit's activation
      * Notice that a hidden unit in layer 1 will see a relatively small portion of the NN, so if you plotted it it will match a small image in the shallower layers, while a larger image can be seen by a unit in deeper layers
   * Repeat for other units and layers
* It turns out that layer 1 is learning the low level representations like colors and edges
   * You will find out that as you go deeper into the NN, each layer is learning increasingly more complex representations

     <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/4a09b169-a704-45e8-b47c-8ef1d220cb8f.png" width="700" />

* The first layer was created using the weights of the first layer; other images are generated using the receptive fields in the image that triggered the unit to produce maximal activation values
* Paper: [Zeiler and Fergus., 2013, Visualizing and understanding convolutional networks](https://arxiv.org/abs/1311.2901)

## Cost Function
* We will define a cost function for the generated image that calculates how good the image is
* Give a content image C, a style image S, and a generated image G:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/fa4ef091-e37d-4c58-9d88-067aa8e841e5.png" width="700" />

  * J_content(C, G) measures how similar the content is between the content image C and the generated image G
  * J_style(S, G) measures how similar the style is between image S and image G
    * alpha and beta are relative weighting to the similarity and these are hyperparameters
  * Paper: [Gatys et al., 2015, A neural algorithm of artistic style](https://arxiv.org/abs/1508.06576)
* How to find the generated image G:
   * Initiate G randomly e.g., G: 100 x 100 x 3
   * Use gradient descent to minimize J(G) i.e., `G = G - dG` 
      * We compute the gradient image and use gradient decent to minimize the cost function
   * The iterations might be as the following image by minimizing the cost function:
      * To Generate this:

        <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/c25b393e-b812-4741-a250-91eb63de8332.png" width="700" />

      * You will go through this:

        <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/577dd182-a4f3-4852-9dbb-f5e1468e23d9.png" width="700" />

   * Note that the pixel values of G are updated after each iteration of the optimization algorithm

## Content Cost Function
* In the previous section, we showed that we need separate cost functions for the content image as well as the style image to measure how similar images are to each other
* Let's use an example
   * Say you use hidden layer l to compute content cost
   * If we choose l to be small (like layer 1), we will force the network to get similar output to the original content image
   * In practice, l is typically not too shallow and not too deep but somewhere in the middle
   * Use pre-trained ConvNet (e.g., VGG network)
   * Let a<sup>l C</sup> and a<sup>[l](G)</sup> be the activation of layer l on the images
   * If a<sup>[l](C)</sup> and a<sup>[l](G)</sup> are similar, then both images have similar content
   * So the J_content(C, G) function becomes at layer l = ||a<sup>[l](C)</sup> - a<sup>[l](G)</sup>||<sup>2</sup>



<img src=".png" width="700" />
