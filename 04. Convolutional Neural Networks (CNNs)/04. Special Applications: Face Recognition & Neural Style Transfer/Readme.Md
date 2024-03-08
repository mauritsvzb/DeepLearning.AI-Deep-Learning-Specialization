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
*  The loss function will be d(x1, x2) = || f(x1) - f(x2) ||^2
   * If X1, X2 are the same person, we want d to be low
   * If X1, X2 are different persons, we want d to be high
* Paper: [Taigman et al., 2014, DeepFace closing the gap to human level performance](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Taigman_DeepFace_Closing_the_2014_CVPR_paper.html)

## Triplet Loss

<img src=".png" width="700" />







# Neural Style Transfer
## 
