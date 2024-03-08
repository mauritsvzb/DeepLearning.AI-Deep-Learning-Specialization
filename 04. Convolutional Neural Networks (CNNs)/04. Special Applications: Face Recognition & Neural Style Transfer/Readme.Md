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

## 





<img src=".png" width="700" />







# Neural Style Transfer
## 
