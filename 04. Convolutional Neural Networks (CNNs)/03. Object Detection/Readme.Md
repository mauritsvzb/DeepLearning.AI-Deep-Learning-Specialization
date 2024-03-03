# Goal and Learning Objectives
Apply your new knowledge of CNNs to one of the hottest (and most challenging!) fields in computer vision: object detection
* Identify the components used for object detection (landmark, anchor, bounding box, grid, ...) and their purpose
* Implement object detection
* Implement non-max suppression to increase accuracy
* Implement intersection over union
* Handle bounding boxes, a type of image annotation popular in deep learning
* Apply sparse categorical crossentropy for pixelwise prediction
* Implement semantic image segmentation on the CARLA self-driving car dataset
* Explain the difference between a regular CNN and a U-net
* Build a U-Net

# Object Detection
* Object detection is one of the areas in which deep learning is doing great in the past two years.
* What are localization and detection?
  * Image Classification:
    * Classify an image to a specific class. The whole image represents one class. We don't want to know the location of the the object. Usually only one object is presented.
  * Classification with localization:
    * Given an image we want to learn the class of the image and where are the class location in the image. We need to detect a class and a rectangle of where that object is. Usually only one object is presented.
  * Object detection:
    * Given an image we want to detect all the object in the image that belong to a specific classes and give their location. An image can contain more than one object with different classes.

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/10a3361f-ac7b-404a-ad27-8e15f6e4cfe3.png" width="700" />

* To make image classification we use a Conv Net with a Softmax attached to the end of it
* To apply classification with localization, we use a Conv Net with a `softmax` attached to the end of the model and four numbers `bx`, `by`, `bh`, and `bw` which tell you the location of the bounding box around an object/class. The dataset should contain these four numbers with the class too.

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/f43a051a-87ba-4f3d-9e74-696fe4716b88.png" width="700" />

* Defining the target label Y in classification with localization problem:
```{python}
  Y = [
        Pc  # Probability of an object is presented
        bx  # Bounding box
        by  # Bounding box
        bh  # Bounding box
        bw  # Bounding box
        c1  # The classes
        c2
        ...
      ]
```
  * When object is presented:
    ```{python}
    Y = [
            1     # Object is present
            0
            0
            100
            100
            0
            1
            0
          ]
    ```
  * When object isn't presented:
    ```{python}
    Y = [
           0   # Object isn't presented
           ?   # ? means we dont care with other values
           ?
           ?
           ?
           ?
           ?
           ?
         ]
    ```

* The loss function for the Y we have created (Example of the square error):
  ```{python}
  L(y',y) = {
                (y1'-y1)^2 + (y2'-y2)^2 + ... + y8'-y8)^2     #if y1 = 1
                (y1'-y1)^2      #if y1 = 0
             }
  ```

* In practice we use logistic regression for Pc, log likely hood loss for classes, and squared error for the bounding box

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/a03fdcaa-fcaa-46e2-b056-b275c435b988.png" width="700" />

## Landmark Detection
* In some of the computer vision problems you will need to output some points, a process called landmark detection
* e.g., if you are working on a face-recognition problem, you might want some points on the face like corners of the eyes, corners of the mouth, and corners of the nose and so on. This can help in a lot of application like detecting the pose of the face.
```{python}
Y shape for the face recognition problem that needs to output 64 landmarks:
  Y = [
        THere_Is_A_face  # Probability of face is presented 0 or 1
        l1x,
        l1y,
        ...
        l64x,
        l64y
      ]
```
* Another application is when you need to get the skeleton of the person using different landmarks/points in the person which helps in some applications
* Note: in your labeled data, if l1x,l1y is the left corner of left eye, all other l1x,l1y of the other examples has to be the same i.e., consistent!

## Object Detection
* We will use a ConvNet to solve the object detection problem using a technique called 'sliding windows detection algorithm'
  * E.g., let's say we are working on Car object detection
  * First, we will train a ConvNet on cropped car images and non car images

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/d74f0318-b2d6-4a25-bc6e-c0db3223e2b7.png" width="700" />

  * After we finish training of this ConvNet, we will then use it with the sliding windows technique
* Sliding windows detection algorithm:
  * Decide a sliding window (rectangle) size
  * Slide your window across the image, ensuring each region is covered (you can use whatever value for stride)
  * For each window, feed the image into the ConvNet and decide it decide if there's a car or not
  * Pick a larger window size and repeat the process from the previous 2 steps
  * Store the rectangles that contain a car
    * If two or more rectangles intersects choose the rectangle with the best accuracy.
* Disadvantage of sliding window is the computation time
  * Before the deep learning era, people used hand-crafted linear classifiers to classify an object and then used the sliding window technique. The linear classier make it a cheap computation. But in the deep learning era this is computationally expensive due to the complexity of the deep learning model
  * To solve this problem, we can implement the sliding window detection algorithm with a convolutional approach

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/bcccf904-5572-4a23-a510-68f62d3bbb09.png" width="700" />

## 

## 
