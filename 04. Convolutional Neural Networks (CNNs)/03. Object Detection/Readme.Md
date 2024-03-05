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
        Pc  # Probability of an object is present
        bx  # Bounding box
        by  # Bounding box
        bh  # Bounding box
        bw  # Bounding box
        c1  # The classes
        c2
        ...
      ]
```
  * When object is present:
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
  * When object isn't present:
    ```{python}
    Y = [
           0   # Object isn't present
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
  * Slide your window across the image (from left to right, top to bottom), ensuring each region is covered (you can use whatever value for stride)
  * For each window, feed the image into the ConvNet and decide it decide if there's a car or not
  * Pick a larger window size and repeat the process from the previous 2 steps
  * Store the rectangles that contain a car
    * If two or more rectangles intersects choose the rectangle with the best accuracy.
* Disadvantage of sliding window is the computation time
  * Before the deep learning era, people used hand-crafted linear classifiers to classify an object and then used the sliding window technique. The linear classier make it a cheap computation. But in the deep learning era this is computationally expensive due to the complexity of the deep learning model
  * To solve this problem, we can implement the sliding window detection algorithm with a convolutional approach

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/bcccf904-5572-4a23-a510-68f62d3bbb09.png" width="700" />

## Convolutional Implementation of Sliding
* Turning a FC layer into convolutional layers (predict image class from four classes):

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/c19fda5f-2e2f-4d15-8c83-b14a1065acd4.png" width="700" />

  * As can be seen in the above image, we turned the FC layer into a Conv layer using a convolution with the width and height of the filter being the same as the width and height of the input
* Convolution implementation of sliding windows:
  * First, let's consider that the ConvNet you trained is like this (No FC: all is conv layers):

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/ae80c878-a67e-457f-81fd-ccdd463c29c9.png" width="700" />

  * Say now we have a 16 x 16 x 3 image from the test set that we want to apply the sliding windows to. From the previous, we would run this ConvNet 4 times, using a rectangle size of 14 x 14 with stride 2.
  * The convolution implementation will be as follows:
  
    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/a88c6f86-1fb2-4235-a8f9-7f697a9a7389.png" width="700" />

  * We have fed the image into the same ConvNet we just trained
  * The left cell of the result "the blue square" will represent the the first sliding window of the normal implementation. The other squares will represent the other windows  (slide right, slide down, slide right and down).
  * This is more efficient because ConvNet now shares a lot of the computations between each of the four windows, because the model combines all 4 computations into 1 forward propagation function

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/8350eab3-ad24-401a-8575-a5915f6185cc.png" width="700" />

  * Another example would be:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/ed589eb7-736d-43ed-9638-ad666afbe371.png" width="700" />

  * This example applies all sliding windows simultaneously whereby the  CNN shares computations due to 1 forward prop, hopefully capturing the image successfully
  * Paper: [Sermanet et al., 2014, OverFeat: Integrated recognition, localization and detection using convolutional networks](https://arxiv.org/abs/1312.6229)


## Bounding Box Predictions
* A weakness of the sliding window detection algorithm is that the position of the rectangle may not be that accurate. Maybe none of the rectangles is exactly on the object you aim to detect.

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/6b4f2491-3673-48f7-a82a-7985c0cd1160.png" width="700" />

  * In red, the rectangle we want and in blue is the rectangle we get  from applying this algorithm
* A better algorithm is the YOLO (You Only Look Once) algorithm
  * Paper: [Redmon et al., 2015, You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
* YOLO: developed back in 2015
* YOLO basic idea:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/33520dd1-d76d-498a-8436-61d52d5f0191.png" width="700" />

  1. Lets say we have an image of 100 X 100
  2. Place a 3 x 3 window on the image. For smoother results use 19 x 19 for the 100 x 100
  3. Apply the classification and localization algorithm we discussed in a previous section to each section of the grid. bx and by will represent the center point of the object in each grid and will be relative to the box so the range is between 0 and 1 while bh and bw will represent the height and width of the object which can be > 1

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/1b78c1ef-6973-46a7-9248-a15da5aa0fa5.png" width="700" />

  4. Do everything at once with the convolution sliding window. If Y shape is 1 x 1 x 8 as we discussed before then the output of the 100 x 100 image should be 3 x 3 x 8 which corresponds to 9 cell results
  5. Merging the results using predicted localization mid point
* We have a problem if we have found more than one object in one grid box
* One of the best advantages that makes the YOLO algorithm popular is that it has great speed and a ConvNet implementation.
* How is YOLO different from other Object detectors? YOLO uses a single CNN network for both classification and localizing the object using bounding boxes
* In the next sections we will see some ideas that can make the YOLO algorithm better.

## Intersection Over Union
*  Intersection Over Union is a function used to evaluate the object detection algorithm
* It computes the size of intersection (area where both bounding boxes overlap) and divide it by the union (total area size of both bounding boxes). More generally, IoU is a measure of the overlap between two bounding boxes
* For example:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/fbcf6a7b-997f-4c26-8598-5b417b0c06fa.png" width="700" />

* The red is the labeled output and the purple is the predicted output
* To compute Intersection Over Union, we first compute the union area of the two rectangles which is "the first rectangle + second rectangle", then compute the intersection area between these two rectangles.
* Finally IOU = intersection area / Union area
* If IOU >=0.5 then the algorithm is good. 
  * The best answer will be 1 i.e., bounding boxes completely overlap
  * Threshold value chosen rather arbitrarily, but  generally falls between 0.5 and 0.7
* The higher the IOU the higher the accuracy

## Non-Max Suppression
* One of the problems we have discussed previously is that our algorithm (YOLO) can detect an object multiple times
* Non-max Suppression is a way to make sure that YOLO detects the object just once
* For example:
  * Each car has two or more detections with different probabilities. This result came from the fact that some of the grid cellss thought that they have the center point of the object in their grid cell.

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/d451d01f-afb3-465e-a802-130cfb8507bd.png" width="700" />

* Non-max suppression algorithm details:
  * Lets assume that we are targeting one class as an output class (so we can drop c1, c2, and c3 from output Y)
  * Y shape should be `[Pc, bx, by, bh, hw]` where `Pc` is the probability if that object occurs
  * Discard all boxes with `Pc < 0.6`
  * While there are any remaining boxes:
    * Pick the box with the highest Pc value and output that as a prediction (e.g., highlight the bounding box in the video)
    * Discard any remaining boxes with IoU > 0.5 with that box output from the previous step i.e., any box with with an IoU > 0.5 gets suppressed (darkened color of bounding box in video)
    * Repeat until you processed all bounding boxes (either box used for prediction or discard)
* If there are multiple classes/object types c you want to detect, you should run the Non-max suppression c times, once for every output class.

## Anchor Boxes
* What we've seen so far is that with YOLO only 1 object is detected
* What if a grid cell detects multiple objects simultaneously? e.g., if a the center of a car and person are in the same grid cell?
  * In practice this happens rarely (especially if you're using a fine grid)
* We can use anchor boxes (see Redmon et al., 2015)
  
  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/4278d94e-5fc9-44da-ab76-4529b5e791de.png" width="700" />

  * If `Y = [Pc, bx, by, bh, bw, c1, c2, c3]` for 1 object detection, then we just repeat this once for detection 2 objects using 2 anchor boxes
    
  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/e74ee191-ebed-44d1-96a8-4e06b9e65ad4.png" width="700" />

  * You have to check where your object should be based on its rectangle closest to which anchor box

* Anchor box example:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/e27df99c-9953-4c1b-95f1-ef3d0e26cbcb.png" width="700" />

  * Where the car bounding box better matches anchor box 2 i.e., has higher overlap compared to with anchor box 1, and the human bounding box better matches anchor box 1
* You may have two or more anchor boxes but you should know their shapes
  * How do you choose the anchor boxes and people used to just choose them by hand. Maybe five or ten anchor box shapes that span a variety of shapes that cover the types of objects you seem to detect frequently.
  * Better, more advanced way to choose anchor boxes is by using k-means clustering algorithm on your dataset to specify shapes of anchor boxes
* Anchor boxes allows your algorithm to specialize, means in our case to easily detect fat, wider objects or tall, skinny ones
* However, the algorithms doesn't handle well when
  * There are 3 objects associated in the same grid cell with only 2 anchor boxes
  * There are 2 objects associated with the same grid cell, and both of them have the same anchor box shape

## YOLO Algorithm
* YOLO is a state-of-the-art object detection model that is fast and accurate
* Let's summarize the whole YOLO algorithm given an example:
  * Construct the training set:
    * Suppose we need to do object detection for our autonomous driver system. It needs to identify three classes:
      * Pedestrian (Walks on ground)
      * Car
      * Motorcycle
  * Suppose we decide to choose two anchor boxes, a taller one and a wide one (In practice, they use five or more anchor boxes handmade or generated using k-means
  * Our labeled Y shape will be `[Ny, HeightOfGrid, WidthOfGrid, 16]`, where `Ny` is number of instances and each row (of size 16) is as follows:
    * `[Pc, bx, by, bh, bw, c1, c2, c3, Pc, bx, by, bh, bw, c1, c2, c3]`
  * Your dataset could be an image with a multiple labels and a rectangle for each label
  * An example:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/3e6473fc-19f5-4997-a74a-5e02cd0ae613.png" width="700" />

  * We first initialize all of them to zeros and ?, then for each label and rectangle choose its closest grid point then the shape to fill it and then the best anchor box based on the IOU. so that the shape of Y for one image should be `[HeightOfGrid, WidthOfGrid, 16]`
  * Train the labeled images on a Conv net. You should get an output of [HeightOfGrid, WidthOfGrid, 16] for our case
  * To make predictions, run the Conv net on an image and run Non-max suppression algorithm for each class you have (in our case there are 3 classes)
  * You could get something like that:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/78122e3e-a52b-4ed2-b09d-d5f01e08c1b6.png" width="700" />

    * Total number of generated boxes are `grid_width * grid_height * no_of_anchors` = `3 x 3 x 2`
  * By removing the low probability predictions you should have:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/5673df76-6937-4f34-a247-f034d1b74a5b.png" width="700" />

  * Then get the best probability followed by repeating the non-max suppression as many times as you have number of classes:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/2a04e405-5e76-427b-a149-3301a82114f7.png" width="700" />

## 
