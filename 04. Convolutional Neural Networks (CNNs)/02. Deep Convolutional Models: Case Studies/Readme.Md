# Goal and Learning Objectives:
Discover some powerful practical tricks and methods used in deep CNNs, straight from the research papers, then apply transfer learning to your own deep CNN
* Implement the basic building blocks of ResNets in a deep neural network using Keras
* Train a state-of-the-art neural network for image classification
* Implement a skip connection in your network (a form of regularization)
* Create a dataset from a directory
* Preprocess and augment data using the Keras Sequential API
* Adapt a pretrained model to new data and train a classifier using the Functional API and MobileNet
* Fine-tine a classifier's final layers to improve accuracy

# Deep Convolutional Models: Case Studies
## Why Look at Case Studies
* We learned about Conv layer, pooling layer, and fully connected layers. It turns out that computer vision researchers spent the past few years on how to put these layers together
* To get some intuitions you have to see some examples
* Some neural networks architecture that works well in some tasks can also work well in some other tasks
* Here are some classical CNN networks:
  * LeNet-5
  * AlexNet
  * VGG
* The best CNN architecture that won the last ImageNet competition is called ResNet and it has 152 layers!
* There is also an architectures called Inception that was made by Google, which is very useful to learn in order to apply to your tasks
* Reading and trying the aforementioned models can boost you and give you a lot of ideas to solve your tasks

## Classic Networks

