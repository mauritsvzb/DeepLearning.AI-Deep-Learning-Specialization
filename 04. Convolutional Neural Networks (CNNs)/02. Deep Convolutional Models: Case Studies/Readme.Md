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
### LeNet-5
* The goal for this model was to identify handwritten digits in a 32x32x1 gray image. Here is the drawing of it:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/2e4f63f8-16f5-42ed-81ce-37c9ebffc416.png" width="700" />

* This model was published in 1998
* The last layer wasn't using softmax back then, but we would do that now
* It has 60k parameters
* The dimensions of the image decrease as the number of channels increases
* Conv ==> Pool ==> Conv ==> Pool ==> FC ==> FC ==> softmax this type of arrangement is quite common
* The activation function used in the paper was Sigmoid and Tanh. Modern implementation uses ReLU in most of the cases
* Paper: [LeCun et al., 1998. Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791?reload=true)

### AlexNet
* Named after Alex Krizhevsky who was the first author of this paper
* The goal for the model was the ImageNet challenge which classifies images into 1,000 classes 
* Here are the drawing of the model:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/65f33c17-69fa-4295-9480-81499b46cfe3.png" width="700" />

* Summary:
  * Conv => Max-pool => Conv => Max-pool => Conv => Conv => Conv => Max-pool ==> Flatten ==> FC ==> FC ==> Softmax
  * Similar to LeNet-5, but bigger
  * Has 60 Million parameters compared to 60k parameters of LeNet-5
  * It used the ReLU activation function
  * The original paper contains multiple GPUs and Local Response Normalization (LRN)
  * Multiple GPUs were used because the GPUs were not so fast back then
  * Researchers proved that Local Response Normalization doesn't help much (so for now don't bother yourself for understanding or implementing it)
  * This paper convinced the computer vision researchers that deep learning is so important
  * Paper: [Krizhevsky et al., 2012. ImageNet classification with deep convolutional neural networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

### VGG-16
* A modification for AlexNet
* Instead of having a lot of hyperparameters, it has a much simpler network
* Focus on having only these blocks:
  * CONV = 3 x 3 filter, s = 1, "same"
  * MAX-POOL = 2 x 2 , s = 2
* Here are the architecture:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/6233028b-998e-48ac-94df-5697be5cf5e5.png" width="700" />

* This network is large even by modern standards
* It has around 138 M parameters
* Most of the parameters are in the fully connected layers
* It has a total memory of 96 MB per image for only forward propagation!
* Most memory are in the earlier layers
* Number of filters increases from 64 to 128 to 256 to 512; 512 was made twice
* Pooling was the only layer responsible for shrinking the dimensions
* There is another version called VGG-19 which is a bigger version; most people uses the VGG-16 instead of the VGG-19 because it does the same
* VGG paper is attractive as it tries to make some rules regarding using CNNs
* Paper: [Simonyan & Zisserman 2015. Very deep convolutional networks for large-scale image recognition](https://arxiv.org/abs/1512.03385)

## Residual Networks (ResNets)
* Very, very deep NNs are difficult to train because of vanishing and exploding gradients problems
* Skip connections, which make you take the activation from one layer and feed it to another layer deeper in the NN, circumvent this problem, enabling the training of large NNs with > 100 layers 
* <b>Residual block</b>
  * ResNets are constructed from residual blocks

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/61020df0-3850-4728-bb4d-0957329acaa3.png" width="700" />

  * Add a short cut/skip connection before the second activation
  * The authors of this block find that you can train a deeper NNs using stacking this block
  * Paper: [He et al., 2015. Deep residual networks for image recognition](https://arxiv.org/abs/1512.03385)
* <b>Residual Network</b>
  * NNs that consist of some residual blocks

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/3ac507f0-e604-444b-ae78-03307c77cd0b.png" width="700" />

  * These networks can go deeper without hurting performance; in the normal NN ("Plain networks") the theory tells us that if we go deeper we will get a better solution to our problem, but because of the vanishing and exploding gradients problems the performance of the network suffers as it goes deeper
  * Thanks to Residual Network we can go as deep as we want now, although you can likely experience diminishing returns (plateau)

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/99e94ad9-2dac-488b-99fa-7d9e1614f0c0.png" width="700" />

  * On the left is the normal NN and on the right is the ResNet: the performance of ResNet increases as the network goes deeper
  * In some cases going deeper won't further improve the performance (the point at which this happens depends on the problem on your hand)
  * Some people are experimenting now with 1000 layer ResNets but this isn't used in practice

## Why ResNets work
* Lets see some example that illustrates why resNet work
  * We have a big NN as the following:
    `X --> Big NN --> a[l]`
  * Lets add two layers to this network as a residual block:
    `X --> Big NN --> a[l] --> Layer1 --> Layer2 --> a[l+2]`
    And `a[l]` has a direct connection to `a[l+2]`
  * Suppose we are using ReLU activations.
  * Then:
    ```
    a[l+2] = g( z[l+2] + a[l] )
           = g( W[l+2] a[l+1] + b[l+2] + a[l] )
    ```
  * Then if we are using L2 regularization for example, `W[l+2]` will be zero. Lets say that `b[l+2]` will be zero too
  * Then `a[l+2] = g( a[l] ) = a[l]` with no negative values
  * This shows that the <b>identity function is easy for a residual block to learn. And that why it can train deeper NNs</b>
  * Also that the two layers we added doesn't hurt the performance of big NN we made
  * Note: dimensions of `z[l+2]` and `a[l]` have to be the same in resNets. In case they have different dimensions what we put a matrix parameters (Which can be learned or fixed)
    * `a[l+2] = g( z[l+2] + ws * a[l] ) # The added Ws should make the dimensions equal`
   * `Ws` also can be a zero padding
* Using a skip-connection helps the gradient to backpropagate and thus helps you to train deeper networks
* Lets take a look at ResNet on images
  * Here are the architecture of ResNet-34:
    
    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/95dd2f69-3d5c-400a-82e5-3084bcd34045.png" width="700" />

  * All the 3x3 Conv are "Same Conv"
  * Keep it simple in design of the network
  * spatial size /2 => # filters x2
  * Two main types of blocks are used in a ResNet, depending mainly on whether the input/output dimensions are same or different
  * The solid line is the case when the dimensions are 'same'
  * The dotted lines is the case when the dimensions are different. To solve then they down-sample the input by 2 and then pad zeros to match the two dimensions. There's another trick which is called bottleneck which we will explore later

## 
