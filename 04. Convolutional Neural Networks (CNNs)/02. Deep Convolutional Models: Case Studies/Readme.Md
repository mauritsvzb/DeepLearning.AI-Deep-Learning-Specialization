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
  * Then if we are using L2 regularization for example, and assume that `W[l+2]` will be 0 and `b[l+2]` will be 0 too, 
  * Then `a[l+2] = g( a[l] ) = a[l]` with no negative values
  * This shows that the <b>identity function is easy for a residual block to learn. And that why it can train deeper NNs</b>
  * Also that the two layers we added doesn't hurt the performance of big NN we made, and can even improve performance if they learn something better than the identity function (even identity functions are difficult to learn for a deep plain network!)
    * To summarize: it's very easy for ResNets to learn the identity function (that doesn't increase nor decrease perforamnce), but you can get lucky with the ResNet learning something better that does improve perforance; i.o.w., performance can only be the same or get better!
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

## Networks in Networks and 1x1 Convolutions
*  A 1 x 1 convolution - We also call it Network in Network - is so useful in many CNN models
* What does a 1 x 1 convolution do? Isn't it just multiplying by a number?
  * Lets first consider an example:
    * Input: 6x6x32
    * Conv: 1x1x32 one filter. # The 1 x 1 Conv
    * Output: 6x6x32
  * Another example:
    * Input: 6x6x32
    * Conv: 1x1x32 5 filters. # The 1 x 1 Conv
    * Output: 6x6x5
* Paper: [Lin et al., 2013. Network in network](https://arxiv.org/abs/1312.4400)
  * It has been used in a lot of modern CNN implementations like ResNet and Inception models
* A 1 x 1 convolution is useful when:
  * We want to shrink the number of channels (we also call this feature transformation)
    * In the second discussed example above we have shrinked the input from 32 to 5 channels
  * We will later see that by shrinking it we can save a lot of computations
  * If we have specified the number of 1 x 1 Conv filters to be the same as the input number of channels (like the first example) then the output will maintain the same number of channels. Then the 1 x 1 Conv will act like a non linearity operator allowing for more complex function to be learned in your NN through adding another layer.

## Inception Network Motivation
* When you design a CNN you have to decide which layers to include yourself: will you pick a 3 x 3 Conv or 5 x 5 Conv or maybe a max pooling layer (there are so many choices!)
* What inception tells us is, why not use all of them at once?
* Inception module, naive version:
    
  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/d727d4b1-4213-460f-a1cc-4a42965a7894.png" width="700" />

  * Note that max-pool here are of the form 'same convolution'
  * Input to the inception module is a 28 x 28 x 192 dimensional volume and the output is a 28 x 28 x 256 dimensional volume
  * Paper: [Szegedy et al. 2014. Going deeper with convolutions](https://arxiv.org/abs/1409.4842)
* The problem of computational cost in Inception model:
  * Let's illustrate the issue using the 5 x 5 Conv from last example:
    * There are 32 'same convolution' filters of dimensions 5 x 5 (x 192), and the input is of dimensions 28 x 28 x 192
    * Given an output of 28 x 28 x 32 (32 = number of filters), the total number of multiplications required is:
      output dimensional volume * filter dimensional volume = (28 * 28 * 32) * (5 * 5 * 192) = 120 M

      <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/6963dc68-c04c-44df-a931-057503efe083.png" width="700" />

    * 120 M multiplications is still computationally expensive using modern-day computers
  * Using a 1 x 1 Conv we can reduce 120 M to just 12 M computations
* Using 1 X 1 Conv reduces the computational cost as follows:
  * Let's use the input shape again from last example (28, 28, 192)
  * We then convolve the input with 16 (1 x 1 (x 192)) convolution to produce a shape of 28 x 28 x 16 dimensional volume (reducing the dimensions to 16 filters from 192)
  * Then apply 32 (5 x 5 Convolution) producing a shape (28, 28, 32)
  * Now lets calculate the number of multiplications:
    * For the first Conv: 28 * 28 * 16 * 1 * 1 * 192 = 2.5 M
    * For the second Conv: 28 * 28 * 32 * 5 * 5 * 16 = 10 M
    * So the total number is 12.5 M approx, which is a factor 10 lower than without the 1 x 1 Conv!
  
    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/11d7b190-2e57-4211-8b85-4ebece233e85.png" width="700" />

  * A 1 x 1 Conv here is called 'Bottleneck'
  * It turns out that the 1 x 1 Conv won't hurt the performance, as long as you use it within reason

## Inception Network
* Inception module, dimensions reduction version:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/1681c979-d857-463f-b4f0-798aa3b70b7d.png" width="700" />

* Inception network (GooLeNet)
  * The Inception network consist of repeated concatenated blocks of the Inception module:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/66c01c7d-1689-4acc-8237-569b623a4738.png" width="700" />

  * The paper mentions 2 additional side branches that also terminate in a 'softmax' function, that takes some hidden layer and tries to use that to make a prediction
    * Helps ensure that the features computed, even at intermediate layers, are not 'too bad' for predicting  the output cost of an image, which appears to have a regularizing effect on the inception network i.e.,  prevents the network from overfitting

      <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/dba69b80-7613-44e2-83e8-ae8b9c93e421.png" width="700" />

* The name inception was taken from a meme image which was taken from Inception movie

## MobileNet
* Motivation for MobileNet
  * Low computation cost at deployment
  * Useful for mobile and embedded vision applications
  * Key idea: Normal vs. Depthwise Separate convolutions
* Normal Convolution:
  * Normaly, we just do a volume Conv operation on a f x f x n<sub>c</sub> block of filter

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/9d136e5a-977c-4742-a6b7-cf50b8d51db6.png" width="700" />

* Depthwise Separable Convolution
* Consists of 2 steps/parts, a depthwise Conv followed by a pointwise Conv

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/79f0ce5a-e37c-4a18-8141-6031d011202a.png" width="700" />

  * Depthwise Conv: does a conv computation separately on each channel, then stack their outputs 

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/e243db93-7713-4f04-a86c-8edfb4319a18.png" width="700" />

  * Pointwise Conv: works similarly to the normal convolution operation but with 1x1 size filter

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/4c9a2dae-c430-4994-a886-b771f0fe9607.png" width="700" />

* Cost summary (with 6x6x3 input, 5 3x3x3 filters and 4x4x5 output)
  * Normal Conv: 2160
  * Depthwise Separable Conv
    * Depthwise: 432
    * Pointwise: 240
    * Total: 672
      672 / 2160 = 0.31 => ~2/3 time cheaper than the normal Conv cost!
* In the general case: Cost_normal / Cost_depth = 1/n<sub>c</sub>' * 1/(f<sup>2</sup>)
  * Example: if the output has 512 channels and the filter size is 3, then using depth separable conv may reduce 1/512 + 1/9 ~= 1/9 ~= 10 times cheaper than using the normal one

## MobileNet Architecture
* The idea of MobileNet: rather than use the expensive conv operation, we can use the Depthwise Separable Conv (DS Conv) with the depthwise and pointwise sub-functions

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/a89fa4d9-7d24-431b-82e4-240233e353fb.png" width="700" />

* MobileNet v1:
  * `Input => 13 x DS Conv => Pool => FC => Softmax => Output`
    * [Howard et al. 2017, MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
* MobileNet v2:
  * `Input => Residual Connection[Expansion => Depthwise => Projection (Pointwise)] x 17 => Pool => FC => Softmax => Output`
  * The Residual Connection block is also called the Bottleneck
    * [Sandler et al. 2019, MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

* MobileNet v2 BottleNeck:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/dd00f0cd-77ee-4dbe-afa3-02b45a20e110.png" width="700" />

  * As the output of Pointwise is as same as the input of the bottleneck, the Pointwise function is also called the Projection
  * The reason of using Bottleneck:
    * The Expansion increases the size of representation within the bottleneck, which enables the learning of richer, more complex features
    * The Projection keeps the memories small throughout the network, making this NN more suitable for mobile devices
    * MobileNet v2 has better performance than v1 by using a small amount of compute & memory resources

## EfficientNet
* Helps scale up or down NNs (MobileNet) for a specific device (different brand of mobile phone with different amounts of computational budget)
  * [Tan and Le, 2019, EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
* Architecture:
  * Baseline NN architecture with image resolution `r`, depth `d`, weight `w`
  * We can modify `r`, `d`, and `w` to scaling the NN for fitting the particular device
  * We can even use compount scaling to modify all three parameters at same time

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/7bc34a11-cd82-4d2f-aea7-6fadcd5dc4f7.png" width="700" />

* What's a good choice of modifying these parameters? What's a good trade-off When modify a parameter but fixing the other two?
* You can looks up for some open-source implementations of EfficientNet to help choose a good trade-off between r, d, and w

## Using Open-Source Implementation
* We have learned a lot of NNs and ConvNets architectures
* It turns out that a lot of these NN are difficult to replicate, because there are often details that may not presented on its papers e.g., 
  * Learning rate decay
  * Parameter tuning
* A lot of deep learning researchers are opening sourcing their code onto internet on sites like Github
* If you see a research paper and you want to build over it, the first thing you should do is to look for an open source implementation for this paper
* Some advantage of doing this is that you might download the network implementation along with its parameters/weights (transfer learning). The author might have used multiple GPUs and spent some weeks to reach this result and its right in front of you after you download it

## Transfer Learning
* If you are using a specific NN architecture that has been trained before, you can use this pretrained parameters/weights instead of random initialization to solve your problem
* It can help you boost the performance of the NN
* The pretrained models might have trained on a large datasets like ImageNet, Ms COCO, or pascal and took a lot of time to learn those parameters/weights with optimized hyperparameters, which can save you a lot of time
* Example 1:
  * Lets say you have a cat classification problem which contains 3 classes Tigger, Misty and neither
  * You don't have much a lot of data to train a NN on these images
  * Andrew recommends to go online and download a good NN with its weights, remove the softmax activation layer and put your own softmax one and make the network learn only the new layer while other layer weights are fixed/frozen

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/a91b77ca-9269-440b-8a12-684f6ec7ebbd.png" width="700" />

    * Frameworks have options to make the parameters frozen in some layers using `trainable = 0` or `freeze = 0`
  * One of the tricks that can speed up your training, is to run the pretrained NN without final softmax layer and get an intermediate representation of your images and save them to disk. And then use these representations to a shallow NN network, which can save you the time needed to run an image through all the layers
  * Its like converting your images into vectors
* Example 2:
  * What if in the last example you have a lot of pictures (dataset) for your cats
  * One thing you can do is to freeze few layers from the beginning of the pretrained network and learn the other weights in the network

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/8d73829e-2d68-4459-aec6-27b491812d3a.png" width="700" />

  * Some other idea is to throw away the layers that aren't frozen and developed your own layers there
* Example 3:
  * If you have enough data, you can fine-tune all the layers in your pretrained network but don't random initialize the parameters, leave the learned parameters as it is and learn from there

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/0c6e4815-b4de-4c02-b167-9fc19e2144f4.png" width="700" />

## Data Augmentation
* If data is increased, your deep NN will perform better. Data augmentation is one of the techniques that deep learning uses to increase the performance of a deep NN
* The majority of computer vision applications requires more data
* Some data augmentation methods that are used for computer vision tasks include:
  * Mirroring (flip horizontal or vertical)
  * Random cropping
    * An issue with this technique is that you might take a wrong crop
    * The solution to this would be to make your crops big enough
  * Rotation
  * Shearing
  * Local warping

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/2ea462dd-5983-4516-b683-e6a315f3ddeb.png" width="700" />

  * Color shifting
    * For example, we add to R, G, and B some distortions that will make the image identified as the same for the human but is different for the computer
    * In practice the added value are pulled from some probability distributions
    * Makes your algorithm more robust to changes of colors in images
    * There is an algorithm which is called <b>PCA color augmentation</b> and decides the shifts needed automatically (detailed are in the AlexNet paper)

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/65aa7a58-6863-45c2-bd57-74232defef2e.png" width="700" />

* Implementing distortions during training:
  * You can use a different CPU thread to create distorted mini batches while you are training your NN
* Data Augmentation also has some hyperparameters
  * A good place to start is to find an open source data augmentation implementation and then use it or fine tune these hyperparameters

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/dd800cf9-e6ed-44a4-ad71-63905a08777a.png" width="700" />

## State of Computer Vision
* For a specific problem we may have either little data or a lot of data
  * E.g., speech recognition problems have a big amount of data, while image recognition/classification has a medium amount of data; the object detection has a small amount of data
* If your problem has a large amount of data, researchers tend to:
  * Use simpler algorithms
  * Less hand engineering
* If you don't have that much data, researcher tend to:
  * Try more hand engineering for the problem ("Hacks") e.g., choosing a more complex NN architecture
* Two sources of knowledge:
  * Labeled data (x,y)
    * Use for supervised learning => building up the learning system
  * Hand engineered features/network architecture/other components:
    * Very difficult, very skillful task that requires a lot of insight
* Because we haven't got that much data in a lot of computer vision problems (but the latest problems are getting more difficult and complex), we rely more on hand engineering
  * As object detection has less data than image recognition, a more complex NN architectures will be applied for this topic

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/c3f8ef9d-b039-4230-9b92-76f2b2488d35.png" width="700" />

* Tips for doing well on benchmarks/winning competitions:
  * Ensembling
    * Train several networks independently and average their outputs (y_hat); requires lot of memory due to the requirement of storing each NN
    * After you decide the best architecture for your problem, initialize some of that randomly and train them independently
    * This can give you a push by 2%
    * But this will slow down your production by the number of the ensembles
    * People use this in competitions but few uses this in a real production
  * Multi-crop at test time
    * Run classifier on multiple versions of test versions (i.e. applying augmentation) and average results
    * There is a technique called 10 crops that uses this
    * This can give you a better result in the production

      <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/eead6ec1-4a9f-4c93-83f9-9b962325d9b2.png" width="700" />

* Use open source code
  * Use architectures of networks published in the literature
  * Use open source implementations if possible
  * Use pretrained models and fine-tune on your dataset
