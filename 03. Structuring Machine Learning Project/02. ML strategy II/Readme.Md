# ML Strategy II

## Goal
Develop time-saving error analysis procedures to evaluate the most worthwhile options to pursue and gain intuition for how to split your data and when to use multi-task, transfer, and end-to-end deep learning.

## Learning Objectives
* Describe multi-task learning and transfer learning
* Recognize bias, variance and data-mismatch by looking at the performances of your algorithm on train/dev/test sets

## Carrying Out Error Analysis
* Error analysis - process of manually examining mistakes that your algorithm is making. It can give you insights into what to do next e.g.,
  * In the cat classification example, if you have 10% error on your dev set and you want to decrease the error.
  * You discovered that some of the mislabeled data are dog pictures that look like cats. Should you try to make your cat classifier do better on dogs (this could take some weeks)?
  * Error analysis approach:
    * Get 100 mislabeled dev set examples at random
    * Count up how many are dogs
    * if 5 of 100 are dogs then training your classifier to do better on dogs will decrease your error up to 0.5% (called ceiling) to 9.5%, which can be too little
    * if 50 of 100 are dogs then you could decrease your error up to 5%, which is reasonable and you should work on that
* Based on the last example, error analysis helps you to analyze the error before taking an action that could take lot of time with no need.
* Sometimes, you can evaluate multiple error analysis ideas in parallel and choose the best idea. Create a spreadsheet to do that and decide, e.g.:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/b2fbad86-4c34-4426-9ce8-4f47c874c220.png" width="700" />

* In the last example you will decide to work on great cats or blurry images to improve your performance.
* This quick counting procedure, which you can often do in, at most, small numbers of hours can really help you make much better prioritization decisions, and understand how promising different approaches are to work on.

## Cleaning up incorrectly labeled data
* DL algorithms are quite robust to random errors in the training set but less robust to systematic errors; it's OK to go and fix these labels if you can
* If you want to check for mislabeled data in dev/test set, you should also try error analysis with the mislabeled column e.g., 

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/45bf1476-7637-47d4-99d4-bfe82c4b5204.png" width="700" />

  * Then:
    * If overall dev set error: 10%
      * Then errors due to incorrect data: 0.6%
      * Then errors due to other causes: 9.4%
    * Then you should focus on the 9.4% error rather than the incorrect data.
* Consider these guidelines while correcting the dev/test mislabeled examples:
  * Apply the same process to your dev and test sets to make sure they continue to come from the same distribution
  * Consider examining examples your algorithm got right as well as ones it got wrong. (Not always done if you reached a good accuracy)
  * Train and (dev/test) data may now come from a slightly different distributions
  * It's very important to have dev and test sets to come from the same distribution, but it could be OK for a train set to come from slightly other distribution

## Build your first system quickly, then iterate
* The steps you take to make your deep learning project:
  * Setup dev/test set and metric
  * Build initial system quickly
  * Use Bias/Variance analysis & Error analysis to prioritize next steps

## Training and Testing on Different Distributions
* A lot of teams are working with deep learning applications that have training sets that are different from the dev/test sets due to the hunger of deep learning to data
* There are some strategies to follow when the training set distribution differs from dev/test sets distribution
  1. (not recommended): shuffle all the data together and extract randomly training and dev/test sets
     * Advantages: all the sets now come from the same distribution
     * Disadvantages: the other (real world) distribution will only constitute a very small fraction of the data in the dev/test sets which causes the model to primarily train on the data from the wrong distribution, and this is not what you want
  2. Put all the data from the 'wrong' distribution only in the training set, and add (some of) the data from the target distribution to the dev/test set
     * Advantages: the distribution you care about is your target now
     * Disadvantage: the distributions in training and dev/test sets are now different. But you will get a better performance over a long time

Example 1: cat identifier

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/60b160ab-3af6-42b6-8fd0-ee23367b3cda.png" width="700" />

Example 2: speech recognition

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/5ade867b-e6d1-4d8a-9c52-ce5c1152b9bb.png" width="700" />

## Bias and Variance with Mismatched Data Distributions
* Bias and Variance analysis changes when training and Dev/test set is from the different distribution
* Example: the cat classification example. Suppose you've worked in the example and reached this:
  * Human error: 0%
  * Train error: 1%
  * Dev error: 10%
* In this example, you'll think that this is a variance problem, but because the distributions aren't the same you can't tell for sure. Because it could be that train set was easy to train on, but the dev set was more difficult.
* To solve this issue we create a new set called train-dev set as a random subset of the training set (so it has the same distribution) and we get:
   * Human error: 0%
   * Train error: 1%
   * Train-dev error: 9%
   * Dev error: 10%
* Now we are sure that this is a high variance problem (the model doesn't generalize enough)
* Suppose we have a different situation:
   * Human error: 0%
   * Train error: 1%
   * Train-dev error: 1.5%
   * Dev error: 10%
* In this case we have something called Data mismatch problem
* Conclusions:
  1. Human-level error (proxy for Bayes error)
  2. Train error
     * Calculate avoidable bias = training error - human level error
     * If the difference is big then it's avoidable bias problem and you should use a strategy to counter high bias
  3. Train-dev error
     * Calculate variance = training-dev error - training error
     * If the difference is big then it's high variance problem and you should use a strategy for solving that
  4. Dev error
     * Calculate data mismatch = dev error - train-dev error
     * If difference is much bigger then train-dev error, then it's a data mismatch problem
  5. Test error
     * Calculate degree of overfitting to dev set = test error - dev error
     * If the difference is big (positive) then maybe you need to find a bigger dev set
       * dev set and test set come from the same distribution, so the only way for there to be a huge gap here, for it to do much better on the dev set than the test set, is if you somehow managed to overfit the dev set
* Unfortunately, there aren't many systematic ways to deal with data mismatch, but there are some things to try about this in the next section

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/772c5f84-f1be-4293-b314-d4e4f81e9d9e.png" width="700" />

## Addressing Data Mismatch
* There aren't completely systematic solutions to this, but there some things you could try:
  * Carry out manual error analysis to try to understand the difference between training and dev/test sets
  * Make training data more similar, or collect more data similar to dev/test sets
    * If your goal is to make the training data more similar to your dev set, one of the techniques you can use Artificial Data Synthesis that can help you make more training data
      * Combine some of your training data with something that can convert it to the dev/test set distribution
      * Examples:
        * Combine normal audio with car noise to get audio with car noise example
        * Generate cars using 3D graphics in a car classification example
    * Be cautious and bear in mind whether or not you might be accidentally simulating data only from a tiny subset of the space of all possible examples because your NN might overfit these generated data (like particular car noise or a particular design of 3D graphics cars)

## Transfer Learning
* Apply the knowledge you took in a task A and apply it to another task B
* For example, you have trained a cat classifier with a lot of data, you can use the part of the trained NN it to solve x-ray classification problem
* To do transfer learning, delete the last layer of NN and it's weights and:
   1. Option 1: If you have a small data set - keep all the other weights as a fixed weights. Add a new last layer(-s) and initialize the new layer weights and feed the new data to the NN and learn the new weights
   2. Option 2: If you have enough data, you can retrain all the weights
* Option 1 and 2 are called fine-tuning and training on task A called pretraining
* When transfer learning make sense, if:
  * Task A and B have the same input X (e.g. image, audio)
  * You have a lot of data for the task A you are transferring from and relatively less data for the task B your transferring to (assuming task B is what you want to do well on)
  * Low level features from task A could be helpful for learning task B

## Multi-task learning
* Whereas in transfer learning, you have a sequential process where you learn from task A and then transfer that to task B. In multi-task learning, you start off simultaneously, trying to have one neural network do several things at the same time. And then each of these tasks helps hopefully all of the other tasks.
* Example:
  * You want to build an object recognition system that detects pedestrians, cars, stop signs, and traffic lights (image has multiple labels)
  * Then Y shape will be (4,m) because we have 4 classes and each one is a binary one
* Then `Cost = (1/m) * sum(sum(L(y_hat(i)_j, y(i)_j)))`, i = 1..m, j = 1..4, where
  `L = - y(i)_j * log(y_hat(i)_j) - (1 - y(i)_j) * log(1 - y_hat(i)_j)`
* In the last example you could have trained 4 neural networks separately but if some of the earlier features in neural network can be shared between these different types of objects, then you find that training one neural network to do four things results in better performance than training 4 completely separate neural networks to do the four tasks separately
* Multi-task learning will also work if y isn't complete for some labels. For example:
   ```
   Y = [1 ? 1 ...]
       [0 0 1 ...]
       [? 1 ? ...]
   ```
  * And in this case it will do good with the missing data, just the loss function will be different:
    `Loss = (1/m) * sum(sum(L(y_hat(i)_j, y(i)_j)` for all j which y(i)_j != ?))
* Multi-task learning makes sense:
  * Training on a set of tasks that could benefit from having shared lower-level features
  * Usually, amount of data you have for each task is quite similar
  * Can train a big enough network to do well on all the tasks
* If you can train a big enough NN, the performance of the multi-task learning compared to splitting the tasks is better
* Today transfer learning is used more often than multi-task learning

## What is end-to-end deep learning?
* Some systems have multiple stages to implement. An end-to-end deep learning system implements all these stages with a single NN
* Example 1:
  * Speech recognition system:
    ```
    Audio ---> Features --> Phonemes --> Words --> Transcript    # non-end-to-end system
    Audio ---------------------------------------> Transcript    # end-to-end deep learning system
    ```
  * End-to-end deep learning gives data more freedom, it might skip the use of 'features' or 'phonemes' for training
* To build the end-to-end deep learning system that works well, we need a big dataset (more data then in non end-to-end system). If we have a small dataset the ordinary implementation could work just fine.
* Example 2:
  * Face recognition system:
    ```
    Image ---------------------> Face recognition    # end-to-end deep learning system
    Image --> Face detection --> Face recognition    # deep learning system - best approach for now
    ```

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/4004afba-85b3-47e0-b84a-a6579a14578e.png" width="700" />

  * In practice, the best approach is the second one for now
  * In the second implementation, it's a two steps approach where both parts are implemented using deep learning
  * It's working well because it's harder to get a lot of pictures with people in front of the camera than getting faces of people and compare them
  * In the second implementation at the last step, the NN takes two faces as an input and outputs if the two faces are the same person or not
* Example 3:
  * Machine translation system:
    ```
    English --> Text analysis --> ... --> French    # non-end-to-end system
    English ----------------------------> French    # end-to-end deep learning system - best approach
    ```
  * Here end-to-end deep leaning system works better because we have enough data to build it
* Example 4:
  * Estimating child's age from the x-ray picture of a hand:
    ```
    Image --> Bones --> Age    # non-end-to-end system - best approach for now
    Image ------------> Age    # end-to-end system
    ```
    * In this example non-end-to-end system works better because we don't have enough data to train end-to-end system

## Whether to use End-to-end Deep Learning
* Pros of end-to-end deep learning:
  *  Let the data speak. By having a pure machine learning approach, your NN learning input from X to Y may be more able to capture whatever statistics are in the data, rather than being forced to reflect human preconceptions
  * Less hand-designing of components needed.
* Cons of end-to-end deep learning:
  * May need a large amount of data
  * Excludes potentially useful hand-design components (it helps more on the smaller dataset)
  * Applying end-to-end deep learning:
    * Key question: Do you have sufficient data to learn a function of the complexity needed to map x to y?
    * Use ML/DL to learn some individual components
    * When applying supervised learning you should carefully choose what types of X to Y mappings you want to learn depending on what task you can get data for

