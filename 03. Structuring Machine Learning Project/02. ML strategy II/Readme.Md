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

Example 1:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/60b160ab-3af6-42b6-8fd0-ee23367b3cda.png" width="700" />

Example 2:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/5ade867b-e6d1-4d8a-9c52-ce5c1152b9bb.png" width="700" />

## Bias and Variance with Mismatched Data Distributions

