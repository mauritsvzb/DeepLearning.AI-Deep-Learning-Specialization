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
    * if 5 of 100 are dogs then training your classifier to do better on dogs will decrease your error up to 9.5% (called ceiling), which can be too little
    * if 50 of 100 are dogs then you could decrease your error up to 5%, which is reasonable and you should work on that
* Based on the last example, error analysis helps you to analyze the error before taking an action that could take lot of time with no need.
* Sometimes, you can evaluate multiple error analysis ideas in parallel and choose the best idea. Create a spreadsheet to do that and decide, e.g.:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/b2fbad86-4c34-4426-9ce8-4f47c874c220.png" width="700" />

* In the last example you will decide to work on great cats or blurry images to improve your performance.
* This quick counting procedure, which you can often do in, at most, small numbers of hours can really help you make much better prioritization decisions, and understand how promising different approaches are to work on.

## Cleaning up incorrectly labeled data
