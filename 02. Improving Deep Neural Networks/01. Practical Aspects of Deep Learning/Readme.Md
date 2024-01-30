# Practical Aspects of Deep Learning

## Train / Dev / Test sets
* It's impossible to get all your hyperparameters right on a new application from the first time
* So the idea is you go through the loop: Idea ==> Code ==> Experiment
* You have to go through the loop many times to figure out your hyperparameters
* Your data will be split into three parts:
  * Training set (Has to be the largest set)
  * Cross validation set / Development or "dev" set
  * Testing set
* You will try to build a model usingg the training set, then try to optimize hyperparameters on the dev set as much as possible. Then after your model is ready, you try and evaluate the testing set
*  So the trend on the ratio of splitting the models (what matters is that the absolute number of examples in the test set is high enough):
  * If size of the dataset is 100 to 1.000.000 ==> 60/20/20
  * If size of the dataset is over 1.000.000 ==> 98/1/1 or 99.5/0.25/0.25
* The dev/test set just needs to be big enough for evaluating different algorithm choices and quickly decides which one is better => does not need the whole 20% dataset for that (in a large dataset case)
* The trend now gives the training data the biggest sets
* Make sure the dev and test set are coming from the same distribution (i.e., acquired in the same way e.g., in  binary classification, use camereras that yield identical or highly simular results).
  * For example, if cat training pictures is from the web and the dev/test pictures are from users cell phone they will mismatch. It is better to make sure that dev and test set are from the same distribution.
* The dev set rule is to try them on some of the good models you've created.
* It's OK to only have a dev set without a testing set. But a lot of people in this case call the dev set as the test set. A better terminology is to call it a dev set as its used in the development.

## Bias / Variance
* Bias / variance techniques are easy to learn, but difficult to master
* So here the explanation of Bias / Variance:
  * If your model is underfitting (logistic regression of non linear data) it has a "high bias"
  * If your model is overfitting then it has a "high variance"
  * Your model will be alright if you balance the bias / variance

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/a5117d29-3f14-4652-8252-48dada6a6ba8.png" width="700" />

* Another idea to get the bias / variance if you don't have a 2D plotting mechanism:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/c1317dcb-b358-47cc-beaa-24c504c10d18.png" width="700" />

* These assumptions based on the idea that a human has 0% error in classifying images. If the problem that you're trying to solve isn't like that e.g., could be dealing with blurry images that even a human has difficulties with classifying, you'll need to use a different human error as baseline.
