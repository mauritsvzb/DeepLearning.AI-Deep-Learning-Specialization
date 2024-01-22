# Neural Networks Basics
Learn to set up a machine learning problem with a neural network mindset. Learn to use vectorization to speed up your models

## Binary classification
* Mainly he is talking about how to do a logistic regression to make a binary classifier.

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/4ec78558-30b1-4ec6-9a04-827db7518ed8.png" width="600" />

* He talked about an example of knowing if the current image contains a cat or not.
* Here are some notations:
    * M is the number of training vectors
    * N<sub>x</sub> is the size of the input vector
    * N<sub>y</sub> is the size of the output vector
    * X(1) is the first input vector
    * Y(1) is the first output vector
    * X = [x(1) x(2).. x(M)]
    * Y = (y(1) y(2).. y(M))
* We will use python in this course.
* In NumPy we can make matrices and make operations on them in a fast and reliable time.

## Logistic regression
* Algorithm is used for classification algorithm of 2 classes
* Equations:
   * Simple equation: y = w * x + b
   * If x is a vector: y = w<sup>T</sup> * x + b
   * If we need y to be in between 0 and 1 (probability): y = sigmoid(w<sup>T</sup> * x + b)
   * In some notations this might be used: y = sigmoid(w<sup>T</sup> * x)
      * While b is w<sub>0</sub> of w and we add x<sub>0</sub> = 1. But we won't use this notation in the course (Andrew said that the first notation is better)
* In binary classification Y has to be between 0 and 1
* In the last equation, w is a vector of N<sub>x</sub> and b is a real number

## Logistic regression cost function
* First loss function would be the square root error: L(ŷ, y) = 1/2 (ŷ - y)<sup>2</sup>
   * But we won't use this notation because it leads us to optimization problem which is non convex, means it contains multiple local optima instead of 1 global optimum
* This is the function that we will use: L(ŷ, y) = - (y * log(ŷ) + (1 - y) * log(1 - ŷ))
* To explain the last function lets see:
   * if y = 1 ==> L(ŷ, 1) = - log(ŷ) ==> we want ŷ to be the largest ==> ŷ biggest value is 1
   * if y = 0 ==> L(ŷ, 0) = - log(1 - ŷ) ==> we want 1 - ŷ to be the largest ==> ŷ to be smaller as possible because it can only has 1 value
* Then the Cost function will be: J(w,b) = (1/m) * Sum( L (ŷ[i], y[i]))
* The loss function computes the error for a single training example; the cost function is the average of the loss functions of the entire training set

## Gradient Descent
* We want to predict w and b that minimize the cost function
* Our cost function is convex ie 1 global optimum (= minimum)
* First we initialize `w` and `b` to 0,0 or initialize them to a random value in the convex function and then try to improve the values the reach minimum value
* In Logistic regression people always use 0,0 instead of random  (it doesn't matter where  you start, you'll always arrive at the global optimium or very close to it)
* The gradient decent algorithm repeats: `w = w - alpha * dw`, where `alpha` is the learning rate (i.e., how big of a step in horizontal plane) and `dw` is the derivative of `w` (Change to w OR slope of 'w' in vertical plane).
* Looks like greedy algorithms. the derivative give us the direction to improve our parameters (i.e., direction of largest descent)
* The actual equations we will implement:
   * `w = w - alpha * d(J(w,b) / dw)` (how much the function slopes in the w direction)
   * `b = b - alpha * d(J(w,b) / db)` (how much the function slopes in the b direction)
<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/9c5e09d1-6a5d-4a38-a319-b37d2e262d60.png" width="600" /><img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/b03206bb-07dd-451e-a423-dbcd831984ea.png" width="600" />

## Derivatives
* We will talk about some of required calculus
* You don't need to be a calculus geek to master deep learning but you'll need some skills from it
* Derivative of a linear line is its slope
   * ex. f(a) = 3a d( f(a) ) / d(a) = 3
   * if a = 2 then f(a) = 6
   * if we move a a little bit a = 2.001 then f(a) = 6.003 means that we multiplied the derivative (slope) to the moved area and added it to the last result

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/67e31110-f7f8-4044-920c-72fe6def61c9.png" width="600" />

## More Derivatives examples
* f(a) = a<sup>2</sup> ==> d(f(a))/d(a) = 2a
* a = 2 ==> f(a) = 4
* a = 2.0001` ==> f(a) = 4.0004 approx
* f(a) = a<sup>3</sup> ==> d( f(a) ) / d(a) = 3 * a<sup>2</sup>
* f(a) = log(a) ==> d( f(a) ) /d (a) = 1/a
* To conclude, Derivative is the slope and slope is different in different points in the function thats why the derivative is a function

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/8aca22ce-7a6c-4292-b990-09cc80dbaa0c.png" width="600" />

## Computation graph
* Its a graph that organizes the computation from left to right
   * <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/3e924bac-01af-45bd-9617-a81d20e04063.png" width="600" />

## Derivatives with a Computation Graph
* Calculus chain rule says: If `x -> y -> z` (x effect y and y effects z) Then `d(z) / d(x) = d(z) / d(y) * d(y) / d(x)`
* The video illustrates a big example
   * <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/e8f480cf-33b3-46d4-94c6-7e2891714fdc.png" width="600" />
* We compute the derivatives on a graph from right to left and it will be a lot more easier
* `dvar` means the derivatives of a final output variable with respect to various intermediate quantities

## Logistic Regression Gradient Descent
In the video he discussed the derivatives of gradient decent example for one sample with two features `x1` and `x2`
* <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/fa9f581b-981c-4be4-9377-19280f4c9fe5.png" width="600" />

## Gradient Descent on m Examples
* Lets say we have these variables:

`	X1                  Feature
	X2                  Feature
	W1                  Weight of the first feature.
	W2                  Weight of the second feature.
	B                   Logistic Regression parameter.
	M                   Number of training examples
	Y(i)                Expected output of i`

* So we have:

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/1722c109-4adf-482a-95c8-991dc797d5de.png" width="600" />

* Then from right to left we will calculate derivations compared to the result:

```python	
   d(a)  = d(l)/d(a) = -(y/a) + ((1-y)/(1-a))
	d(z)  = d(l)/d(z) = a - y
	d(W1) = X1 * d(z)
	d(W2) = X2 * d(z)
	d(B)  = d(z)
```
* From the above we can conclude the logistic regression pseudo code:

```python	
   J = 0; dw1 = 0; dw2 =0; db = 0;                 # Devs.
	w1 = 0; w2 = 0; b=0;							# Weights
	for i = 1 to m
		# Forward pass
		z(i) = W1*x1(i) + W2*x2(i) + b
		a(i) = Sigmoid(z(i))
		J += -(Y(i)*log(a(i)) + (1-Y(i))*log(1-a(i)))

		# Backward pass
		dz(i) = a(i) - Y(i)
		dw1 += dz(i) * x1(i)
		dw2 += dz(i) * x2(i)
		db  += dz(i)
	J /= m
	dw1/= m
	dw2/= m
	db/= m

	# Gradient descent
	w1 = w1 - alpha * dw1
	w2 = w2 - alpha * dw2
	b = b - alpha * db
```
*  The above code should run for some iterations to minimize error.

* So there will be two inner loops to implement the logistic regression.

* Vectorization is so important on deep learning to reduce loops. In the last code we can make the whole loop in one step using vectorization!

## Vectorization
* `for` loops will make you wait a long time to get a result due to iterating over rows and columns (aka 'indexing'. Vectorization circumvents the need for `for' loops and are much faster due to parallelization of computations; 'for' loops compute in series i.e., one computation at a time
* NumPy library (dot) function is using vectorization by default.
* The vectorization can be done on CPU or GPU thought the SIMD (Single Instruction Multiple Data) operation. But its faster on GPU.
* Whenever possible avoid for loops (in general, not just in python)
* Most of the NumPy library methods are vectorized versions.

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/f9ba8b6e-ec4c-49e8-acd5-37eb0060bb32.png" width="600" />

## Vectorizing Logistic Regression
* We will implement Logistic Regression using a for loop then without any for loop.

* As an input we have a matrix X (= [Nx, m]) and a matrix Y = ([Ny, m])

* We will then compute at instance [z1,z2...zm] = W' * X + [b,b,...b]. This can be written in python as: `Z = np.dot(W.T,X) + b`    # Vectorization, then broadcasting, Z shape is (1, m) A = 1 / 1 + np.exp(-Z)   # Vectorization, A shape is (1, m)

* Vectorizing Logistic Regression's Gradient Output: dz = A - Y                  # Vectorization, dz shape is (1, m) dw = np.dot(X, dz.T) / m    # Vectorization, dw shape is (Nx, 1) db = dz.sum() / m           # Vectorization, dz shape is (1, 1)

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/b50e6c1f-bbc5-4c49-943a-4ac4ac044edc.png" width="600" />

## Vectorizing Logistic Regression's Gradient Output

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/5743e0e8-5f8e-4296-b9e6-6005b50811ca.png" width="600" />
<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/36156e10-5263-4811-b7f8-0e4274e2f662.png" width="600" />

## Broadcasting in Python
* In NumPy, `obj.sum(axis = 0)` sums the columns while `obj.sum(axis = 1)` sums the rows.
* In NumPy, `obj.reshape(1,4)` changes the shape of the matrix by broadcasting the values.
* Reshape is cheap in calculations so put it everywhere you're not sure about the calculations.
* Broadcasting works when you do a matrix operation with matrices that doesn't match for the operation, in this case NumPy automatically makes the shapes ready for the operation by broadcasting the values.
* In general principle of broadcasting. If you have an (m,n) matrix and you add(+) or subtract(-) or multiply(*) or divide(/) with a (1,n) matrix, then this will copy it m times into an (m,n) matrix. The same with if you use those operations with a (m , 1) matrix, then this will copy it n times into (m, n) matrix. And then apply the addition, subtraction, and multiplication of division element wise.

<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/33c33464-78a1-4c43-8b11-1003061a2616.png" width="600" />
<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/ceaf8c5d-6a58-4b14-ac14-e21d4a212806.png" width="600" />
<img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/a24e54b0-c163-4f32-bcde-f467e6c4f7c1.png" width="600" />

In sum: 
* Broadcasting optimizes Python code for faster execution
* It allows performing operations on matrices of different shapes efficiently
* Copying smaller matrices or vectors before element-wise operations is a key principle.
* The technique improves code efficiency and conciseness, eliminating the need for explicit for-loops

##  Notes on Python and NumPy
* Some tricks to eliminate all the strange bugs in the code:
	* If you didn't specify the shape of a vector, it will take a shape of (m,) and the transpose operation won't work. You have to reshape it to (m, 1)
	* Try to not use the rank one matrix in ANN
	* Don't hesitate to use assert(a.shape == (5,1)) to check if your matrix shape is the required one.
	* If you've found a rank one matrix try to run reshape on it.
* Jupyter / IPython notebooks are so useful library in python that makes it easy to integrate code and document at the same time. It runs in the browser and doesn't need an IDE to run.
* To open Jupyter Notebook, open the command line and call: jupyter-notebook It should be installed to work.
* To Compute the derivative of Sigmoid:
	```python
s = sigmoid(x)
ds = s * (1 - s)       # derivative  using calculus
```
* To make an image of (width,height,depth) be a vector, use this:
	* v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)  #reshapes the image.
* Gradient descent converges faster after normalization of the input matrices.
