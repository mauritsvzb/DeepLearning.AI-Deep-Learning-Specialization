# Goal and Learning Objectives
Discover recurrent neural networks, a type of model that performs extremely well on temporal data, and several of its variants, including LSTMs, GRUs and Bidirectional RNNs
* Define notation for building sequence models
* Describe the architecture of a basic RNN
* Identify the main components of an LSTM
* Implement backpropagation through time for a basic RNN and an LSTM
* Give examples of several types of RNN
* Build a character-level text generation model using an RNN
* Store text data for processing using an RNN
*  Sample novel sequences in an RNN
* Explain the vanishing/exploding gradient problem in RNNs
* Apply gradient clipping as a solution for exploding gradients
* Describe the architecture of a GRU
* Use a bidirectional RNN to take information from two points of a sequence
* Stack multiple RNNs on top of each other to create a deep RNN
* Use the flexible Functional API to create complex models
* Generate your own jazz music with deep learning
* Apply an LSTM to a music generation task

# Recurrent Neural Networks (RNNs)
## Why Sequence Models?
* Sequence Models like RNN have greatly transformed learning on sequences in the past few years
* Examples of sequence data in applications:
  * Speech recognition (sequence to sequence):
    * X: wave sequence
    * Y: text sequence
  * Music generation (one to sequence):
    * X: nothing or an integer
    * Y: wave sequence
  * Sentiment classification (sequence to one):
    * X: text sequence
    * Y: integer rating from one to five
  * DNA sequence analysis (sequence to sequence):
    * X: DNA sequence
    * Y: DNA Labels
  * Machine translation (sequence to sequence):
    * X: text sequence (in one language)
    * Y: text sequence (in other language)
  * Video activity recognition (sequence to one):
    * X: video frames
    * Y: label (activity)
  * Name entity recognition (sequence to sequence):
    * X: text sequence
    * Y: label sequence
    * Can be used by seach engines to index different type of words inside a text
* All of these problems with different input and output (sequence or not) can be addressed as supervised learning with label data X, Y as the training set.

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/342262a6-81c8-4c9e-9bbb-ab4bff4b620c.png" width="700" />

## Notation
* In this section we will discuss the notations that we will use through the course
* Motivating example:
  * Named entity recognition example:
    * X: "Harry Potter and Hermoine Granger invented a new spell"
    * Y:    1     1     0    1        1        0     0  0    0
    * Both elements have a shape of 9. 1 denotes (part of) a name, whereas 0 denotes no (part of a) name
  * We will index the first element of x by x<sup><1></sup>, the second x<sup><2></sup> and so on, e.g., 
    * x<sup><1></sup> = Harry
    * x<sup><2></sup> = Potter
  * Similarly, we will index the first element of y as y<sup><1></sup>, the second y<sup><2></sup>, etc
    * y<sup><1></sup> = 1
    * y<sup><2></sup> = 2
  * T<sub>x</sub> is the size of the input sequence; T<sub>y</sub> output sequence
    * T<sub>x</sub> = T<sub>y</sub> = 9 in the last example, although they can be different in other problems
  * x<sup>(i)<t></sup> refers to the t<sup>th</sup> element in the input sequence of the i<sup>th</sup> training example; similarly, y<sup>(i)<t></sup> refers to the t<sup>th</sup> element in the output sequence of the i<sup>th</sup> training example
 * T<sub>x</sub><sup>i</sup> is the input sequence length for training example i, which can be different across examples. Similarly, for T<sub>y</sub><sup>i</sup> is the length of the output sequence in the i-th training example



Representing words:
We will now work in this course with NLP which stands for natural language processing. One of the challenges of NLP is how can we represent a word?
We need a vocabulary list that contains all the words in our target sets.
Example:
[a ... And ... Harry ... Potter ... Zulu]
Each word will have a unique index that it can be represented with.
The sorting here is in alphabetical order.
Vocabulary sizes in modern applications are from 30,000 to 50,000. 100,000 is not uncommon. Some of the bigger companies use even a million.
To build vocabulary list, you can read all the texts you have and get m words with the most occurrence, or search online for m most occurrent words.
Create a one-hot encoding sequence for each word in your dataset given the vocabulary you have created.
While converting, what if we meet a word thats not in your dictionary?
We can add a token in the vocabulary with name <UNK> which stands for unknown text and use its index for your one-hot vector.
Full example:

The goal is given this representation for x to learn a mapping using a sequence model to then target output y as a supervised learning problem.
