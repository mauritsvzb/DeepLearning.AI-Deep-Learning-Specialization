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

   <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/d872c8e6-ba4e-4385-ae0c-a0b7e1acea22.png" width="700" />

* Representing words:
  * We will work in this course with natural language processing (NLP) 
  * One of the challenges of NLP is how can we represent individual words in a sentence?
    * We will need a vocabulary dictionary i.e., a list containing all the words that you'll use in your representation e.g.,:
      * [a ... And ... Harry ... Potter ... Zulu]
      * Each word will have a unique index that it can be represented with
      * The sorting here is in alphabetical order
    * Vocabulary sizes in modern applications range from 30k to 50k. 100k is not uncommon. Some of the bigger companies use even a million+
    * To build a vocabulary dictionary, you can read all the texts you have and get m words with the most occurrence, or search online for m most occurrent words
    * Create a one-hot encoding sequence for each word in your dataset given the vocabulary you have created
    * While converting, what if we meet a word thats not in your dictionary?
      * We can add a token in the vocabulary with name <UNK> which stands for unknown text and use its index for your one-hot vector
    * Full example:

      <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/291a33c3-fdde-402c-9a17-3561bbe6b3ca.png" width="700" />

  * The goal is, given this representation for x, to learn a mapping using a sequence model to then target output y as a supervised learning problem

## Recurrent Neural Networks (RNNs)
* Why not to use a standard network for sequence tasks? There are two problems:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/6eaebdda-d6c3-4090-9106-376ef1e3e145.png" width="700" />

* RNNs don't have either of the two mentioned problems
* Let's build a RNN that solves a name entity recognition task:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/5efb23d5-18fa-4406-a22e-737b538f6d80.png" width="700" />

  * In this problem, T<sub>x</sub> = T<sub>y</sub> (in other problems where they aren't equal, the RNN architecture may be different)
  * a<sup><0></sup> is usually initialized with zeros, but some others may initialize it randomly in some cases
  * There are three weight matrices here: W<sub>ax</sub>, W<sub>aa</sub>, and W<sub>ya</sub> with shapes:
    * W<sub>ax</sub>: (NoOfHiddenNeurons, n<sub>x</sub>), whereby n<sub>x</sub> refers to number of input features i.e., the input dimension
      * connects input features to hidden state
    * W<sub>aa</sub>: (NoOfHiddenNeurons, NoOfHiddenNeurons)
      * models recurrent connections within the hidden layer
    * W<sub>ya</sub>: (n<sub>y</sub>, NoOfHiddenNeurons), whereby n<sub>y</sub> refers to number of output classes/labels i.e., the number of categories in a classification task
      * transforms the hidden state to the output
* The weight matrix W<sub>aa</sub> is the memory the RNN as it is passed from the previous layers the the subsequent layers
* A lot of papers and books write the same architecture this way:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/720878fe-add3-49ea-89b8-77caa2f076d0.png" width="300" height="400" />

* But this notation is harder to interpret. It's easier to roll this drawings to the unrolled version shown in the previous image
* In the discussed RNN architecture above, the current output ŷ</sup><t></sup> depends only on the previous inputs and activations i.e. the prediction on ŷ<sup><3></sup> can be made with the information from x<sup><3></sup>, x<sup><2></sup> and x<sup><1></sup>, but not from x<sup><4></sup> and so on...

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/6deffb79-d422-466e-bcab-d38f8151bed9.png" width="700" />

* Let's take this example why this is an issue 
  * 'He Said, "Teddy Roosevelt was a great president"'. In this example, 'Teddy is a person' refers to a person but we know that only from the words that proceed the first 3 words, because you could formulate a sentence where 'teddy' refers to a teddy bear
* Conclusion: a limitation of the discussed architecture is that it can not learn from elements later in the sequence. To address this problem we will later discuss **Bidirectional RNN (BRNN)**
* Now let's discuss the forward propagation equations on the discussed architecture:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/1c73e90b-56f4-4e0d-89ad-0f2aec2a1f15.png" width="700" />

  * The activation function of `a` is usually `tanh` or `ReLU` and for `y` depends on your task choosing some activation functions like `sigmoid` and `softmax`. In a name entity recognition task, we will use sigmoid because we only have two classes
* In order to help us develop complex RNN architectures, the last equations needs to be simplified a bit
* Simplified RNN notation:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/d2976944-dc58-42f5-831c-ad720b492e2f.png" width="700" />

  * W<sub>a</sub> is W<sub>aa</sub> and W<sub>ax</sub> stacked horizontally
  * [a<sup><t-1></sup>, x<sup><t></sup>] is a<sup><t-1></sup> and x<sup><t></sup> stacked vertically
  * W<sub>a</sub> shape: (NoOfHiddenNeurons, NoOfHiddenNeurons + n<sub>x</sub>)
  * [a<sup> <t-1> </sup>, x<sup><t></sup>] shape: (NoOfHiddenNeurons + n<sub>x</sub>, 1)

## Backpropagation Through Time

