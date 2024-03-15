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
  * x<sup>(i)< t ></sup> refers to the t<sup>th</sup> element in the input sequence of the i<sup>th</sup> training example; similarly, y<sup>(i)< t ></sup> refers to the t<sup>th</sup> element in the output sequence of the i<sup>th</sup> training example
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
* In the discussed RNN architecture above, the current output ŷ<sup>< t ></sup> depends only on the previous inputs and activations i.e. the prediction on ŷ<sup><3></sup> can be made with the information from x<sup><3></sup>, x<sup><2></sup> and x<sup><1></sup>, but not from x<sup><4></sup> and so on...

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
  * [a<sup><t - 1></sup>, x<sup>< t ></sup>] is a<sup><t - 1></sup> and x<sup>< t ></sup> stacked vertically
  * W<sub>a</sub> shape: (NoOfHiddenNeurons, NoOfHiddenNeurons + n<sub>x</sub>)
  * [a<sup> <t - 1> </sup>, x<sup>< t ></sup>] shape: (NoOfHiddenNeurons + n<sub>x</sub>, 1)

## Backpropagation Through Time
* In modern deep learning frameworks, you only have to implement the forward pass, and the framework takes care of the backward pass, so most deep learning engineers do not need to bother with the details of the backward pass. If however you are an expert in calculus and want to see the details of backprop in RNNs, you can work through this.
* Let's see how backpropagation works with the RNN architecture
* Here is the graph:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/87427133-d955-4a69-b04c-1bc8265d9637.png" width="700" />

  * where W<sub>a</sub>, b<sub>a</sub>, W<sub>y</sub>, and b<sub>y</sub> are shared across each element in a sequence
* We will use the cross-entropy loss function:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/bf3a349f-0169-4b9e-b891-a677b8fc4f9b.png" width="700" />

  * Where the first equation is the loss for one example and the loss for the whole sequence is given by the summation over all the calculated single example losses
  * Graph with losses:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/0426fbd7-1384-428a-9ccb-62634b639358.png" width="700" />

* The backpropagation here is called **backpropagation through time**, because we pass activations a from one sequence element to another like backwards in time (red line)

## Different types of RNNs
* So far we have seen only one RNN architecture in which T<sub>x</sub> equals T<sub></sub>y</sub>. In some other problems, they may not be equal so we need different architectures
* The ideas in this section was inspired by Andrej Karpathy's [blog](https://karpathy.github.io/2015/05/21/rnn-effectiveness/). Mainly this image has all types:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/bb2e0a6a-a6d4-446f-9b1a-9530ae687b4f.png" width="700" />

* The architecture we have described until now is called **Many to Many**, but here are some others

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/ed354e98-6c30-4bc7-8bdf-b69c392e6e0a.png" width="700" />

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/de9a785d-d98f-4ba4-938f-581e3435867a.png" width="700" />

## Language Model and Sequence Generation
* RNNs do very well in language model problems. In this section, we will build a language model using RNNs.
* What is a language model?
  * Let's say we are solving a speech recognition problem and someone says a sentence that can be interpreted into to two sentences:
    * The apple and pair salad
    * The apple and pear salad
  * 'Pair' and 'pear' sounds exactly the same, so how would a speech recognition application choose from the two?
  * This is where the language model comes in: it gives a probability for the two sentences and the application choses the best sentencce based on this probability
  * The job of a language model is to give a probability of any given sequence of words
* How to build language models with RNNs?
  * The first thing to do is to get a training set: a large 'corpus' of target language text
  * Then tokenize this training set by getting the vocabulary and then one-hot each word
  * Put an end-of-sentence token <EOS> with the vocabulary and include it with each converted sentence. Also, use the token <UNK> for the unknown words.
* Given the sentence "Cats average 15 hours of sleep a day. <EOS>"
  * In training time we will use this:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/468a5180-198a-4230-aeb5-562de620cc1a.png" width="700" />

  * The loss function is defined by cross-entropy loss:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/c064b4e0-ad1e-49eb-8c58-97ac4c3896bd.png" width="300" />

      * 'i' is for all elements in the corpus, 't -' for all time steps
  * To use this model:
    * For predicting the chance of next word, we feed the sentence to the RNN and then get the final y^<t> hot vector and sort it by maximum probability
    * For taking the probability of a sentence, we compute this:
      * `p(y<1>, y<2>, y<3>) = p(y<1>) * p(y<2> | y<1>) * p(y<3> | y<1>, y<2>)`
    * This is simply feeding the sentence into the RNN and multiplying the probabilities (outputs)

## Sampling Novel Sequences



