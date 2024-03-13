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

