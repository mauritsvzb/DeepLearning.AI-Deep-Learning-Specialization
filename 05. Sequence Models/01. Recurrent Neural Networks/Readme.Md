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
    * For predicting the chance of next word, we feed the sentence to the RNN and then get the final ŷ<sup>< t ></sup> hot vector and sort it by maximum probability
    * For taking the probability of a sentence, we compute this:
      * p(y<sup><1></sup>, y<sup><2></sup>, y<sup><3></sup>) = p(y<sup><1></sup>) * p(y<sup><2></sup> | y<sup><1></sup>) * p(y<sup><3></sup> | y<sup><1></sup>, y<sup><2></sup>)
    * This is simply feeding the sentence into the RNN and multiplying the probabilities (outputs)

## Sampling Novel Sequences
* After a sequence model is trained on a language model, to check what the model has learned you can apply it to sample a novel sequence
* Lets see the steps of how we can sample a novel sequence from a trained sequence language model:
  1. Given this model:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/7e97cf6c-4fcb-4fd9-b7fd-1a81e99a4aeb.png" width="700" />

  2. We first pass a<sup><0></sup> = 0s vector, and x<sup><1></sup> = 0s vector
  3. Then we choose a prediction randomly from the distribution obtained by ŷ<sup><1></sup> e.g., it could be "The"
    * In Numpy this can be implemented using: `numpy.random.choice(...)`
    * This is the line where you get a random beginning of the sentence each time you sample run a novel sequence
  4. We pass the last predicted word ŷ<sup><1></sup> to the next block a<sup><2></sup> to calculate ŷ<sup><2></sup>
  5. We keep doing repeating 3 & 4 for a fixed length or until we get the `<EOS>` token

     <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/6d2669bb-d637-40b2-80a1-41a6293f5825.png" width="700" />

  6. You can reject any `<UNK>` token if you mind finding it in your output
* So far we have to build a word-level language model, but it's also possible to implement a character-level language model
* In the character-level language model, the vocabulary will contain [a,..,z,A,..Z,<special chars>,0..9], punctuation, special characters and possibly tokens
* Character-level language models have some pros and cons compared to the word-level language model
  * Pros:
    * There will be no `<UNK>` token - it can create any word
  * Cons:
      * The main disadvantage is that you end up with much longer sequences
      * Character-level language models are not as good as word-level language models at capturing long range dependencies between how the the earlier parts of the sentence also affect the later part of the sentence
      * Also more computationally expensive and harder to train
* The trend Andrew has seen in NLP is that for the most part, a word-level language model is still used, but as computers get faster there are more and more applications where people are, at least in some special cases, starting to look at more character-level models. Also, they are used in specialized applications where you might need to deal with unknown words or other vocabulary words a lot. Or they are also used in more specialized applications where you have a more specialized vocabulary.

## Vanishing gradients with RNNs
* One of the problems with naive RNNs that they run into vanishing gradient problem if they have a long sequence size
* A RNN that processes sequence data with the size of 10,000 time steps has 10,000 deep layers which is very hard to optimize
* E.g., suppose we are working with a language modeling problem and there are two sequences that model tries to learn:
  * "The cat, which already ate ..., was full"
  * "The cats, which already ate ..., were full"
    * Dots represent many random words in between
  * What the deal is here is that "was" is used because it refers to "cat" (singular), whereas "were" refers to "cats"
    * The naive RNN is not very good at capturing very long-term dependencies like this

      <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/93ed131f-0434-4af6-9bb2-26db6bc766d6.png" width="700" />

    * To compute the word "was", we need to compute the gradient for every word that preceeds this word. Multiplying fractions tends to cause gradients to vanish, while multiplication of large number tends to cause gradient explosion
    * Therefore, some of your weights may not be updated properly
* In the problem we described it means that it's hard for the network to memorize it should predict "was" because this pertains to a single "cat". So in this case, the network won't identify the singular/plural words so that it gives it the right grammar form of verb "was/were"
* The conclusion is that basic RNNs aren't good in long-term dependencies
> In theory, RNNs are absolutely capable of handling such “long-term dependencies.” A human could carefully pick parameters for them to solve toy problems of this form. Sadly, in practice, RNNs don’t seem to be able to learn them. http://colah.github.io/posts/2015-08-Understanding-LSTMs/
* **Vanishing gradient** problems tend to be the bigger problem with RNNs than the **exploding gradient** problems (We will discuss how to solve it in next sections)
* Exploding gradients can be easily seen when your weight values become `NaN`. So one of the ways to solve an exploding gradient is to apply gradient clipping, meaning if your gradient is more than some threshold - rescale some of your gradient vector so that values are not too big (so they are cliped according to some maximum value)

   <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/c4b3484f-f105-4c86-93fb-38b1121ff675.png" width="700" />

## Gated Recurrent Unit (GRU)
* GRU is an RNN type that can help solve the vanishing gradient problem and can remember the long-term dependencies
* Papers:
  * [Cho et al., 2014, On the properties of neural machine translation: Encoder-decoder approaches](https://arxiv.org/abs/1409.1259)
  * [Chung et al., 20014, Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555)
* The basic RNN unit can be visualized to be like this:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/d9118f56-e627-47a8-8eee-0824f6cda5ab.png" width="700" />

* We will represent a GRU with a similar drawing
* Each layer in a GRU has a new variable C which denotes a memory cell. It can be used to determine whether to memorize something or not
* In GRUs, c<sup>< t ></sup> = a<sup>< t ></sup>
* Equations of the GRUs:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/99e0f249-744a-4d06-bbfa-48d617d493ab.png" width="700" />

  * where c<sup><t - 1></sup> is the old value and c̃<sup>< t ></sup> is the candidate value
* The update gate (called Gamma) is always between 0 and 1
* To understand GRUs, imagine that the update gate is either 0 or 1 most of the time
* So we update the memory cell based on the update cell and the previous cell
  * Drawings like in http://colah.github.io/posts/2015-08-Understanding-LSTMs/ can help you understand GRUs and LSTMs
* Let's take the cat sentence example again and apply it to understand this equations:
  * Sentence: "The cat, which already ate ........................, was full"
  * We now suppose that U is 0 or 1 and is the part that tells us if a singular word needs to be memorized
  * Splitting the words and get values of C and U at each place:

    |**Word**	| **Update gate(U)** |	**Cell memory (C)** |
    | --- | --- | --- |
    |The |	0	| val |
    |cat	| 1	| new_val |
    | which |	0	| new_val |
    | already	| 0	| new_val |
    |...|	0	| new_val|
    | was |	1 (I don't need it anymore)	| newer_val |
    |full	| ...	| ... |

* Because the update gate U is usually a small number like 0.00001, GRUs don't suffer from the vanishing gradient problem
  * In the equation this makes c<sup>< t ></sup> = c<sup><t - 1><sup> in a lot of cases
* Shapes:
  * a<sup>< t >sup</sup> shape is (NoOfHiddenNeurons, 1)
  * c<sup>< t ></sup> is the same as a<sup>< t ></sup>
  * c̃<sup>< t ></sup> is the same as a<sup>< t ></sup>
  * u<sup>< t ></sup> is also the same dimensions of a<sup>< t ></sup>
* The multiplications in the equations are element-wise multiplications
* What has been described so far is the simplified GRU unit: Let's now describe the full GRU one:
  * The **full GRU** contains a new gate that is used to calculate the candidate c̃. The gate tells you how relevant is c<sup><t - 1></sup> to c<sup>< t ></sup>
  * Equations:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/5b82889a-a3fe-49f7-9342-e649255dd990.png" width="700" />

    * where gamma(r) represents how relevant is the c<sup><t - 1></sup> to computing c<sup>< t ></sup>
* So why do we use these architectures, why don't we change them, how we know they will work, why not add another gate, why not use the simpler GRU instead of the full GRU; because researchers have experimented over years all the various types of these architectures with many  different versions and also addressing the vanishing gradient problem. They have found that full GRUs are one of the best RNN architectures to be used for many different problems. You can make your design but put in mind that GRUs and LSTMs are standards.

## Long Short Term Memory (LSTM)
* LSTM - the other type of RNN that can enable you to account for long-term dependencies. It's more powerful and general than GRU, but also a bit more complicated to use
  * Paper: [Hochreiter & Schmidhuber, 1997, Long Short-term memory](https://ieeexplore.ieee.org/abstract/document/6795963)
    * It is recommended to read other resources for the details of LSTM rather than using the original paper
* In LSTM , C<sup>< t ></sup> != a<sup>< t ></sup>
* Here are the equations of an LSTM unit:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/907f674f-a3e2-43e2-ac4b-0785decd1647.png" width="700" />

* In GRU we have an update gate u, a relevance gate r, and a candidate cell variables c̃<sup>< t ></sup> while in LSTM we have an update gate u (sometimes it's called input gate I), a forget gate f, an output gate O, and a candidate cell variables c̃<sup>< t ></sup>
* Drawings (inspired by http://colah.github.io/posts/2015-08-Understanding-LSTMs/):

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/816f0b7b-2804-4a83-8c36-76d690ec5c63.png" width="700" />

* And we simply connect the LSTM blocks together like this:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/47b9f90b-7bd5-4513-b808-351cc7c7671f.png" width="700" />

  * Also, there is a direct line (red line) at the top to have some initial values i.e. c<sup><0></sup> pass all the way to later blocks i.e. c<sup><3></sup> so long as the forget and update gates are set appropriately
  * Some variants on LSTM includes:
    * LSTM with **peephole connections**
      * The normal LSTM with C<sup><t - 1></sup> included with every gate
* There isn't a universal superior between LSTM and it's variants. One of the advantages of GRU is that it's simpler and can be used to build much bigger network but the LSTM is more powerful and general.

## Bidirectional RNN
* There are still some ideas to let you build much more powerful sequence models, e.g., a bidirectional RNN and a Deep RNN
* As we saw before, here is an example of the Name entity recognition task:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/7f9c6f4c-2e51-46e7-ae96-0e265c306fb1.png" width="700" />

* Whether Teddy refers to a name in this case cannot be learned from the first two words 'he said', but can be learned from the word that comes after Teddy i.e., 'bears'
* BiRNNs fix this issue
* Here is BRNNs architecture:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/a6f73d64-b4b5-44c8-b1d6-e4a381850152.png" width="700" />

  * Note, that BiRNN is an acyclic graph
* Part of the forward propagation goes from left to right, and part - from right to left; a BiRNN learns from going from either side
* To make predictions we use ŷ<sup>< t ></sup> by using the two activations that come from left and right
* The blocks here can be any RNN block including the basic RNNs, LSTMs, or GRUs
* For a lot of NLP or text processing problems, a BiRNN with LSTM appears to be commonly used
* The **disadvantage** of BiRNNs is that they need to the entire sequence before you can process it e.g., in live speech recognition, if you use BiRNNs you will need to wait for the person who speaks to stop to take the entire sequence and then make your predictions

## Deep RNNs
* In a lot of cases the standard one layer RNNs will solve your problem. But in some problems it's useful to stack some RNN layers to make a deeper network
* For example, a deep RNN with 3 layers would look like this:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/92fadb19-1b79-46af-a766-cb0ae4d389c6.png" width="700" />

* With the activation i.e. a<sup>[2]<3></sup> = g(W<sub>a</sub><sup>[2]</sup>[a<sup>[2]<2></sup>,a<sup>[1]<3></sup>] + b<sub>a</sub><sup>[2]</sup>)
* In feed-forward deep nets, there could be 100 or even 200 layers. In deep RNNs stacking 3 layers is already considered deep and expensive to train
* In some cases you might see some feed-forward network layers connected after recurrent cell

## What you should remember:
* The recurrent neural network, or RNN, is essentially the repeated use of a single cell
* A basic RNN reads inputs one at a time, and **remembers** (through memory) information through the hidden layer activations (hidden states) that are passed from one time step to the next
* The time step dimension determines how many times to re-use the RNN cell
* Each cell takes two inputs at each time step:
  1. The hidden state from the previous cell
  2. The current time step's input data
* Each cell has two outputs at each time step:
  1. A hidden state
  2. A prediction
* A Long Short Term Memory (LSTM) is similar to an RNN in that they both use hidden states to pass along information, but an LSTM also uses a cell state, which is like a long-term memory, to help deal with the issue of vanishing gradients
* An LSTM cell consists of a cell state, or long-term memory, a hidden state, or short-term memory, along with 3 gates that constantly update the relevancy of its inputs:
  * A **forget** gate, which decides which input units should be remembered and passed along. It's a tensor with values between 0 and 1.
      * If a unit has a value close to 0, the LSTM will "forget" the stored state in the previous cell state
      * If it has a value close to 1, the LSTM will mostly remember the corresponding value
  * An **update** gate, again a tensor containing values between 0 and 1. It decides on what information to throw away, and what new information to add.
      * When a unit in the update gate is close to 1, the value of its candidate is passed on to the hidden state
      * When a unit in the update gate is close to 0, it's prevented from being passed onto the hidden state
  * And an **output** gate, which decides what gets sent as the output of the time step
* Very large, or "exploding" gradients updates can be so large that they "overshoot" the optimal values during back prop -- making training difficult
* Clip gradients before updating the parameters to avoid exploding gradients
* Sampling is a technique you can use to pick the index of the next character according to a probability distribution
* To begin character-level sampling:
  * Input a "dummy" vector of zeros as a default input
  * Run one step of forward propagation to get 𝑎⟨1⟩ (your first character) and 𝑦̂ ⟨1⟩ (probability distribution for the following character)
  * When sampling, avoid generating the same result each time given the starting letter (and make your names more interesting!) by using `np.random.choice`
* A sequence model can be used to generate musical values, which are then post-processed into midi music
* You can use a fairly similar model for tasks ranging from generating dinosaur names to generating original music, with the only major difference being the input fed to the model
* In Keras, sequence generation involves defining layers with shared weights, which are then repeated for the different time steps  1,…,𝑇𝑥
