# Goal and Learning Objectives
Natural language processing with deep learning is a powerful combination. Using word vector representations and embedding layers, train recurrent neural networks with outstanding performance across a wide variety of applications, including sentiment analysis, named entity recognition and neural machine translation.
* Explain how word embeddings capture relationships between words
* Load pre-trained word vectors
* Measure similarity between word vectors using cosine similarity
* Use word embeddings to solve word analogy problems such as Man is to Woman as King is to ______.
* Reduce bias in word embeddings
* Create an embedding layer in Keras with pre-trained word vectors
* Describe how negative sampling learns word vectors more efficiently than other methods
* Explain the advantages and disadvantages of the GloVe algorithm
* Build a sentiment classifier using word embeddings
* Build and train a more sophisticated classifier using an LSTM

# Natural Language Processing & Word Embeddings
## Introduction to Word Embeddings
### Word Representation
* NLP has been revolutionized by deep learning and especially by RNNs and deep RNNs
* Word embeddings are a way of representing words. It lets your algorithm automatically understand the analogies between words like "king" and "queen"
* So far we have defined our language by a vocabulary. Then represented our words with a one-hot vector that represents the word in the vocabulary
  * An image example would be:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/a9dde336-188c-4f67-a82e-cce7c24791b3.png" width="700" />

    * Annotation O<sub>5391</sub> etc stands for 'one-hot' vector
  * We will use the annotation O<sub>idx</sub> for any word that is represented with one-hot like in the image
  * One of the weaknesses of this representation is that it treats a word as a thing in itself and it doesn't allow an algorithm to generalize across words
      * e.g., "I want a glass of orange ______", a model should predict the next word as juice
      * A similar example e.g., "I want a glass of apple ______", a model won't easily predict juice here if it wasn't trained on it. And if so, the two examples aren't related although "orange" and "apple" are similar.
  * Inner product between any one-hot encoding vector is 0. Also, the distances between them are the same
* So, instead of a one-hot presentation, won't it be nice if we can learn a featurized representation with each of these words: man, woman, king, queen, apple, and orange?
    

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/9f69d916-5ea8-4beb-b9fe-75a277f1df14.png" width="700" />

    * Each word will have a, for example, 300 features with a type of float point number
    * Each word column will be a 300-D vector which will be the representation
    * We will use the notation e5391 to describe man word features vector.
    * Now, if we return to the examples we described again:
      * "I want a glass of orange ______"
      * I want a glass of apple ______
    * Orange and apple now share a lot of similar features which makes it easier for an algorithm to generalize between them
    * We call this representation Word embeddings
* To visualize word embeddings, we use a t-SNE algorithm to reduce the features to 2 dimensions which makes it easy to visualize:
    * Paper: [van der Maaten and Hinton, 2008, Visualizing data using t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html)

      <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/e14188fe-9bff-417e-8c85-1f8755b7a59c.png" width="700" />

    * You will get a sense that more related words are closer to each other
* The word **embeddings** comes from that we need to embed a unique vector inside a n-dimensional space.

### Using Word Embeddings
* Let's see how we can take the feature representation we have extracted from each word and apply it in the Named Entity Recognition problem
* Given this example (from named entity recognition):

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/968a2e46-78ea-486c-ae0d-bbd412080c71.png" width="700" />

  * **Sally Johnson** is a person's name
  * After training on this sentence, the model should find out that the sentence "Robert Lin is an apple farmer" contains Robert Lin as a name, as apple and orange have near representations
  * Now if you have tested your model with this sentence "Mahmoud Badry is a durian cultivator", the model should learn the name even if it hasn't seen the word durian before (during training)
    * That's the power of word representations
  * Note that you would actualy use a bidirectional RNN or BiNNR instead of a unidirectional RNN
* The algorithms that are used to learn word embeddings can examine billions of words of unlabeled text - for example, 100 B words and learn the representation from them
* **Transfer learning** and word embeddings:
  1. Learn word embeddings from large text corpus (1-100 B words)
    * Or download pre-trained embedding online
  2. Transfer embedding to new task with the smaller training set (say, 100k words)
  3. Optional: continue to finetune the word embeddings with new data
    * You bother doing this if your smaller training set (from step 2) is big enough
* Word embeddings tend to make the biggest difference when the task you're trying to carry out has a relatively small training set
* Also, one of the advantages of using word embeddings is that it reduces the size of the input!
  * 10k one-hot compared to 300 features vector
* Word embeddings have an interesting relationship to the face recognition task:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/bb3f862c-979d-4501-9701-2cab09bcff6f.png" width="700" />

* In this problem, we encode each face into a vector and then check how similar are these vectors
* Words encoding and embeddings have a similar meaning here
* In the word embeddings task, we are learning a representation for each word in our vocabulary (unlike in image encoding where we have to map each new image to some n-dimensional vector)
  * We will discuss the algorithm in next sections

### Properties of Word Embeddings
* One of the most fascinating properties of word embeddings is that they can also help with analogy reasoning. While analogy reasoning may not be by itself the most important NLP application, but it might help convey a sense of what these word embeddings can do.
* Analogies example:
  * Given this word embeddings table:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/4738674e-1cba-43de-8dbe-44ce2e85ac97.png" width="700" />

  * Can we conclude this relation:
    * Man ==> Woman
    * King ==> ??
  * Lets subtract e<sub>Man</sub> from e<sub>Woman</sub>, which will approximately equal the vector [-2  0  0  0]
  * Similarly, e<sub>King</sub> - e<sub>Queen</sub> ≈ [-2  0  0  0]

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/e05f4f6c-5fb6-4e77-bbce-372404bcbd7d.png" width="700" />

  * So the difference is about the gender in both

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/63571823-93fa-44c9-945f-f7eb4ac43966.png" width="300" />

      * This vector represents the gender
      * This drawing is a 2D visualization of the 4D vector that has been extracted by a t-SNE algorithm. It's a drawing just for visualization. Don't rely on the t-SNE algorithm for finding parallels.
    * So we can reformulate the problem to find:
      * e<sub>Man</sub> - e<sub>Woman</sub> ≈ e<sub>King</sub> - e<sub>?</sub>
    * It can also be represented mathematically by:

      <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/1d63886f-f5ed-43e4-bad4-39f018d2ad80.png" width="700" />

    * It turns out that e<sub>Queen</sub> is the best solution here that gets the similar vector
      * Paper: [Mikolov et al., 2013, Linguistic Regularities in Continuous Space Word Representations](https://arxiv.org/abs/1301.3781)
* Cosine similarity - the most commonly used similarity function:
  * Equation:

   <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/f200fe7b-fb50-4a8f-86e6-ae90af419091.png" width="700" />

    * `CosineSimilarity(u, v)` = `(u.v)/(||u||.||v||)` = `cos(θ)`
    * The top part represents the inner product of u and v vectors. It will be large if the vectors are very similar.
* You can also use Euclidean distance as a similarity function (but it rather measures a dissimilarity, so you should take it with negative sign)
* We can use this equation to calculate the similarities between word embeddings and on the analogy problem where u = e<sub>w</sub> and v = e<sub>king</sub> - e<sub>man</sub> + e<sub>woman</sub>

## Embedding Matrix
* When you implement an algorithm to learn a word embedding, what you end up learning is a embedding matrix
* Let's take an example:
  * Suppose we are using 10,000 words as our vocabulary (plus token)
  * The algorithm should create a matrix E of the shape (300, 10000) in case we are extracting 300 features

   <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/6f549673-10a8-4202-8779-6cde7a64ea23.png" width="700" />

  * If O<sub>6257</sub> is the one hot encoding of the word orange of shape (10000, 1), then
    * E * O<sub>6257</sub> = e<sub>6257</sub>
    * **The result is an embedded vector of shape (300, 1)**

   <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/1dd4571e-e0e5-4447-8c67-94c411be59be.png" width="700" />

## Learning Word Embeddings: Word2vec & GloVe
### Learning Word Embeddings
* Let's start learning some algorithms that can learn word embeddings
* Historically, word embeddings algorithms were complex but over time they got simpler and simpler
* We will start by learning the complex examples to make more intuition
* **Neural language model**:
  * Paper: [Bengio et al., 2003, A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
  * Let's start with an example:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/cde7b1a4-0cda-4b8c-9f06-eb0eb90edaec.png" width="700" />

  * We want to build a language model so that we can predict the next word
  * So we use this neural network to learn the language model:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/d6f2b9cc-8f10-433b-bede-fd14d1231271.png" width="700" />

    * We get e<sub>j</sub> by multipling the embedding matrix E by the one-hot O<sub>j</sub>
    * Input dimension is (300*6, 1) if the window size is 6 (six previous words)
    * Here we are optimizing matrix E and layers parameters. We need to maximize the likelihood to predict the next word given the context (previous words)
  * This model was build in 2003 and tends to work pretty decent for learning word embeddings
* In the last example, we took a window of 6 words that come before the word that we want to predict i.e., the 'target' word. There are other choices when we are trying to learn word embeddings.
* Suppose we have an example: "I want a glass of orange juice to go along with my cereal."
* To learn the word 'juice', choices of context are:
  
  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/2d383476-4392-417f-b293-723f778097fb.png" width="700" />

* These idea is much simpler and works remarkably well, and we will talk about this in the next section
* Researchers discovered that if you really want to build a **language model**, it's natural to use the last few words as a context. But if your main goal is really to **learn a word embedding**, then you can use all of these other contexts and they will result in very meaningful work embeddings as well.
* To summarize, the language modeling problem poses a machine learning problem where you input the context (like the last four words) and predict some target words. And posing that problem allows you to learn good word embeddings.

### Word2Vec
* Paper: [Mikolov et al., 2013, Efficient Estimation of Word Representations in Vector Space]
* Before presenting Word2Vec, lets talk about skip-grams:
  * For example, we have the sentence: "I want a glass of orange juice to go along with my cereal."
  * We will choose context and target
  * The target is chosen randomly based on a window with a specific size

| Context| 	Target	| How far |
| --- | --- | --- |
| orange | 	juice	| +1
| orange	glass	| -2 |
| orange	| my |	+6 |

  * Obviously, this is not a very easy learning problem, because within plus minus 10 words of the word orange, it could be a lot of different words. But a goal of setting up this supervised learning problem isn't to do well on the supervised learning problem per se, it is that we want to use this learning problem to learn good word embeddings

* word2vec model:
  * Vocabulary size = 10,000 words
  * Let's say that the context word is 'c' ("orange") and the target word is 't' ("juice")
  * We want to learn to map from c to t
  * We get e<sub>c</sub> by np.dot(E, O<sub>c</sub>)
  * We then use a softmax layer to get P(t|c) which is ŷ
  * Also we will use the cross-entropy loss function
    * L(ŷ,y) = -Sum<sub>i=1:n</sub><sub>y</sub>i.log(ŷ<sub>i</sub>) with n = 10,000
  
    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/01c85a10-9967-40ec-aaf9-cdcb824183f7.png" width="700" />

    * This model (encircled in green ) is called skip-grams model (a form of word2vec model); e has its own parameters, and so does the softmax
* The last model has a problem with the softmax layer:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/2d8f40c8-1afc-4576-bc5d-66b1181642d2.png" width="300" />

* Here we are summing 10,000 numbers which corresponds to the number of words in our vocabulary
* If this number is large e,g., 1 million, the computation will become very slow
* One of the solutions for the last problem is to use **"Hierarchical softmax classifier"**, which works as a binary tree classifier

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/2db76473-d96e-44c8-825e-49dc5ee640f3.png" width="300" />

* In practice, the hierarchical softmax classifier doesn't use a balanced tree like the drawn one. Common words are at the top and less common are further down the tree.

* How to sample the context c?
  * One way is to choose the context by random from your corpus
  * If you have done it that way, there will be frequent words like "the, of, a, and, to,..." that can dominate other words like "orange, apple, durian,..."
  * In practice, we don't take the context uniformly random, but instead there are some heuristics to balance the common words and the non-common words
* Word2Vec paper includes 2 ideas of learning word embeddings. One is skip-gram model and another is CBoW (continuous bag-of-words)
* But the key problem with this algorithm with the skip-gram model as presented so far is that the softmax step is very expensive to calculate because needing to sum over your entire vocabulary size into the denominator of the soft packs

### Negative Sampling
* Negative sampling allows you to do something similar to the skip-gram model, but with a much more efficient learning algorithm. We will create a different learning problem.
  * Paper: Mikolow et al., 2013, Distributed Representations of Words and Phrases and their Compositionality
* Given this example:
  * "I want a glass of orange juice to go along with my cereal."
  * The sampling will look like this:
    | Context	| Word	| target
    | --- | --- | --- |
    | orange	| juice	| 1 |
    | orange	| king	| 0 |
    | orange	| book	| 0 |
    | orange	| the	| 0 |
    | orange	| of	| 0 |
  * We get a positive example by using the same skip-grams technique, with a fixed window that goes around
  * To generate a negative example, we pick a word randomly from the vocabulary
  * So the steps to generate the samples are:
    1. Pick a positive context
    2. Pick k negative contexts from the dictionary
  * k is recommended to be from 5-20 in small datasets; for larger ones use 2-5 contexts
  * We will have a ratio of k negative examples to 1 positive ones in the data we are collecting

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/f7b95560-33b3-40b2-97da-7060e6263b23.png" width="700" />

* Now let's define the model that will learn this supervised learning problem:
  * Lets say that the context word is 'c' and the target word is 't' and y is the target prediction
  * We will apply the simple logistic regression model

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/a33030c5-cf4f-4b21-8af1-a587ddad9dfb.png" width="700" />

The logistic regression model can be drawn like this:

    *  So  we are like having 10

* How to select negative samples:
  * We can sample according to empirical frequencies in words corpus which means according to how often different words appears
    * The problem with this is that we will end up more frequently with words like words the, of, and, etc.
  * The best is to sample with this equation (according to authors):

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/37b22352-e743-435b-990c-fd6e66ef7715.png" width="300" />

With f(a) is the observed frequency of the word a in the training set corpus




    * So we are like having 10,000 binary classification problems, and we only train k negative + 1 positive classifier of them in each iteration, whereby each iteration consists of a different subset of k classifiers

    * So we are like having 10,000 binary classification problems, and we only train k negative + 1 positive classifier of them in each iteration, whereby each iteration consists of a different subset of k classifiers

    <img src=".png" width="700" />
