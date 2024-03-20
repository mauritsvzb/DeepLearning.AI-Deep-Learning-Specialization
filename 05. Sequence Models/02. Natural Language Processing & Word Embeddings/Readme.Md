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
## Word Representation
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

## Using Word Embeddings
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

## Properties of Word Embeddings
* One of the most fascinating properties of word embeddings is that they can also help with analogy reasoning. While analogy reasoning may not be by itself the most important NLP application, but it might help convey a sense of what these word embeddings can do.
* Analogies example:
  * Given this word embeddings table:

   <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/4738674e-1cba-43de-8dbe-44ce2e85ac97.png" width="700" />

  * Can we conclude this relation:
    * Man ==> Woman
    * King ==> ??
  * Lets subtract e<sub>Man</sub> from e<sub>Woman</sub>, which will approximately equal the vector [-2  0  0  0]
  * Similarly, e<sub>King</sub> - e<sub>Queen</sub> ≈ [-2  0  0  0]
  * So the difference is about the gender in both

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/63571823-93fa-44c9-945f-f7eb4ac43966.png" width="300" />

      * This vector represents the gender
      * This drawing is a 2D visualization of the 4D vector that has been extracted by a t-SNE algorithm. It's a drawing just for visualization. Don't rely on the t-SNE algorithm for finding parallels.
    * So we can reformulate the problem to find:
      * e<sub>Man</sub> - e<sub>Woman</sub> ≈ e<sub>King</sub> - e<sub>?</sub>
    * It can also be represented mathematically by:

      <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/1d63886f-f5ed-43e4-bad4-39f018d2ad80.png" width="700" />

    * It turns out that e<sub>Queen</sub> is the best solution here that gets the similar vector
      * Paper: Mikolov et al., 2013, Linguistic Regularities in Continuous Space Word Representations
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
    * E * O<sub>6257</sub> = e<sub>6257</sub>, which shape is (300, 1)

   <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/1dd4571e-e0e5-4447-8c67-94c411be59be.png" width="700" />

## Learning Word Embeddings





   <img src=".png" width="700" />
