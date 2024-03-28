# Goal and Learning Objectives:
Augment your sequence models using an attention mechanism, an algorithm that helps your model decide where to focus its attention given a sequence of inputs. Then, explore speech recognition and how to deal with audio data.
* Describe a basic sequence-to-sequence model
* Compare and contrast several different algorithms for language translation
* Optimize beam search and analyze it for errors
* Use beam search to identify likely translations
* Apply BLEU score to machine-translated text
* Implement an attention model
* Train a trigger word detection model and make predictions
* Synthesize and process audio recordings to create train/dev datasets
* Structure a speech recognition project

# Sequence Models & Attention Mechanism
## Various Sequence to Sequence Architectures
### Basic Models
* In this section we will learn about sequence to sequence - Many to Many - models which are useful in various applications including machine translation and speech recognition
* Let's start with the basic model:
  * Given this machine translation problem in which X is a French sequence and Y is an English sequence:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/adb5322d-2f0c-4ba0-a3eb-351e356343e1.png" width="350" />

  * Our architecture will include an **encoder** and **decoder**
  * The encoder is an RNN - LSTM or GRU are included - and takes the input sequence and then outputs a vector that should represent the whole input
  * After that the decoder network, also an RNN, takes the sequence built by the encoder and outputs the new sequence:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/7a141172-d6aa-4108-85a2-282c5b30caa4.png" width="350" />

  * These ideas are from the following papers:
    * [Sutskever et al., 2014, Sequence to sequence learning with neural networks](https://arxiv.org/abs/1409.3215)
    * [Cho et al., 2014, Learning phrase representations using RNN encoder-decoder for statistical machine translation](https://arxiv.org/abs/1406.1078)
* An architecture similar to the mentioned above works for image captioning problem:
  * In this problem X is an image, while Y is a sentence sequence (caption)
  * The model architecture image:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/5e769ad2-6066-4e06-8fa8-298039aee74d.png" width="700" />

  * The architecture uses a pretrained CNN (like AlexNet) as an encoder for the image, and the decoder is an RNN
  * Ideas are from the following papers (they share similar ideas):
    [Mao et et. al., 2014, Deep captioning with multimodal recurrent neural networks](https://arxiv.org/abs/1412.6632)
    [Vinyals et. al., 2014, Show and tell: Neural image caption generator](https://arxiv.org/abs/1411.4555)
    [Karpathy and Li, 2015, Deep visual-semantic alignments for generating image descriptions](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)

### Picking the Most Likely Sentence
* There are some similarities between the language model we have learned previously, and the machine translation model we have just discussed, but there are some differences as well
* The language model we have learned is very similar to the decoder part of the machine translation model, except for a<sup><0></sup>

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/a6e315f4-4ec3-4283-9d6a-e1a52c6ec316.png" width="700" />

* Rather than starting with an initial representation with all zeros in language model, the machine translation has an encoded network to figure out some representation for input sentence => **Conditional language model**
* Problems formulations also are different:
  * In language model: P(y<sup><1></sup>, ..., y</sup><Ty></sup>)
  * In machine translation: P(y<sup><1></sup>, ..., y<sup><Ty></sup> | x<sup><1></sup>, ..., x<sup><Tx></sup>)
* What we don't want is to sample outputs at random. e.g., maybe one time you may get a good translation, but other times you may get a different (bad) translation
* Example:
  * X = "Jane visite l’Afrique en septembre."
  * Y may be:
    * Jane is visiting Africa in September
    * Jane is going to be visiting Africa in September
    * In September, Jane will visit Africa
    * Her African friend welcomed Jane in September
* So, when using this model for machine translation, you're not trying to sample at random from this distribution. Instead, what you would like is to find the English sentence, y, that maximizes that conditional probability.
* So in developing a machine translation system, one of the things you need to do is come up with an algorithm that can actually find the value of y that maximizes the probability:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/fe8f7e9a-49b1-4a3b-8f2b-468bd9a1827a.png" width="350" />

* The most common algorithm is the beam search, which we will explain in the next section
* Why not use greedy search? Why not get the best choices each time?
  * It turns out that this approach doesn't really work!
  * Lets explain it with an example:
    * The best output for the example we talked about is *"Jane is visiting Africa in September."*
    * Suppose that when you are choosing with greedy approach, the first two words were *"Jane is"*, the word that may come after that will be *"going"* as *"going"* is the most common word (with high probability) that comes after *"is"* so the result may look like this: *"Jane is going to be visiting Africa in September."*. And that isn't the best/optimal solution.
* If you want to find the sequence of words, y1, y2, all the way up to the final word that together maximize the probability, it's not always optimal to just pick one word at a time. And, of course, the total number of combinations of words in the English sentence is exponentially larger.
* So what is better than greedy approach, is to get an approximate solution for the whole sentence, that will try to maximize the output (the last equation above).

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/0a277f76-c59b-40d9-aa6c-7dbbf800aad1.png" width="700" />

### Beam Search
* Beam search is the most widely used algorithm to get the best output sequence. It's a heuristic search algorithm.
* To illustrate the algorithm we will stick with the example from the previous section. We need Y = *"Jane is visiting Africa in September."*
* The algorithm has a parameter B which is the beam width. Lets take B = 3 which means the algorithm will consider/get 3 possibility outputs at a time.
  * For the first step you will get [*"in"*, *"jane"*, *"september"*] words that are the most likely 3 best candidates for the first words (top 3) among 10,000 possibilities

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/f21218cf-5c71-4126-b03b-2fac2eb45048.png" width="700" />

  * Then for each word in the first output, get B next (second) words and select top best B combinations where the best are those what give the highest value of multiplying both probabilities: P(y<sup><1></sup>,y<sup><2></sup>|x) = P(y<sup><1></sup>|x) * P(y<sup><2></sup>|x, y<sup><1></sup>)

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/b19f915f-5384-4abc-ad1a-594408f0c782.png" width="700" />

  * Which means, we have to work with 30,000 choices of word in total. And we have to make 3 copies of the network to evaluate the possible output (10,000 possibilities for each)
  * Then we will have [*"in september"*, *"jane is"*, *"jane visits"*]. Notice, that we automatically discard "september" as a first word if Beam Search chose this triplet.
  * Repeat the same process and get the next best B words for [*"in september"*, *"jane is"*, *"jane visits"*] and so on until the end of sentence 
* In this algorithm, keep only B instances of your network. Which means we have to work with B * N choices for each step
* If `B = 1`, then this will become the greedy search

### Refinements to Beam Search
* In the previous section, we have discussed the basic beam search. In this section, we will try to do some refinements to it.
* The first thing is ** Length optimization** 
  * In beam search we are trying to optimize:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/f4e705a7-2dab-426b-9d98-3ab597931e58.png" width="350" />

    * And to do that we multiply: P(y<1> | x) * P(y<2> | x, y<1>) * ... * P(y<t> | x, y<y(t-1)>)
      * Each probability is a fraction, most of the time a small fraction
    * However, multiplying small fractions will cause a numerical overflow. Meaning that the value is too small for the floating part representation in your computer to store accurately
  * So in practice we use **summing logs** of probabilities instead of multiplying directly

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/43becca6-568f-4f86-8bed-9160edd3d817.png" width="350" />

    *  But there's another problem. The two optimization functions we have mentioned prefer small sequences rather than long ones. Because multiplying more fractions gives a smaller value, so fewer fractions - bigger values = better result.
  * So there's another step - dividing by the number of elements in the sequence

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/89299807-6ee9-4107-a96c-3b29a6d72068.png" width="350" />

    * with alpha is a hyperparameter to tune
      * If `alpha = 0` - no sequence length normalization
      * If `alpha = 1` - full sequence length normalization
      * In practice, `alpha = 0.7` is a good thing (somewhere in between two extremes)
* The second thing is **how can we choose best beam width `B`**?
  * The larger the value of B, the more possibilities are considered, and the better the result is achieved
    * However, increasing B also comes with increasing computational cost
  * In practice, you might see in the production setting `B = 10`
    * `B = 100`, `B = 1000` are uncommon (sometimes used in research settings)
  * Unlike exact search algorithms like BFS (Breadth First Search) or DFS (Depth First Search), Beam Search runs faster but is not guaranteed to find the exact solution

### Error Analysis in Beam Search












## Speech Recognition (Audio Data)
### Speech Recognition


### Trigger Word Detection
