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
* We have talked before about Error analysis in "Structuring Machine Learning Projects" course. We will apply these concepts to improve our beam search algorithm.
* We will use error analysis to figure out if the B hyperparameter of the beam search is the problem (it doesn't get an optimal solution) or our RNN part.
* Let's take an example:
  * Initial info:
    * x = *"Jane visite l’Afrique en septembre."*
    * y* = *"Jane visits Africa in September."* - right answer (human)
    * ŷ = *"Jane visited Africa last September."* - answer produced by model
  * Observation: Our model has produced not a good result
  * We now want to know who to blame - the RNN or the beam search?
  * To do that, we calculate P(y* | X) and P(ŷ | X). There are two cases:
    * Case 1: P(y* | X) > P(ŷ | X)
      * Beam search chose ŷ. But y* attains higher P(y | X)
      * Conclusion: Beam search is at fault
    * Case 2: P(y* | X) ≤ P(ŷ | X)
      * y* is a better translation than ŷ. But RNN predicted otherwise
      * Conclusion: RNN model is at fault
* The error analysis process is as following:
  * You choose N error examples and make the following table:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/4efc1981-7a38-431c-92b5-b6ae93a21531.png" width="700" />

* B is for beam search, R is for the RNN
* Get counts and then decide what to work on next

### Bleu Score (Optional)
* One of the challenges of machine translation is that, given a sentence in a language, there are one or more possible good translations in another language. So how do we evaluate our results?
* The way we do this is by using BLEU score (*bilingual evaluation understudy*)
  * Paper: [Papineni et al., 2002, BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040.pdf)
  * The intuition is: as long as the machine-generated translation is pretty close to any of the references provided by humans, then it will receive a high BLEU score
* Let's take an example:
  * X = *"Le chat est sur le tapis."*
  * Y1 = *"The cat is on the mat."*   (human reference 1)
  * Y2 = "There is a cat on the mat." (human reference 2)
  * Suppose that the machine outputs: 
    * "the the the the the the the."
  * One way to evaluate the machine translation output is to look at each word in the output and check if it is in the references. This is called *precision*:
    * In this case: precision = 7/7 because every word in the output 'the' appears in Y1 or Y2
    * This is not a useful measure!
  * We can use a *modified precision* in which we are looking for the reference with the maximum number of a particular word and set the maximum appearing of this word to this number. So:
    * modified precision = 2/7 because the max of 'the' is 2 in Y1
    * We clipped 'the' 7 times by the max which is 2.
  * Here we are looking at one word at a time - *unigrams*, we may look at *n-grams* too
* BLEU score on bigrams
  * The n-grams are typically collected from a text or speech corpus. When the items are words, n-grams may also be called shingles. An n-gram of size 1 is referred to as a "unigram"; size 2 is a "bigram" (or, less commonly, a "digram"); size 3 is a "trigram", etc
  * X = "Le chat est sur le tapis."
  * Y1 = "The cat is on the mat."
  * Y2 = "There is a cat on the mat."
  * Suppose that the machine outputs the translation: "The cat the cat on the mat."
  * The bigrams in the machine output:
    | Pairs	| Count |	Count clip |
    | --- | --- | --- |
    | the cat	| 2	| 1 (Y1) |
    | cat the	| 1	| 0 |
    | cat on	| 1	| 1 (Y2) |
    | on the	| 1	| 1 (Y1 & Y2) |
    | the mat | 1 |	1 (Y1 & Y2) |
    | Totals |	6	| 4 |
      * `Modified precision = sum(Count clip) / sum(Count) = 4/6`
* So here are the equations for modified precision for the n-grams case:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/7879ec7a-10d2-4f4a-89c9-2889bc1015bb.png" width="700" />

* Let's put this together to formalize the BLEU score:
  * P<sub>n</sub> = Bleu score on one type of n-gram
  * **Combined BLEU score** = BP * exp(1/n * sum(Pn))
    * e.g., if we want BLEU for 4, we compute P<sub>1</sub>, P<sub>2</sub>, P<sub>3</sub>, P<sub>4</sub> and then average them and take the exp
  * BP stands for **brevity penalty**. It turns out that if a machine outputs a small number of words it will get a better score so we need to handle that

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/3e515cbe-188e-4db5-8894-23e13061fe28.png" width="700" />

* BLEU score has several open-source implementations
* It is used in a variety of systems like machine translation and image captioning

### Attention Model Intuition
* So far we were using sequence to sequence models with an encoder and decoders, but there is a technique called *attention* which makes these models even better
* The attention idea has been one of the most influential ideas in deep learning
* The problem of long sequences:
  * Given this model, inputs, and outputs:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/95b649c9-af50-4923-8188-9fb1ae0fe0b5.png" width="700" />

  * The encoder should memorize this long sequence into one vector, and the decoder has to process this vector to generate the translation
* If a human would translate this sentence, he/she wouldn't read the whole sentence and memorize it then try to translate it but rather translate a part each time
* The performance of this model decreases if a sentence is long
* We will discuss the attention model that works like a human that looks at parts at a time. That will significantly increase the accuracy even with longer sequence:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/42c1154c-e898-43d0-b76b-9cdae793fe89.png" width="700" />

  * Blue is the normal model, while green is the model with attention mechanism.
* In this section we will give just some intuitions about the attention model and in the next section we will discuss its details
* The attention model was descried in this paper:
  * [Bahdanau et al., 2014. Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473)
* Now for the intuition:
  * Suppose that our encoder is a bidirectional RNN: 

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/3f7894f1-30fa-4b78-b5b4-b15c4e43dc0c.png" width="700" />

  * We pass the French sentence to the encoder whichh generates a vector that represents the inputs
  * Now to generate the first word in English, which is *"Jane"*, we will make another RNN, which is the decoder
  * Attention weights are used to specify which words are needed when generating a word. So to generate *"Jane"* we will look at *"jane"*, *"visite"*, *"l'Afrique"*

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/a8564e80-4b2b-4c2a-a230-48204798b86d.png" width="700" />

    * S<sup><0></sup>, S<sup><1></sup>,... is the hidden State (S) activation of the other RNN (same as a<sup><0></sup>, a<sup><1></sup>,... but denoted with symbol S to avoid confusion
    * alpha<sup><1,1></sup>, alpha<sup><1,2></sup>, and alpha<sup><1,3></sup> are the attention weights being used
  * And so to generate any word there will be a set of attention weights that controls which words we are looking at right now

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/ee650961-dc8a-4323-acfa-5cfcb0538010.png" width="700" />

    * With alpha<sup><3, t></sup> can be affected by a<sup>< t ></sup> forward, alpha<sup>< t ></sup> backward (as we used the Bidirectional RNN) and the state from previous step S<sup><2></sup> => how much you pay attention to S<sup><3></sup>

### Attention Model
* Let's formalize the intuition from the last section into the exact details on how this can be implemented
* First we will have an bidirectional RNN (most common is LSTMs) that encodes French language:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/4a6a79da-1c53-4d1f-9c46-4743ec8874f3.png" width="700" />

  * For learning purposes, let's assume that a<sup><t'></sup> will include both directions activations at time step t' a<sup><t'></sup> = (a<sup><t'></sup> forward, a<sup><t'></sup> backward)
* We will have a unidirectional RNN to produce the output using a context 'c' which is computed using the attention weights, which denote how much information does the output needs to look in a<sup><t'></sup>

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/12fa0c39-bd3e-4c5f-9299-084bacc59dcd.png" width="700" />

* Sum of the attention weights for each element in the sequence should be 1:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/c854251a-69fd-4015-90d2-bc7838b9b509.png" width="250" />

* The context 'c' is calculated using this equation:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/43f43b2c-bafb-421d-a01a-68aa6029cfaf.png" width="250" />

* Compute the attention weights alpha<sup><t, t'></sup>:
  * So alpha<sup><t, t'></sup> = amount of attention y<sup>< t ></sup> should pay to a<sup><t'></sup>
    * Like for example we payed attention to the first three words through alpha<1,1>, alpha<1,2>, alpha<1,3>
  * We are going to softmax the attention weights so that their sum is 1:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/d842d9d4-93eb-40ca-89da-55bb6229c55e.png" width="250" />

  * Now we need to know how to calculate e<sup><t, t'></sup>. We will compute 'e' using a small neural network (usually 1-layer, because we will need to compute this a lot):

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/e958211a-5395-4c5b-80ca-bbe8bd992c6e.png" width="250" />

    * S<sup><t-1></sup> is the hidden state 'S' of the RNN, and a<sup><t'></sup> is the activation of the other bidirectional RNN
  * One of the disadvantages of this algorithm is that it takes quadratic time or quadratic cost to run
  * Other application of attention in image captioning: [Xu et al., 2015, Show, attend and tell: Neural image caption generation with visual attention](https://arxiv.org/abs/1502.03044)
* One fun way to see how attention works is by visualizing the attention weights:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/4aeae855-c09a-45ae-87cf-7027fc5a8987.png" width="250" />

## Speech Recognition (Audio Data)
### Speech Recognition
* One of the most exciting developments using sequence-to-sequence models has been the rise of very accurate speech recognition
* Let's define the speech recognition problem:
  * X: audio clip
  * Y: transcript
  * If you plot an audio clip it will look like this:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/37abc6fb-f43c-40a1-96bc-dafe4344aab6.png" width="250" />

    * The horizontal axis is time while the vertical is changes in air pressure.
  * What really is an audio recording? A microphone records little variations in air pressure over time, and it is these little variations that your ear perceives as sound. You can think of an audio recording as a long list of numbers measuring the little air pressure changes detected by the microphone. We will use audio sampled at 44100 Hz (or 44100 Hertz). This means the microphone gives us 44100 numbers per second. Thus, a 10 second audio clip is represented by 441000 numbers (= 10 * 44100).
  * It is quite difficult to work with "raw" representations of audio
  * Because even the human ear doesn't process raw wave forms, the human ear can process different frequencies
  * There's a common preprocessing step for an audio - generate a spectrogram which works similarly to human ears

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/18c3cc51-d0d1-41b0-ba1e-5c9d0c2c0253.png" width="250" />

    * The horizontal axis is time while the vertical is frequencies. Intensity of different colors shows the amount of energy - how loud is the sound - for different frequencies (a human ear does a very similar preprocessing step).
  * A spectrogram is computed by sliding a window over the raw audio signal, which calculates the most active frequencies in each window using a Fourier transformation
  * In the past, speech recognition systems were built using phonemes that are a hand-engineered basic units of sound. Linguists used to hypothesize that writing down audio in terms of these basic units of sound called phonemes   would be the best way to do speech recognition.
  * End-to-end deep learning found that phonemes was no longer needed. One of the things that made this possible is the large audio datasets
  * Research papers have around 300 - 3000 hours of training data while the best commercial systems are now trained on over 100,000 hours of audio
* You can build an accurate speech recognition system using the attention model that we have descried in the previous section:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/16a135ce-f565-44f5-9e5c-020a20e05341.png" width="700" />

* One of the methods that seem to work well is **CTC cost**, which stands for "Connectionist temporal classification"
  * To explain this let's say that Y = "the quick brown fox"
  * We are going to use an RNN with input, output structure:

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/c6164e23-0e21-471c-b7f9-b09483881ce5.png" width="500" />

  * Note: this is a unidirectional RNN, but in practice a bidirectional RNN is used
  * Notice that the number of inputs and number of outputs are the same here, but in speech recognition problem input X tends to be a lot larger than output Y
    * 10 seconds of audio at 100Hz gives us X with shape (1000, ). These 10 seconds don't contain 1000 character outputs.
  * The CTC cost function allows the RNN to output something like this:
    * `ttt_h_eee<SPC>___<SPC>qqq___` - this covers "the q".
    * The _ is a special character called "blank" and <SPC> is for the "space" character.
    * Basic rule for CTC: collapse repeated characters not separated by "blank"
  * So the 19 characters in our Y can be generated into 1000 character output using CTC and it's special blanks
  * The ideas were taken from this paper:
    * [Graves et al., 2006, Connectionist Temporal Classification: Labeling unsegmented sequence data with recurrent neural networks](https://dl.acm.org/doi/10.1145/1143844.1143891)
    * This paper's ideas were also used by Baidu's DeepSpeech
* Using both attention model and CTC cost can help you to build an accurate speech recognition system

### Trigger Word Detection
