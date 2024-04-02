# Learning Objectives
* Create positional encodings to capture sequential relationships in data
* Calculate scaled dot-product self-attention with word embeddings
* Implement masked multi-head attention
* Build and train a Transformer model
* Fine-tune a pre-trained transformer model for Named Entity Recognition
* Fine-tune a pre-trained transformer model for Question Answering
* Implement a QA model in TensorFlow and PyTorch
* Fine-tune a pre-trained transformer model to a custom dataset
* Perform extractive Question Answering

## Transformers
### Transformer Network IntuitionSelf-Attention
* Many of the most effective algorithms NLP (and also Computer Vision) today are based on Transformer architecture
  * Transformers have a relatively complex neural network architecture
* Previous sequence models: RNN => GRU => LSTM
  * Improved control over the flow of information, also increase in computation complexity
  * They still ingested the input one word/token at the time
    * So each unit is like a bottleneck to the flow of information. As to compute the output of the final unit, you have to compute the outputs of all units that come before
* ***Transformer Network Intuition***: allows you to run these computations for an entire sequence in parallel
  * You can ingest the sequence all at oncec rather than just processing words one by one from left to right
  * Paper: [Vaswani et al., 2017, Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  * Transformer: Combination of the use of Attention representations with CNN style of processing
    * RNN may process one output at a time
      * Attention is a way of computing very rich, very useful representations of words
    * CNN takes a lot of inputs (pixels) and computes representations in parallel
  * Two key ideas:
    * Self-Attention: compute n representations for n words in parallel
    * Multi-Head Attention: basically a for loop over self-attention => multiple versions of these representations

### Self-Attention
* Example: *"Jane visite l'Afrique en semptembre"*
* For each word, we have:
  * A(q, K, V) = attention-based vector representation of a word
  * Calculate for each word, we have A<sup><1></sup>,...,A<sup><1></sup>
    * Normally, one way to represent a particular word is using word embedding. But depending on the **context we might choose to represent that word differently** (like l'Afrique can be historical interest, continent, festival destination,...)
  * With a particular representation, i.e. A<sup><3></sup> l'Afrique, it will look at the surrounding words to try to figure out the context i.e., how we talk about Africa, then find the most appropriate representation for this
    * It is simmilar to the Attention mechanism discussed in the previous week as applied for RNNs, but this time the model computes these representations in parallel for all five words in a sentence
* Equation for A(q, k, v):

  <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/066c0212-3646-4df3-869a-f188feb5750e.png" width="360" />

  * For every word, we have 3 values: query q<sup><i></sup>, key K<sup><i></sup>, and value V<sup><i></sup>
    * q<sup><i></sup> lets you ask a question about that word i, like what's happening there?
    * K<sup><i></sup> helps you figure out if this word gives the most relevant answer to that question, i.e. K<sup><1></sup> is a person, or an action
    * V<sup><i></sup> determines if this particular word i should be represented within the representation A<sup><i></sup>
  * To calculate a representation A<sup><3></sup>:
    * First:
      * q<sup><3></sup> = Wq . x<sup><3></sup> (with W<sub>q</sub> is a learned weight matrix)
      * K<sup><3></sup> = WK . x<sup><3></sup>
      * V<sup><3></sup> = WV . x<sup><3></sup>
    * Second, we calculate the inner product between q<sup><3></sup> and other K<sup><1:5></sup> => how good is a word i.e. visite is an answer to the question of what happened in l'Afrique
      * Then compute the Softmax for each (with the blue block (q<sup><3></sup>.K<sup><2></sup> has the largest value))
    * Then, we multiply with the corresponding V<sup><i></sup> and sum them up to get A<sup><3></sup>

      <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/a2a7a7a4-ae69-46ca-927c-36120e65f228.png" width="700" />

      * The key advantage of this computation is the word *'l'Afrique'* lets the self-attention mechanism realise that this word is the destination of a *'visite'*, rather than a fixed word embedding => enables compute richer, more useful representation of this word
    * Eventualy, we do the same with other word representations
* Finally, the Self-Attention is calculated by:
  * `Attention(Q, K, V) = softmax(Q.dot(K.T)/sqrt(d_k)).V`
  * This formula is the reason why the Self-Attention is also called the scaled dot-product attention

### Multi-Head Attention
* Each time you calculate the self-attention is called the head
* With multi-head attention, you take the same set of [query, key, value] as input and calculate multiple self-attentions for the Attention (W<sup>Q1</sup>Q, W<sup>K1</sup>K, W<sup>V1V</sup>) (visite) to answer 'what's happened?'
* Then, we repeat the process with a new set of W<sup>Q2</sup>, W<sup>K2</sup>, W<sup>V2</sup> values, to answer the question i.e, When?, and so on (Who, what,...)
  * For each, we have a different attention for different word, like red for September (when), black for Jane (who),... and this time, we call it a head
  * These heads are then concatentated together, and called Multi-Head Attention:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/36da17c5-302e-47dd-b864-73e3c3257128.png" width="700" />

* The Multi-Head Attention is calculated by:
  * Multihead(Q,K,V) = concat(heads).W0

### 


