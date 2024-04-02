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

  test test test
