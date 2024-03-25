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

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/adb5322d-2f0c-4ba0-a3eb-351e356343e1.png" width="300" />

  * Our architecture will include an **encoder** and **decoder**
  * The encoder is an RNN - LSTM or GRU are included - and takes the input sequence and then outputs a vector that should represent the whole input
  * After that the decoder network, also an RNN, takes the sequence built by the encoder and outputs the new sequence:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/7a141172-d6aa-4108-85a2-282c5b30caa4.png" width="300" />

  * These ideas are from the following papers:
    * [Sutskever et al., 2014, Sequence to sequence learning with neural networks](https://arxiv.org/abs/1409.3215)
    * [Cho et al., 2014, Learning phrase representations using RNN encoder-decoder for statistical machine translation](https://arxiv.org/abs/1406.1078)
* An architecture similar to the mentioned above works for image captioning problem:
  * In this problem X is an image, while Y is a sentence sequence (caption)
  * The model architecture image:

    <img src="https://github.com/mauritsvzb/DeepLearning.AI-Deep-Learning-Specialization/assets/13508894/5e769ad2-6066-4e06-8fa8-298039aee74d.png" width="300" />

  * The architecture uses a pretrained CNN (like AlexNet) as an encoder for the image, and the decoder is an RNN
  * Ideas are from the following papers (they share similar ideas):
    [Mao et et. al., 2014, Deep captioning with multimodal recurrent neural networks](https://arxiv.org/abs/1412.6632)
    [Vinyals et. al., 2014, Show and tell: Neural image caption generator](https://arxiv.org/abs/1411.4555)
    [Karpathy and Li, 2015, Deep visual-semantic alignments for generating image descriptions](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)

### 



## Speech Recognition (Audio Data)
### Speech Recognition


### Trigger Word Detection
