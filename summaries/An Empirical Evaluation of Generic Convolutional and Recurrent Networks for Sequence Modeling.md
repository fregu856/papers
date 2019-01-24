##### [19-01-24] [paper31]
- An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling [[pdf]](https://arxiv.org/abs/1803.01271) [[code]](https://github.com/locuslab/TCN) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/An%20Empirical%20Evaluation%20of%20Generic%20Convolutional%20and%20Recurrent%20Networks%20for%20Sequence%20Modeling.pdf)
- *Shaojie Bai, J. Zico Kolter, Vladlen Koltun*
- `2018-04-19`

****

### General comments on paper quality:
- Well-written and interesting paper.

### Paper overview:
- _"We conduct a systematic evaluation of generic convolutional and recurrent architectures for sequence modeling. The models are evaluated across a broad range of standard tasks that are commonly used to benchmark recurrent networks. Our results indicate that a simple convolutional architecture outperforms canonical recurrent networks such as LSTMs across a diverse range of tasks and datasets, while demonstrating longer effective memory. We conclude that the common association between sequence modeling and recurrent networks should be reconsidered, and convolutional networks should be regarded as a natural starting point for sequence modeling tasks."_

- The authors introduce a quite straightforward CNN designed for sequence modeling, named Temporal Convolutional Network (TCN). They only consider the setting where the output at time t, y_t, is predicted using only the previously observed inputs, x_0, ..., x_t. TCN thus employs causal convolution (zero pad with kernel_size-1 at the start of the input sequence).

- To achieve a long effective history size (i.e., that the prediction for y_t should be able to utilize inputs observed much earlier in the input sequence), they use residual blocks (to be able to train deep networks, the effective history scales linearly with increased depth) and dilated convolutions. 

- They compare TCN with basic LSTM, GRU and vanilla-RNN models on a variety of sequence modeling tasks (which include polyphonic music modeling, word- and character-level language modeling as well as synthetic "stress test" tasks), and find that TCN generally outperforms the other models. The authors do however note that TCN is outperformed by more specialized RNN architectures on a couple of the tasks.

- They specifically study the effective history/memory size of the models using the Copy Memory task (Input sequences are digits of length 10 + T + 10, the first 10 are random digits in {1, ..., 8}, the last 11 are 9:s and all the rest are 0:s. The goal is to generate an output of the same length that is 0 everywhere, except the last 10 digits which should be a copy of the first 10 digits in the input sequence), and find that TCN *significantly* outperforms the LSTM and GRU models (which is a quite interesting result, IMO).

### Comments:
- Interesting paper that challenges the viewpoint of RNN models being the default starting point for sequence modeling tasks. The presented TCN architecture is quite straightforward, and I do think it makes sense that CNNs might be a very competitive alternative for sequence modeling.
