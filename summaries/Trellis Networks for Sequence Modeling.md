##### [19-10-04] [paper61]
- Trellis Networks for Sequence Modeling [[pdf]](https://arxiv.org/abs/1810.06682) [[code]](https://github.com/locuslab/trellisnet) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Trellis%20Networks%20for%20Sequence%20Modeling.pdf)
- *Shaojie Bai, J. Zico Kolter, Vladlen Koltun*
- `2018-10-15, ICLR2019`

****

### General comments on paper quality:
- Well-written and quite interesting paper.


### Comments:
- Interesting model, quite neat indeed how it can be seen as a bridge between RNNs and TCNs.

- The fact that they share weights across all network layers intuitively seems quite odd to me, but I guess it stems from the construction based on M-truncated RNNs?

- It is not obvious to me why they chose to use a gated activation function based on the LSTM cell, would using a "normal" activation function (e.g. ReLu) result in a significant drop in performance?
