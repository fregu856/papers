##### [20-03-26] [paper93]
- Conservative Uncertainty Estimation By Fitting Prior Networks [[pdf]](https://openreview.net/forum?id=BJlahxHYDS) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Conservative%20Uncertainty%20Estimation%20By%20Fitting%20Prior%20Networks.pdf)
- *Kamil Ciosek, Vincent Fortuin, Ryota Tomioka, Katja Hofmann, Richard Turner*
- `2019-10-25, ICLR 2020`

****

### General comments on paper quality:
- Interesting and somewhat well-written paper.

### Comments:
- I found it quite difficult to actually understand the method at first, I think the authors could have done a better job describing it.

- I guess that "f" should be replaced with "f_i" in equation (2)?

- "...the obtained uncertainties are larger than ones arrived at by Bayesian inference.", I did not quite get this though. The estimated uncertainty is conservative w.r.t. the posterior process associated with the prior process (the prior process defined by randomly initializing neural networks), but only if this prior process can be assumed to be Gaussian? So, do we actually have any guarantees? I am not sure if the proposed method actually is any less "hand-wavy" than e.g. ensembling.

- The experimental results seem quite promising, but I do not agree that this is "an extensive empirical comparison" (only experiments on CIFAR-10).
