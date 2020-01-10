##### [20-01-08] [paper76]
- Practical Deep Learning with Bayesian Principles [[pdf]](https://arxiv.org/abs/1906.02506) [[code]](https://github.com/team-approx-bayes/dl-with-bayes) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Practical%20Deep%20Learning%20with%20Bayesian%20Principles.pdf)
- *Kazuki Osawa, Siddharth Swaroop, Anirudh Jain, Runa Eschenhagen, Richard E. Turner, Rio Yokota, Mohammad Emtiyaz Khan*
- `2019-06-06, NeurIPS 2019`

****

### General comments on paper quality:
- Interesting and quite well-written paper.

### Comments:
- To me, this mainly seems like a more practically useful alternative to [Bayes by Backprop](https://github.com/fregu856/papers/blob/master/summaries/Weight%20Uncertainty%20in%20Neural%20Networks.md), scaling up variational inference to e.g. ResNet on ImageNet. The variational posterior approximation q is still just a diagonal Gaussian.

- I still do not fully understand natural-gradient variational inference.

- Only image classification is considered.

- It seems to perform ish as well as Adam in terms of accuracy (although it is 2-5 times slower to train), while quite consistently performing better in terms of calibration (ECE).

- The authors also compare with MC-dropout in terms of quality of the predictive probabilities, but these results are IMO not very conclusive. 
