##### [19-01-27] [paper36]
- Weight Uncertainty in Neural Networks [[pdf]](https://arxiv.org/abs/1505.05424) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Weight%20Uncertainty%20in%20Neural%20Networks.pdf)
- *Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra*
- `2015-05-20, ICML2015`

****

### General comments on paper quality:
- Well-written and interesting paper. I am not particularly familiar with variational methods, but still found the paper quite easy to read and understand.

### Comments:
- Seems like a good starting point for learning about variational methods applied to neural networks. The theory is presented in a clear way. The presented method also seems fairly straightforward to implement.

- They mainly reference _"Keeping Neural Networks Simple by Minimizing the Description Length of the Weights"_ and _"Practical Variational Inference for Neural Networks"_ as relevant previous work.

- In equation (2), one would have to run the model on the data for multiple weight samples? Seems quite computationally expensive?

- Using a diagonal Gaussian for the variational posterior, I wonder how much of an approximation that actually is? Is the true posterior e.g. very likely to be multi-modal?

- The MNIST models are only evaluated in terms of accuracy. The regression experiment is quite neat (good to see that the uncertainty increases away from the training data), but they provide very little details. I find it difficult to draw any real conclusions from the Bandits experiment.
