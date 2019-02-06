##### [19-02-06] [paper39]
- Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks [[pdf]](https://arxiv.org/abs/1502.05336) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Probabilistic%20Backpropagation%20for%20Scalable%20Learning%20of%20Bayesian%20Neural%20Networks.pdf)
- *José Miguel Hernández-Lobato, Ryan P. Adams*
- `2015-07-15, ICML2015`

****

### General comments on paper quality:
Quite well-written and interesting paper. I did however find it somewhat difficult to fully understand the presented method.

### Comments:
I find it difficult to compare this method (PBP, which is an Assumed Density Filtering (ADF) method) with Variational Inference (VI) using a diagonal Gaussian as q. The authors seem to argue that their method is superior because it only employs one stochastic approximation (sub-sampling the data), whereas VI employs two (in VI one also approximates an expectation using Monte Carlo samples). In that case I guess that PBP should be very similar to [_Deterministic Variational Inference for Robust Bayesian Neural Networks_](https://openreview.net/forum?id=B1l08oAct7)?

I guess it would be quite difficult to extend this method to CNNs?
