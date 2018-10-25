##### [18-10-19] [paper14]
- Uncertainty in Neural Networks: Bayesian Ensembling [[pdf]](https://arxiv.org/abs/1810.05546) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20in%20Neural%20Networks:%20Bayesian%20Ensembling_.pdf)
- *Tim Pearce, Mohamed Zaki, Alexandra Brintrup, Andy Neel*
- `2018-10-12, AISTATS2019 submission`

****

### General comments on paper quality:
- Well-written and interesting paper. Compares different ensembling techniques and techniques for approximate Bayesian inference in neural networks.

### Paper overview:
- The authors present **randomized anchored MAP sampling**, **anchored ensembles** for short, a somewhat modified ensembling process aimed at estimating predictive uncertainty (epistemic/model uncertainty + aleatoric uncertainty) in neural networks.

- They independently train M networks (they set M to just 5-10) on the entire training dataset, just like e.g. [Lakshminarayanan et al.](https://github.com/fregu856/papers/blob/master/summaries/Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles.md). The key difference is that they regularize the network parameters about values drawn from a prior distribution (instead of about 0 as in standard L2 regularization). In practice, each network in the ensemble is regularized, or "anchored", about its initialization parameter values (which are drawn from some prior Gaussian).

- This procedure is motivated/inspired by the Bayesian inference method called *randomized MAP sampling*, which (roughly speaking) exploits the fact that adding a regularization term to the standard MLE loss function results in a MAP parameter estimate. Injecting noise into this loss (to the regularization term) and sampling repeatedly (i.e., ensembling) produces a distribution of MAP estimates which roughly corresponds to the true parameter posterior distribution.

- What this injected noise should look like is however difficult to find in complex cases like NNs. What the authors do is that they study the special case of single-layer, **wide** NNs and claim that ensembling here will approximate the true posterior if the parameters theta of each network are L2-regularized about theta_0 ~ N(mu_prior, sigma_prior).

- The authors do NOT let the network output an estimate of the aleatoric uncertainty like  [Lakshminarayanan et al.](https://github.com/fregu856/papers/blob/master/summaries/Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles.md) do, instead they assume a constant aleatoric uncertainty estimate sigma_epsilon^2. They then use their ensemble to compute the predictive mean y_hat = (1/M)*sum(y_hat_i) and variance sigma_y^2 = (1/(M-1))*sum((y_hat_i - y_hat)^2) + sigma_epsilon^2.

- They evaluate their method on various regression tasks (no classification whatsoever) using single-layer NNs. For 1D regression they visually compare to the analytical GP ("gold standard" but not scalable), Hamiltonian MC ("gold standard" but not scalable), a Variational Inference method (scalable) and MC dropout (scalable). They find that their method here outperforms the other scalable methods.

- They also compared their results on a regression benchmark with the method by [Lakshminarayanan et al.](https://github.com/fregu856/papers/blob/master/summaries/Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles.md) ("normal" ensemble of the same size) which is the current state-of-the-art (i.e., it outperforms e.g. MC dropout). They find that their method outperforms the other in datasets with low aleatoric noise/uncertainty, whereas [Lakshminarayanan et al.](https://github.com/fregu856/papers/blob/master/summaries/Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles.md) is better on datasets with high aleatoric noise. The authors say this is because [Lakshminarayanan et al.](https://github.com/fregu856/papers/blob/master/summaries/Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles.md) explicitly tries to model the aleatoric noise (network outputs both mean and variance).

### Comments:
- Interesting method which is actually very similar to the very simple method by [Lakshminarayanan et al.](https://github.com/fregu856/papers/blob/master/summaries/Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles.md). The only real difference is that you add this regularization about the random initial parameter values, which shouldn't be too difficult to implement in e.g. PyTorch?

- The fact that you just seem to need 5-10 networks in the ensemble ([Lakshminarayanan et al.](https://github.com/fregu856/papers/blob/master/summaries/Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles.md) also used 5-10 networks) also makes the method somewhat practically useful even in real-time applications.

- Would be very interesting to add this regularization to the [Lakshminarayanan et al.](https://github.com/fregu856/papers/blob/master/summaries/Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles.md) method, apply to 3DOD and compare to [Lakshminarayanan et al.](https://github.com/fregu856/papers/blob/master/summaries/Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles.md), MC dropout etc.

- Perhaps the Bayesian motivation used in this paper doesn't really hold for large CNNs (and even so, how do you know that a Gaussian is the "correct" prior for Bayesian inference in this case?), but it still makes some intuitive sense that adding this random regularization could increase the ensemble diversity and thus improve the uncertainty estimate. 
