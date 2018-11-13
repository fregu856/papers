##### [18-11-12] [paper18]
- Large-Scale Visual Active Learning with Deep Probabilistic Ensembles [[pdf]](https://arxiv.org/abs/1811.03575) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Large-Scale%20Visual%20Active%20Learning%20with%20Deep%20Probabilistic%20Ensembles_.pdf)
- *Kashyap Chitta, Jose M. Alvarez, Adam Lesnikowski*
- `2018-11-08`

****

### General comments on paper quality:
- Quite well-written and very interesting paper. Reasonably easy to read.

### Paper overview:
- The authors introduce *Deep Probabilistic Ensembles (DPEs)*, a technique that utilizes a regularized ensemble to perform *approximate* variational inference in Bayesian Neural Networks (BNNs). They experimentally evaluate their method on the task of active learning for classification (CIFAR-10, CIFAR-100, ImageNet) and semantic segmentation (BDD100k), and somewhat outperform similar methods.

- In variational inference, one restricts the problem to a family of distributions over the network weights w, q(w) ~ D. One then tries to optimize for the member of this family D that is closest to the true posterior distribution in terms of KL divergence. This optimization problem is equivalent to the maximization of the Evidence Lower Bound (ELBO), which contains expectations over all possible q(w) ~ D.

- In this paper, the authors approximate these expectations by using an ensemble of E networks, which results in a loss function containing the standard cross-entropy term together with a regularization term OMEGA over the *joint set* of all parameters in the ensemble. Thus, the proposed method is an *approximation* of variational inference.

- They chose Gaussians for both the prior p(w) and q(w), *assume* mutual independence between the network weights and can then compute the regularization term OMEGA by independently computing it for each network weight w_i (each network in the ensemble has a value for this weight w_i) using equation 9, and then summing this up over all network weights. I.e., for each weight w_i, you compute mu_q, sigma_q as the sample mean and variance across the E ensemble networks and then use equation 9. Equation 9 will penalize variances much larger than that of the prior (so that the ensemble members do not diverge completely from each other), penalize variances smaller than that of the prior (promoting diversity) and keep the mean close to that of the prior.

- Note that the E ensemble networks have to be trained jointly, meaning that the memory requirement scales linearly with E.

- They experienced some difficulties when trying to train an ensemble of just E=2, 3 networks, as the regularization term caused instability and divergence of the loss. This problem was mitigated by setting E >= 4, and ended up using E=8 for all of their experiments (beyond E=8 they observed diminishing returns).

- In the experiments, they e.g. compare DPEs to using an ensemble trained using standard L2 regularization on all four datasets. DPEs are found to consistently outperform the standard ensemble, but the performance gain is not very big.
 
### Comments:
- Definitely an interesting method. Nice to see more than just an intuitive argument for why ensembling seems to provide reasonable uncertainty estimates, even though the derivation contains multiple approximations (variational inference approximation, approximation of the expectations). 

- I'm not sure how significant the performance gain compared to standard ensembling actually is though, I would like to see more comparisons also outside of the active learning domain. Would also be interesting to compare with [Bayesian ensembling](https://github.com/fregu856/papers/blob/master/summaries/Uncertainty%20in%20Neural%20Networks:%20Bayesian%20Ensembling.md).
