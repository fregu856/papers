##### [18-10-18] [paper13]
- Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles [[pdf]](https://arxiv.org/abs/1612.01474) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles_.pdf)
- *Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell*
- `2017-11-17, NIPS2017`

****

### General comments on paper quality:
- Well-written and interesting paper. The proposed method is simple and also very clearly explained.

### Paper overview:
- The authors present a simple, non-Bayesian method for estimating predictive uncertainty (epistemic/model uncertainty + aleatoric uncertainty) in neural networks, based on the concept of **ensembling**.

- For regression, they train an ensemble of M networks which output both a mean and a variance (y given x is assumed to be Gaussian) by minimizing the corresponding negative log-likelihood (similar to how [Kendall and Gal](https://github.com/fregu856/papers/blob/master/summaries/What%20Uncertainties%20Do%20We%20Need%20in%20Bayesian%20Deep%20Learning%20for%20Computer%20Vision%3F.md) model aleatoric uncertainty).

- For classification, they train an ensemble of M networks with the standard softmax output layer.

- Each of the M networks in an ensemble is independently trained on the entire training dataset, using random initialization of the network weights and random shuffling of the training data. Typically, they set M=5.

- For classification, the predicted probabilities are averaged over the M networks during inference.

- For regression, the final mean is computed as the average of the means outputted by the individual networks (mu_final = (1/M)*sum(mu_i)), whereas the final variance sigma_final^2 = (1/M)*sum(mu_i^2 - mu_final^2) + (1/M)*sum(sigma_i^2).

- The authors experimentally evaluate their method on various regression (1D toy problem as well as real-world datasets) and classification tasks (MNIST, SVHN and ImageNet). They find that their method generally outperforms (or a least matches the performance of) related methods, specifically MC-dropout.

- They also find that when training a classification model on a certain dataset and then evaluating the model on a separate dataset containing unseen classes, their model generally outputs larger uncertainty (larger entropy) than the corresponding MC-dropout model (which is a good thing, we don't want our model to produce over-confident predictions, we want the model to "know what it doesn't know").

### Comments:
- Conceptually very simple, yet interesting method. The key drawback of using ensembling, especially in real-time applications, is of course that is requires running M networks to obtain a single prediction. However, if a relatively small ensemble size as e.g. M=5 is enough to obtain high-quality uncertainty estimates, it shouldn't really be impossible to still achieve real-time inference speed (50 Hz single-model is needed to obtain 10 Hz in that case). I do actually find this method quite interesting.
