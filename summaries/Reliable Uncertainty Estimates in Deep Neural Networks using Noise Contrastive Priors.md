##### [18-10-18] [paper12]
- Reliable Uncertainty Estimates in Deep Neural Networks using Noise Contrastive Priors [[pdf]](https://arxiv.org/abs/1807.09289) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Reliable%20Uncertainty%20Estimates%20in%20Deep%20Neural%20Networks%20using%20Noise%20Contrastive%20Priors_.pdf)
- *Danijar Hafner, Dustin Tran, Alex Irpan, Timothy Lillicrap, James Davidson*
- `2018-07-24, ICML2018 Workshop`

****

### General comments on paper quality:
Well-written paper, the proposed method is fairly clearly explained.

### Paper overview:
The authors present a method called Noise Contrastive Priors (NCPs). The key idea is to train a model to output high **epistemic/model** uncertainty for data points which lie outside of the training distribution (out-of-distribution data, OOD data). To do this, NCPs add noise to some of the inputs during training and, for these noisy inputs, try to minimize the KL divergence to a wide prior distribution.

NCPs do NOT try to add noise only to the subset of the training data which lie close to the boundary of the training data distribution, but instead add noise to any input data. Empirically, the authors saw no significant difference in performance between these two approaches.

The authors apply NCPs both to a small Bayesian Neural Network and to an OOD classifier model, and experimentally evaluate their models on regression tasks.

In the OOD classifier model, the network is trained to classify noise-upped inputs as OOD and non-modified inputs as "in distribution". During testing, whenever the model classifies an input as OOD, the model outputs the parameters of a fixed wide Gaussian N(0, sigma_y) instead of the mean and variance outputted by the neural network.

### Comments:
Somewhat interesting method, although I must say it seems quite ad hoc. Not sure if just adding random noise to the LiDAR point clouds would be enough to simulate OOD data in the case of LiDAR-only 3DOD. Could be worth trying though I suppose.

