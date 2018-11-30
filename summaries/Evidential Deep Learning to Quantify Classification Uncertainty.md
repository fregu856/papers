##### [18-11-29] [paper23]
-  Evidential Deep Learning to Quantify Classification Uncertainty [[pdf]](https://arxiv.org/abs/1806.01768) [[poster]](https://muratsensoy.github.io/NIPS18_EDL_poster.pdf) [[code example]](https://muratsensoy.github.io/uncertainty.html) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Evidential%20Deep%20Learning%20to%20Quantify%20Classification%20Uncertainty.pdf)
- *Murat Sensoy, Lance Kaplan, Melih Kandemir*
- `2018-10-31, NeurIPS2018`

****

### General comments on paper quality:
- Well-written and very interesting paper. I had to read it a couple of times to really start understanding everything though. 

### Paper overview:
- The authors present a classification model in which they replace the standard softmax output layer with an output layer that outputs parameters of a Dirichlet distribution ([resource1](https://en.wikipedia.org/wiki/Dirichlet_distribution), [resource2](https://endymecy.gitbooks.io/spark-ml-source-analysis/content/%E8%81%9A%E7%B1%BB/LDA/docs/dirichlet.pdf)). I.e., they assume a Dirichlet output distribution, similar to [Gast and Roth](https://github.com/fregu856/papers/blob/master/summaries/Lightweight%20Probabilistic%20Deep%20Networks.md). The authors interpret the behavior of this predictor from an evidential reasoning / subjective logic perspective (two terms which I am unfamiliar with): *"By placing a Dirichlet distribution on the class probabilities, we treat predictions of a neural net as subjective opinions and learn the function that collects the evidence leading to these opinions by a deterministic neural net from data"*. 

- Instead of outputting just a point estimate of the class probabilities (the softmax scores), the network thus outputs the parameters of a distribution over the class probabilities (similar to how a network can output the parameters of a Gaussian instead of just a point estimate in the regression case).

- The only difference in network architecture is that they replace the softmax layer with a ReLU layer (to get non-negative values) to obtain e_1, ..., e_K (K is the number of classes). The parameters alpha_1, ..., alpha_K of the Dirichlet distribution is then set to alpha_i = e_i + 1 (which means alpha_i >= 1, i.e., they are restricting the set of Dirichlet distributions their model can predict? They are setting a maximum value for the variance?). Given this, the Dirichlet mean, alpha/S (S = sum(alpha_i)), is taken as the class probabilities estimate.

- The authors present three different possible loss functions (which are all different from the one used by [Gast and Roth](https://github.com/fregu856/papers/blob/master/summaries/Lightweight%20Probabilistic%20Deep%20Networks.md)?), which all involve averaging over the predicted Dirichlet pdf, and choose one based on their empirical findings. They claim that this chosen loss corresponds to learned loss attenuation (but I struggle somewhat to actually see why that is so). They then also add a KL divergence term to this loss, penalizing divergence from a uniform distribution (which strikes me as somewhat ad hoc?). 

- They train their model on MNIST (digits) and then evaluate on notMNIST (letters), expecting a large proportion of predictions to have maximum entropy (maximum uncertainty). They also do a similar experiment using CIFAR10, training on images of the first 5 classes and then evaluating on images of the remaining 5 classes. 

- They compare their model with e.g. [MC-dropout](https://github.com/fregu856/papers/blob/master/summaries/What%20Uncertainties%20Do%20We%20Need%20in%20Bayesian%20Deep%20Learning%20for%20Computer%20Vision%3F.md) and [Deep Ensembles](https://github.com/fregu856/papers/blob/master/summaries/Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles.md), and find that their model achieves similar test set performance (on MNIST / the first 5 classes of CIFAR10), while producing significantly better uncertainty estimates (their model outputs maximum entropy predictions more frequently when being fed images of unseen classes).

- They also do an experiment with adversarial inputs, finding that their model has a similar drop in prediction accuracy, while being less confident in its predictions (which is a good thing, you don't want the model to become overconfident, i.e., misclassify inputs but still being confident in its predictions). 

### Comments:
- Really interesting paper. It also made me go back and read [Gast and Roth](https://github.com/fregu856/papers/blob/master/summaries/Lightweight%20Probabilistic%20Deep%20Networks.md) much more carefully. 

- Just like I think it makes a lot of sense to assume a Gaussian/Laplacian output distribution in the regression case, it does intuitively seem reasonable to assume a Dirichlet output distribution in classification. As indicated by the fact that the authors and [Gast and Roth](https://github.com/fregu856/papers/blob/master/summaries/Lightweight%20Probabilistic%20Deep%20Networks.md) choose different loss functions (and estimate the Dirichlet parameters in different ways), it is however not at all as clear to me what actually should the best / most natural way of doing this.
