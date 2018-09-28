##### [18-09-25] [paper5]
- Deep Confidence: A Computationally Efficient Framework for Calculating Reliable Errors for Deep Neural Networks [[pdf]](https://arxiv.org/abs/1809.09060) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Confidence:%20A%20Computationally%20Efficient%20Framework%20for%20Calculating%20Reliable%20Errors%20for%20Deep%20Neural%20Networks.pdf)
- *Isidro Cortes-Ciriano, Andreas Bender*
- `2018-09-24`

****

### General comments on paper quality:
- Unfamiliar paper formatting (written by authors from a different field), but actually a well-written and interesting paper. The methods are quite clearly explained.

### Paper overview:
- The authors present a framework for computing "valid and efficient" confidence intervals for individual predictions made by a deep learning network, applied to the task of drug discovery (modeling bioactivity data).

- To create confidence intervals, they use an ensemble of 100 networks (either obtained by independently training 100 networks, or by using *Snapshot Ensembling*: saving network snapshots during the training of a single network (essentially)) together with a method called *conformal prediction*.

- More specifically, predictions are produced by the 100 networks for each example in the validation set, and for each example the sample mean y_hat and sample std sigma_hat are computed. From y_hat and sigma_hat, for each example, a non-comformity value alpha is computed (alpha = |y - y_hat|/exp(sigma_hat)). These non-conformity values are then sorted in increasing order, and the percentile corresponding to the chosen confidence level (e.g. 80%, alpha_80) is selected (I suppose what you actually do here is that you find the smallest alpha value which is larger than 80% of all alpha values?).

- On the test set, for each example, again a prediction is produced by the 100 networks and y_hat, sigma_hat are computed. A confidence interval is then given by: y_hat +/- exp(sigma_hat)*alpha_80.

- The key result in the paper is that the authors find that these confidence intervals are indeed valid. I.e., when they compute the percentage of examples in the test set whose true values lie within the predicted confidence interval, this fraction was equal to or greater than the selected confidence level.

- They also compare the constructed confidence intervals with the ones obtained by a random forest model, and find that they have comparable efficiency (confidence intervals have a small average size -> higher efficiency). 

### Comments:
- Since the paper comes from an unfamiliar field (and I'm not at all familiar with the used datasets etc.),  I'm being somewhat careful about what conclusions I draw from the presented results, but it definitely seems interesting and as it potentially could be of practical use.

- The specific method used, i.e. an ensemble of 100 models, isn't really applicable in my case since it would make inference too slow, but the approach for constructing and validating confidence intervals might actually be useful. 

- For example, what would happen if you replace y_hat, sigma_hat with the predicted mean and std by a single model (aleatoric uncertainty)? Would the induced confidence intervals still be valid, at least for some confidence level?

- And if so, I suppose it doesn't really matter that the network in a sense is still just a black box, since you now actually are able to quantify its uncertainty on this data, and you at least have some kind of metric to compare different models with (obviously, you still don't know how much the output actually can be trusted when the network is applied to completely different data, but the fact that the confidence intervals are valid on this specific dataset is still worth something).
