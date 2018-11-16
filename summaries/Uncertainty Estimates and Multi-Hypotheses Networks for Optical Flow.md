##### [18-11-16] [paper20]
- Uncertainty Estimates and Multi-Hypotheses Networks for Optical Flow [[pdf]](https://arxiv.org/abs/1802.07095) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20Estimates%20and%20Multi-Hypotheses%20Networks%20for%20Optical%20Flow_.pdf)
- *Eddy Ilg, Özgün Çiçek, Silvio Galesso, Aaron Klein, Osama Makansi, Frank Hutter, Thomas Brox*
- `2018-08-06, ECCV2018`
****

### General comments on paper quality:
- Well written and very interesting paper. A recommended read.

### Paper overview:
- The authors study uncertainty estimation in the domain of optical flow estimation, which is a pixel-wise regression problem.

- They compare multiple previously suggested uncertainty estimation methods:
- - Empirical ensembles (each model only outputs a single point estimate), using both MC-dropout, bootstrap ensembles and snapshot ensembles.
- -  Predictive models (the model outputs the parameters (e.g. mean and variance) of an assumed output distribution, trained using the corresponding negative log likelihood).
- - Predictive ensembles (ensemble of predictive models), using both MC-dropout, bootstrap ensembles and snapshot ensembles.
- - (A bootstrap ensemble is created by independently training M models on different (partially overlapping) subsets of the training data, whereas a snapshot ensemble is essentially created by saving checkpoints during the training process of a single model).

- For an empirical ensemble, the empirical mean and variance are taken as the mean and variance estimates (mu = (1/M)sum(mu_i), sigma^2 = (1/M)sum( (mu_i - mu)^2) )).

- A predictive model directly outputs estimates of the mean and variance (the authors assume a Laplacian output distribution, which corresponds to an L1 loss).

- For a predictive ensemble, the outputted mean and variance estimates are combined into the final estimates (mu = (1/M)sum(mu_i), sigma^2 = (1/M)sum( (mu_i - mu)^2) ) + (1/M)sum(sigma^2_i) )

- Since all of the above methods require multiple forward passes to be computed during inference (obviously affecting the inference speed), the authors also propose a multi-headed predictive model architecture that yields multiple hypotheses (each hypothesis corresponds to an estimated mean and variance). They here use a loss that only penalizes the best hypothesis (the one which is closest to the ground truth), which encourages the model to yield a diverse set of hypotheses in case of doubt. A second network is then trained to optimally merge the hypotheses into a final mean and variance estimate. It is however not clear to me how this merging network actually is trained (did I miss something in the paper?).

- They train their models on the FlyingChairs and FlyingThings3D datasets, and mostly evaluate on the Sintel dataset.

- For all ensembles, they use M=8 networks (and M=8 hypotheses in the multi-headed model) (they find that more networks generally results in better performance, but are practically limited in terms of computation and memory).
 
- To assess the quality of the obtained uncertainty estimates, they use sparsification plots as the main evaluation metric. In such a plot, you plot the average error as a function of the fraction of removed pixels, where the pixels are removed in order, starting with the pixels corresponding to the largest estimated uncertainty. This average error should thus monotonically decrease (as we remove more and more pixels) if the estimated uncertainty actually is a good representation of the true uncertainty/error. The obtained curve is compared to the "Oracle" sparsification curve, obtained by removing pixels according to the true error.

- In their results, they find e.g. that:
- - Predictive ensembles have better performance than empirical ensembles. Even a single predictive model they claim to yield better uncertainty estimates than any empirical ensemble.
- - Predictive ensembles only yield slightly better performance than a single predictive model.
- - MC-dropout consistently performs worse than both bootstrap and snapshot ensembles (note that they also use just M=8 forward passes in MC-dropout).
- - The multi-headed predictive model yields the best performance among all models.

### Comments:
- Very interesting paper with a thorough comparison of various uncertainty estimation techniques. 

- I am however not completely convinced by the evaluation. I get how the sparsification plots measure the quality of the *relative* uncertainties (i.e., whether or not the model has learned what pixels are the most/least uncertain), but what about the absolute magnitude? Could it be that a model consistently under/over-estimates the uncertainties? If we were to create prediction intervals based on the estimated uncertainties, would they then have valid coverage?

- The multi-headed network is an interesting idea, I did not expect it to yield the best performance. 
