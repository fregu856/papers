##### [18-09-24] [paper2]
- What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? [[pdf]](https://arxiv.org/abs/1703.04977) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/What%20Uncertainties%20Do%20We%20Need%20in%20Bayesian%20Deep%20Learning%20for%20Computer%20Vision%3F_.pdf)
- *Alex Kendall, Yarin Gal*
- `2017-10-05, NIPS2017`

****

### General comments on paper quality:
- Well written and interesting paper. Seems like a fairly complete introduction to Bayesian deep learning. Clearly defines aleatoric and epistemic uncertainty, and provides good intuition for what they capture and how they differ. A recommended read.

### Paper overview:
- The authors describe the two major types of uncertainty that one can model:
- - **Aleatoric uncertainty**: captures noise inherent in the observations, uncertainty which NOT can be reduced by collecting more data. It can also vary for different inputs to the model, if we for example are trying to estimate the distance to vehicles seen in camera images, we'd expect the distance estimate to be more noisy/uncertain for vehicles far away from the camera. 
- - **Epistemic uncertainty** (a.k.a model uncertainty): accounts for uncertainty in the model parameters, uncertainty which CAN be explained away given enough data.

- To model **epistemic uncertainty** in a neural network (NN), one puts a prior distribution over its weights W and then tries to compute the posterior p(W | X, Y). This is what typically is called a Bayesian NN (BNN). BNNs are easy to formulate, but difficult to perform inference in. Different approximate techniques exist, and the authors use Monte Carlo dropout (Use dropout during both training and testing. During testing, run multiple forward-passes and (essentially) compute the sample mean and variance).

- To model **aleatoric uncertainty**, one assumes a (conditional) distribution over the output of the network (e.g. Gaussian with mean u(x) and sigma s(x)) and learns the parameters of this distribution (in the Gaussian case, the network outputs both u(x) and s(x)) by maximizing the corresponding likelihood function. Note that in e.g. the Gaussian case, one does NOT need extra labels to learn s(x), it is learned implicitly from the induced loss function. The authors call such a model a heteroscedastic NN.

- (At least for the Gaussian case) they note that letting the model output both u(x) and s(x) allows it to intelligently attenuate the residuals in the loss function (since the residuals are divided by s(x)^2), making the model more robust to noisy data. 

- The novel contribution by the authors is a framework for modeling both epistemic and aleatoric uncertainty in the same model. To do this, they use Monte Carlo dropout to turn their heteroscedastic NN into a Bayesian NN (essentially: use dropout and run multiple forward passes in a model which outputs both u(x) and s(x)). They demonstrate their framework for both regression and classification tasks, as they present results for per-pixel depth regression and semantic segmentation. 

- For each task, they compare four different models:
- - Without uncertainty.
- - Only aleatoric uncertainty.
- - Only epistemic uncertainty.
- - Aleatoric and epistemic uncertainty.

- They find that modeling both aleatoric and epistemic uncertainty results in the best performance (roughly 1-3% improvement over no uncertainty) but that the main contribution comes from modeling the aleatoric uncertainty, suggesting that epistemic uncertainty mostly can be explained away when using large datasets.

- Qualitatively, for depth regression, they find that the aleatoric uncertainty is larger for e.g. great depths and reflective surfaces, which makes intuitive sense.

- The authors also perform experiments where they train the models on increasingly larger datasets (1/4 of the dataset, 1/2 of the dataset and the full dataset) and compare their performance on different test datasets. 
- - They here find that the aleatoric uncertainty remains roughly constant for the different cases, suggesting that aleatoric uncertainty NOT can be explained away with more data (as expected), but also that aleatoric uncertainty does NOT increase for out-of-data examples (examples which differs a lot from the training data).
- - On the other hand, they find that the epistemic uncertainty clearly decreases as the training datasets gets larger (i.e., it seems as the epistemic uncertainty CAN be explained away with enough data, as expected), and that it is significantly larger when the training and test datasets are NOT very similar (i.e., the epistemic uncertainty is larger when we train on dataset A-train and test on dataset B-test, than when we train on dataset A-train and test on dataset A-test).

- This reinforces the case that while epistemic uncertainty can be explained away with enough data, it is still required to capture situations not encountered in the training data, which is particularly important for safety-critical systems (where epistemic uncertainty is required to detect situations which have never been seen by the model before).

- Finally, the authors note that the aleatoric uncertainty models add negligible compute compared to deterministic models, but that the epistemic models require expensive Monte Carlo dropout sampling (50 Monte Carlo samples often results in a 50x slow-down). They thus mark finding a method for real-time epistemic uncertainty estimation as an important direction for future research.

### Comments:
- From the authors' experiments, it seems reasonably safe to assume that incorporating some kind of uncertainty measure can help improve model performance. It could thus definitely be of practical use (especially aleatoric uncertainty estimation, since it's quite simple to implement and computationally inexpensive). 

- However, it's still not quite clear to me how much you can actually trust these estimated uncertainties. The NN is still essentially a black box, so how do we know if the outputted aleatoric uncertainty estimate is "correct" in any given case? Is it possible to somehow create e.g. 95% confidence intervals from these estimated uncertainties?
