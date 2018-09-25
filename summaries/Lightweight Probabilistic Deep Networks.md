##### [18-09-24]
- Lightweight Probabilistic Deep Networks [[pdf]](https://arxiv.org/abs/1805.11327) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Lightweight%20Probabilistic%20Deep%20Networks_.pdf)
- *Jochen Gast, Stefan Roth*
- `2018-05-29, CVPR2018`

****

### General comments on paper quality:
- Quite interesting and well written paper, I did however find a couple of the derivations (Deep uncertainty propagation using ADF & Classification with Dirichlet outputs) somewhat difficult to follow.

### Paper overview:
- The authors introduce two lightweight approaches to supervised learning (both regression and classification) with probabilistic deep networks:
- - ProbOut: replace the output layer of a network with a distribution over the output (i.e., output e.g. a Gaussian mean and variance instead of just a point estimate).
- - ADF: go one step further and replace all intermediate activations with distributions. Assumed density filtering (ADF) is used to propagate activation uncertainties through the network.

- I.e., their approach is not a Bayesian network in the classical sense (there's no distribution over the network weights). In the terminology of [Kendall and Gal](https://github.com/fregu856/papers/blob/master/summaries/What%20Uncertainties%20Do%20We%20Need%20in%20Bayesian%20Deep%20Learning%20for%20Computer%20Vision%3F.md), the approach only captures aleatoric (heteroscedastic) uncertainty. In fact, ProbOut is virtually identical to the approach used by [Kendall and Gal](https://github.com/fregu856/papers/blob/master/summaries/What%20Uncertainties%20Do%20We%20Need%20in%20Bayesian%20Deep%20Learning%20for%20Computer%20Vision%3F.md) to model aleatoric uncertainty (they do however use slightly different approaches for classification tasks). The authors choose to disregard epistemic (model) uncertainty in favor of improved computational performance, arguing that epistemic uncertainty is less important since it can be explained away with enough data.
 
- While ProbOut is simple to both formulate and implement, ADF is more involved. ADF is also nearly 3x as slow in inference, while ProbOut adds negligible compute compared to standard deterministic networks.

- The authors evaluate ProbOut and ADF on the task of optical flow estimation (regression) and image classification. They find that their probabilistic approaches somewhat outperform the deterministic baseline across tasks and datasets. There's however no significant difference between ProbOut and ADF.

- They empirically find the estimated uncertainties from their models to be highly correlated with the actual error. They don't really mention if ProbOut or ADF is significantly better than the other in this regard. 

### Comments:
- From the results presented in the paper, I actually find it quite difficult to see why anyone would prefer ADF over ProbOut. ProbOut seems more simple to understand and implement, is quite significantly faster in inference, and seems to have comparable task performance and capability to model aleatoric uncertainty.  

- Thus, I'm not quite sure how significant the contribution of this paper actually is. Essentially, they have taken the method for modeling aleatoric uncertainty from [Kendall and Gal](https://github.com/fregu856/papers/blob/master/summaries/What%20Uncertainties%20Do%20We%20Need%20in%20Bayesian%20Deep%20Learning%20for%20Computer%20Vision%3F.md) and applied this to slightly different tasks.

Also, my question from the Kendall and Gal [summary](https://github.com/fregu856/papers/blob/master/summaries/What%20Uncertainties%20Do%20We%20Need%20in%20Bayesian%20Deep%20Learning%20for%20Computer%20Vision%3F.md) still remains. Even if we assume negligible epistemic (model) uncertainty, how much can we actually trust the outputted aleatoric uncertainty?
