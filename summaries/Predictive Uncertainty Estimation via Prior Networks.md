##### [19-02-27] [paper47]
- Predictive Uncertainty Estimation via Prior Networks [[pdf]](https://arxiv.org/abs/1802.10501) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Predictive%20Uncertainty%20Estimation%20via%20Prior%20Networks.pdf)
- *Andrey Malinin, Mark Gales*
- `2018-02-28, NeurIPS2018`

****

### General comments on paper quality:
- Interesting and very well-written paper. 

### Comments:
- It would be interesting to combine this approach with approximate Bayesian modeling (e.g. ensembling).

- They state in the very last sentence of the paper that their approach needs to be extended also to regression. How would you actually do that? It is not immediately obvious to me. Seems like a quite interesting problem.

- I would have liked to see a comparison with ensembling as well and not just MC-Dropout (ensembling usually performs better in my experience).

- Obtaining out-of-distribution samples to train on is probably not at all trivial actually. Yes, this could in theory be any unlabeled data, but how do you know what region of the input image space is covered by your training data?

- Also, I guess the model could still become over-confident if fed inputs which are far from both the in-distribution and out-of-distribution samples the model has seen during training? So, you really ought to estimate epistemic uncertainty using Bayesian modeling as well?
