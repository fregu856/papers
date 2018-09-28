##### [18-09-27] [paper7]
- Conditional Neural Processes [[pdf]](https://arxiv.org/abs/1807.01613) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Conditional%20Neural%20Processes_.pdf)
- *Marta Garnelo, Dan Rosenbaum, Chris J. Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo J. Rezende, S. M. Ali Eslami*
- `2018-07-04, ICML2018`

****

### General comments on paper quality:
- Quite well-written. Interesting proposed method.

### Paper overview:
- The authors present a family of neural models called Conditional Neural Processes (CNPs), which aim to combine the benefits of Gaussian Processes (exploit prior knowledge to quickly infer the shape of a new function at test time, but computationally scale poorly with increased dimension and dataset size) and deep neural networks (excel at function approximation, but need to be trained from scratch for each new function).

- A CNP feeds each observation (x_i, y_i) (a labeled example) through a neural network h to extract an embedding r_i. The embeddings r_i are aggregated to a single embedding r using a symmetric aggregator function (e.g. taking the mean). The embedding r is then fed together with each target x*_i (unlabeled example) as input to the neural network g, which produces a corresponding prediction y*_i. The predictions are thus made *conditioned* on the observations (x_i, y_i).

- For example, given observations of an unknown function's value y_i at locations x_i, we would like to predict the function's value at new locations x*_i, conditioned on the observations.

- To train a CNP, we have access to a training set of n observations (x_i, y_i). We then produce predictions y^_i for each x_i, conditioned on a randomly chosen subset of the observations, and minimize the negative (conditional) log likelihood. I.e., create r by embedding N randomly chosen observations (x_i, y_i) with the neural network h, then feed r as input to g and compute predictions y^_i for each x_i, and compare these with the true values y_i.

- The authors experimentally evaluate the CNP approach on three different problems:
- - 1D regression:
- - - At every training step, they sample a curve (function) from a fixed Gaussian Process (GP), select a subset of n points (x_i, y_i) from it as observations, and a subset of m points (x'_j, y'_j) as targets. 
- - - The embedding r is created from the observations, used in g to output a prediction (y^_j, sigma^_j) (mean and variance) for each target x'_j, and compared with the true values y`_j.
- - - In inference, they again sample a curve from the GP and are given a subset of points from it as observations. From this they create the embedding r, and are then able to output predictions (mean and variance) at arbitrary points x. Hopefully, these predictions (both in terms of mean and variance) will be close to what is outputted by a GP (with the true hyperparameters) fitted to the same observations. This is also, more or less, what they observe in their experiments.
- - Image completion:
- - - Given a few pixels as observations, predict the pixel values at all pixel locations. CNP is found to outperform both a kNN and GP baseline, at least when the number of given observations is relatively small.
- - One-shot classification:
- - - While CNP does NOT set a new SOTA, it is found to have comparable performance as significantly more complex models.

- The authors conclude by arguing that a trained CNP is more general than conventional deep learning models, in that it encapsulates the high-level statistics of a family of functions. As such it constitutes a high-level abstraction that can be reused for multiple tasks.

- Left as future work is the task of scaling up the proof-of-concept CNP architectures used in the paper, and exploring how CNPs can help tackling problems such as transfer learning and data efficiency.

### Comments:
- Pretty interesting approach, although it's not immediately obvious to me what use it could have in the context of 3DOD and/or uncertainty estimation (outputting both a mean and a variance is of course interesting, but you don't need to explicitly condition the model on some observations). I guess transfer learning to fine-tune performance on a specific subset of your data is the most obvious possible application.
