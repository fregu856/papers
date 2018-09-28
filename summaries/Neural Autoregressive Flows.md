##### [18-09-27] [paper6]
- Neural Autoregressive Flows [[pdf]](https://arxiv.org/abs/1804.00779) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Autoregressive%20Flows_.pdf)
- *Chin-Wei Huang, David Krueger, Alexandre Lacoste, Aaron Courville*
- `2018-04-03, ICML2018`

****

### General comments on paper quality:
- Well-written and interesting paper. As I was quite unfamiliar with the material, it did however require an extra read-through.

### Paper overview:
- The authors introduce Neural Autoregressive Flow (NAF), a flexible method for tractably approximating rich families of distributions. Empirically, they show that NAF is able to model multi-modal distributions and outperform related methods (e.g. inverse autoregressive flow (IAF)) when applied to density estimation and variational inference.

- A Normalizing Flow (NF), is an invertible function f: X --> Y expressing the transformation between two random variables (i.e., is used to translate between the distributions p_{Y}(y) and p_{X}(x)). NFs are most commonly trained to, from an input distribution p_{X}(x), produce an output distributions p_{Y}(y) that matches a target distribution p_{target}(y) (as measured by the KL divergence). E.g. in variational inference, the NF is typically used to transform a simple input distribution (e.g. standard normal) over x into a complex approximate posterior p_{Y}(y).

- Research on constructing NFs, such as this work, focuses on finding ways to parameterize the NF which meet certain requirements while being maximally flexible in terms of the transformations which they can represent.

- One specific (particularly successful) class of NFs are affine autoregressive flows (AAFs) (e.g. IAFs). In AFFs, the components of x and y (x_{i}, y_{i}) are given an (arbitrarily chosen) order, and y_{t} is computed as a function of x_{1:t}:
- - y_{t} = f(x_{1:t}) = tau(c(x_{1:t-1}), x_{t}), where:
- - - c is an autoregressive *conditioner*.
- - - tau is an invertible *transformer*.

- In previous work, tau is taken to be a simple affine function, e.g. in IAFs:
- - tau(mu, sigma, x_{t}) = sigma*x_{t} + (1-sigma)*mu, where mu and sigma are outputted by the conditioner c.

- In this paper, the authors replace the affine transformer tau with a neural network (yielding a more rich family of distributions with only a minor increase in computation and memory requirements), which results in the NAF method:
- - tau(c(x_{1:t-1}), x_{t}) = NN(x_{t}; phi = c(x_{1:t-1})), where:
- - - NN is a (small, 1-2 hidden layers with 8/16 units) neural network that takes the scalar x_{t} as input and produces y_{t} as output, and its weights and biases phi are outputted by c(x_{1:t-1}).
- - To ensure that tau is strictly monotonic an thus invertible, it is sufficient to use strictly positive weights and strictly monotonic activation functions in the neural network.

- The authors prove that NAFs are universal density approximators, i.e., can be used to approximate any probability distribution (over real vectors) arbitrarily well. I.e., NAFs can be used to transform any random variable into any desired random variable.

- The authors empirically evaluate NAFs applied to variational inference and density estimation, and outperform IAF and MAF baselines. For example, they find that NAFs can approximate a multi-modal mixture of Gaussian distribution quite well, while AAFs only produces a uni-modal distribution.

### Comments:
- I probably need to do some more reading on the background material to fully understand and appreciate the results of this paper, but it definitely seems quite interesting. Could probably be useful in some application.
