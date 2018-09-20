##### [18-09-20]
- Gaussian Process Behaviour in Wide Deep Neural Networks [[pdf]](https://arxiv.org/abs/1804.11271) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Gaussian%20Process%20Behaviour%20in%20Wide%20Deep%20Neural%20Networks.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Gaussian%20Process%20Behaviour%20in%20Wide%20Deep%20Neural%20Networks.md)
- *Alexander G. de G. Matthews, Mark Rowland, Jiri Hron, Richard E. Turner, Zoubin Ghahramani*
- `2018-08-16, ICLR2018`
- [[code]](https://github.com/widedeepnetworks/widedeepnetworks)

****

#### General comments on paper quality:
- Very well written and mathematically rigorous paper that I'd recommend anyone interested in theoretical properties of deep learning to read. An interesting and pleasent read.

#### Paper overview:
- The authors study the relationship between random, wide, fully connected, feedforward neural networks and Gaussian processes. 

- - - The network weights are assumed to be independent normally distributed with their variances sensibly scaled as the network grows.

- They show that, under broad conditions, as the network is made increasingly wide, the implied random input-to-output function converges in distribution to a Gaussian process.

- They also compare exact Gaussian process inference with MCMC inference for finite Bayesian neural networks. Of the six datasets considered, five show close agreement between the two models.

- Because of the computational burden of the MCMC algorithms, the problems they study are however quite small in terms of both network size, data dimensionality and datset size. 

- Furthermore, the one dataset on which the Bayesian deep network and the Gaussian process did not make very similar predictions was the one with the highest dimensionality. The authors thus sound a note of caution about extrapolating their empirical findings too confidently to the domain of high-dimensional, large-scale datasets.

- Still, the authors conclude that it seems likely that some experiments in the Bayesian deep learning literature would have given very similar results to a Gaussian process. They thus also recommend the Bayesian deep learning community to routinely compare their results to Gaussian processes (with the kernels specified in the paper).

- Finally, the authors hope that their results will help to further the theoretical understanding of deep neural networks in future follow-up work.
