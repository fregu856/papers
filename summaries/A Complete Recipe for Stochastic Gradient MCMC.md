##### [19-01-25] [paper33]
-  A Complete Recipe for Stochastic Gradient MCMC [[pdf]](https://arxiv.org/abs/1506.04696) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Complete%20Recipe%20for%20Stochastic%20Gradient%20MCMC.pdf)
- *Yi-An Ma, Tianqi Chen, Emily B. Fox*
- `2015-06-15, NeurIPS2015`

****

### General comments on paper quality:
- Well-written and very interesting paper. After reading the papers on [SGLD](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf) and [SGHMC](https://arxiv.org/abs/1402.4102), this paper ties the theory together and provides a general framework for SG-MCMC.

### Paper overview:
- The authors present a general framework and recipe for constructing MCMC and SG-MCMC samplers based on continuous Markov processes. The framework entails specifying a stochastic differential equation (SDE) by two matrices, D(z) (positive semi-definite) and Q(z) (skew-symmetric). Here, z = (theta, r), where theta are the model parameters and r are auxiliary variables (r corresponds to the momentum variables in Hamiltonian MC).

- Importantly, the presented framework is *complete*, meaning that all continuous Markov processes with the target distribution as its stationary distribution (i.e., all continuous Markov processes which provide samples from the target distribution) correspond to a specific choice of the matrices D(z), Q(z). Every choice of D(z), Q(z) also specifies a continuous Markov process with the target distribution as its stationary distribution.

- The authors show how previous SG-MCMC methods (including SGLD, SGRLD and SGHMC) can be casted to their framework, i.e., what their corresponding D(z), Q(z) are.

- They also introduce a new SG-MCMC method, named SGRHMC, by wisely choosing D(z), Q(z).

- Finally, they conduct two simple experiments which seem to suggest (at least somewhat) improved performance of SGRHMC compared to previous methods (SGLD, SGRLD, SGHMC).

### Comments:
- How does one construct \hat{B_t}, the estimate of V(theta_t) (the noise of the stochastic gradient)?

- If one (by computational reasons) only can afford evaluating, say, 10 samples to estimate various expectations, what 10 samples should one pick? The final 10 samples, or will those be heavily correlated? Pick the final sample (at time t = T) and then also the samples at time t=T-k*100 (k = 1, 2, ..., 9)? (when should one start collecting samples and with what frequency should they then be collected?)

- If one were to train an ensemble of models using SG-MCMC and pick the final sample of each model, how would these samples be distributed?

- If the posterior distribution is a simple bowl, like in the right part of figure 2,  what will the path of samples actually look like compared to the steps taken by SGD? In figure 2, I guess that gSHRHMC will eventually converge to roughly the bottom of the bowl? So if one were to only collect samples from this later stage of traversing, the samples would actually NOT be (at least approximately) distributed according to the posterior?
