##### [18-09-30] [paper8]
- Neural Processes [[pdf]](https://arxiv.org/abs/1807.01622) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Processes_.pdf)
- *Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J. Rezende, S.M. Ali Eslami, Yee Whye Teh*
- `2018-07-04, ICML2018 Workshop`

****

### General comments on paper quality:
- Quite well-written overall, although I did find a couple of typos (and perhaps you should expect more for a paper with 7 authors?). Also, I don't think the proposed method is quite as clearly explained as in the [Conditional Neural Processes](https://github.com/fregu856/papers/blob/master/summaries/Conditional%20Neural%20Processes.md) paper.

### Paper overview:
- The authors introduce Neural Processes (NPs), a family of models that aim to combine the benefits of Gaussian Processes (GPs) and neural networks (NNs). NPs are an extension of Conditional Neural Processes (CNPs) by the same main authors ([[summary]](https://github.com/fregu856/papers/blob/master/summaries/Conditional%20Neural%20Processes.md)).

- A blog post that provides a good overview of and some intuition for NPs is found [here](https://kasparmartens.rbind.io/post/np/).

- NPs are very similar to [CNPs](https://github.com/fregu856/papers/blob/master/summaries/Conditional%20Neural%20Processes.md), with the main difference being the introduction of a global latent variable z:
- - Again, each observation (x_i, y_i) (a labeled example) is fed through the encoder NN h to extract an embedding r_i. The embeddings r_i are aggregated (averaged) to a single embedding r. This r is then used to parameterize the distribution of the global latent variable z: z ~ N(mu(r), sigma(r)).
- - z is then sampled and fed together with each target x*_i (unlabeled example) as input to the decoder NN g, which produces a corresponding prediction y*_i.

- The introduced global latent variable z is supposed to capture the global uncertainty, which allows one to sample at a global level (one function at a time) rather than just at a local output level (one y_i value for each x_i at the time). I.e., NPs are able (whereas CNPs are unable) to produce different function samples for the same conditioned observations. In e.g. the application of image completion on MNIST, this will make the model produce different digits when sampled multiple times and the model only has seen a small number of observations (as compared to CNPs which in the same setting would just output (roughly) the mean of all MNIST images).

- The authors experimentally evaluate the NP approach on several different problems (1D function regression, image completion, Bayesian optimization of 1D functions using Thompson sampling, contextual bandits), they do however not really compare the NP results with those of CNPs.

### Comments:
- It's not immediately obvious to me what use NPs could have in the context of e.g. 3DOD, and not what practical advantage using NPs would have over using CNPs either. 

- I suppose the approach might be interesting if you were to train a 3DOD model on multiple datasets with the goal to maximize performance on a specific dataset (e.g., you have collected / have access to a bunch of data corresponding to different LiDAR sensor setups, and now you want to train a model for your final production car sensor setup). In this case I guess this approach might be preferable over just simply training a conventional DNN on all of the data?
