##### [19-04-05] [paper53]
- Stochastic Gradient Descent as Approximate Bayesian Inference [[pdf]](https://arxiv.org/abs/1704.04289) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Stochastic%20Gradient%20Descent%20as%20Approximate%20Bayesian%20Inference.pdf)
- *Stephan Mandt, Matthew D. Hoffman, David M. Blei*
- `2017-04-13, Journal of Machine Learning Research 18 (2017)`

****

### General comments on paper quality:
- Very well-written and quite interesting paper. Good background material on SGD, SG-MCMC and so on. It is however a relatively long paper (26 pages).

### Comments:
- It makes intuitive sense that running SGD with a constant learning rate will result in a sequence of iterates which first move toward a local minimum and then "bounces around" its vicinity. And, that this "bouncing around" thus should correspond to samples from some kind of stationary distribution, which depends on the learning rate, batch size and other hyper parameters.

- Trying to find the hyper parameters which minimize the KL divergence between this stationary distribution and the true posterior then seems like a neat idea. I am however not quite sure how reasonable the made assumptions are in more complex real-world problems. I am thus not quite sure how useful the specific proposed methods/formulas actually are.
