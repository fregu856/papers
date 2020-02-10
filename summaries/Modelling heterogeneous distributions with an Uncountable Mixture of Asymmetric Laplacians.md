##### [20-01-31] [paper84]
- Modelling heterogeneous distributions with an Uncountable Mixture of Asymmetric Laplacians [[pdf]](https://arxiv.org/abs/1910.12288) [[code]](https://github.com/BBVA/UMAL) [[video]](https://vimeo.com/369179175) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Modelling%20heterogeneous%20distributions%20with%20an%20Uncountable%20Mixture%20of%20Asymmetric%20Laplacians.pdf)
- *Axel Brando, Jose A. Rodríguez-Serrano, Jordi Vitrià, Alberto Rubio*
- `2019-10-27, NeurIPS 2019`

****

### General comments on paper quality:
- Quite well-written and interesting paper.

### Comments:
- The connection to quantile regression is quite neat, but in the end, their loss in equation 6 just corresponds to a latent variable model (with a uniform distribution for the latent variable tau) trained using straightforward Monte Carlo sampling.

- I am definitely not impressed with the experiments. They only consider very simple problems, y is always 1D, and they only compare with self-implemented baselines. The results are IMO not overly conclusive either, the single Laplacian model is e.g. better calibrated than their proposed method in Figure 3.
