##### [18-10-04] [paper9]
- On gradient regularizers for MMD GANs [[pdf]](https://arxiv.org/abs/1805.11565) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20gradient%20regularizers%20for%20MMD%20GANs_.pdf)
- *Michael Arbel, Dougal J. Sutherland, Mikołaj Bińkowski, Arthur Gretton*
- `2018-05-29`

****

### General comments on paper quality:
- Well-written but rather heavy paper to read, I did definitely not have the background required neither to fully understand nor to properly appreciate the proposed methods. I would probably need to do some more background reading and then come back and read this paper again.

### Paper overview:
- The authors propose the method *Gradient-Constrained MMD* and its approximation *Scaled MMD*, MMD GAN architectures which are trained using a novel loss function that regularizes the gradients of the critic (gradient-based regularization).

- The authors experimentally evaluate their proposed architectures on the task of unsupervised image generation, on three different datasets (CIFAR-10 (32x32 images), CelebA (160x160 images) and ImageNet (64x64 images)) and using three different metrics (Inception score (IS), FID and KID). They find that their proposed losses lead to stable training and that they outperform (or at least obtain highly comparable performance to) state-of-the-art methods (e.g. Wasserstein GAN).
