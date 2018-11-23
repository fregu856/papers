##### [18-11-22] [paper22]
-  A Probabilistic U-Net for Segmentation of Ambiguous Images [[pdf]](https://arxiv.org/abs/1806.05034) [[code]](https://github.com/SimonKohl/probabilistic_unet) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Probabilistic%20U-Net%20for%20Segmentation%20of%20Ambiguous%20Images.pdf)
- *Simon A. A. Kohl, Bernardino Romera-Paredes, Clemens Meyer, Jeffrey De Fauw, Joseph R. Ledsam, Klaus H. Maier-Hein, S. M. Ali Eslami, Danilo Jimenez Rezende, Olaf Ronneberger*
- `2018-10-29, NeurIPS2018`

****

### General comments on paper quality:
- Well-written and interesting paper.

### Paper overview:
- The authors present a model for semantic image segmentation designed to produce a set of diverse but plausible/consistent segmentations. The model is intended for domains in which it is difficult to assign a unique ground truth segmentation mask to any given input image (there is some ambiguity), as in e.g. some medical imaging applications where it is standard practice to have a group of graders label each image (and the graders might produce quite different looking segmentations).

- The model is a combination of a standard U-Net and a Conditional Variational Auto-Encoder (CVAE). The CVAE encodes the input image into a low-dimensional latent space (they use N=6), and a random sample from this space is injected into the final block of the U-Net to produce a corresponding segmentation mask. 

- A prior-net (essentially the U-Net encoder) takes the image as input and outputs mu_prior, sigma_prior (both in R^N) for a Gaussian (SIGMA = diag(sigma)) in the latent space. To sample z from the latent space, they simply sample from this Gaussian. The sample z in R^N is broadcasted to an N-channel feature map and concatenated with the last feature map of the U-Net. This new feature map is then processed by three 1x1 convolutions to map it to the number of classes (the feature map has the same spatial size as the input image). To output M segmentation masks, one thus only has to sample z_1, ..., z_M and follow the above procedure (prior-net and the U-Net only have to be evaluated once).

- During training, each image - label pair (X, Y) is taken as input to a posterior-net (essentially the U-Net encoder) which outputs mu_post, sigma_post for a Gaussian in the latent space. A sample z is drawn from this Gaussian, the corresponding segmentation map is produced (same procedure as above) and then compared with the label Y using the standard cross-entropy loss. To this loss we also add a KL loss term which penalizes differences between the posterior-net and prior-net Gaussians (the prior net only takes the image X as input).

- They evaluate their method on two different datasets:
- - LIDC-IDRI, a medical dataset for lung abnormalities segmentation in which each lung CT scan has been independently labeled by four experts (each image has four corresponding ground truth labels, i.e., there is inherent ambiguity).
- - A modified version of Cityscapes. They here manually inject ambiguity into the dataset by e.g. changing the class "sidewalk" to "sidewalk2" (a class created by the authors) with some probability. They do this for 5 original classes, and thus end up with 2^5=32 possible modes with probabilities ranging from 10.9% to 0.5% (a given input image could thus correspond to any of these 32 modes, they have manually created some ambiguity).

- Since they are not interested in comparing a deterministic prediction with a unique ground truth, but rather in comparing distributions of segmentations, they use a non-standard performance metric across their experiments. 

- They compare their method to number of baselines (a U-Net with MC-dropout, an ensemble of U-Nets, a U-Net with multiple heads) (same number of forward passes / ensemble networks / heads as the number of samples from the latent space), and basically find that their method outperforms all of them with respect to their performance metric.

### Comments:
- Interesting paper. 

- The method is mainly intended for the medical imaging domain, where I definitely can see why you might want a model that outputs a set of plausible segmentations which then can be further analyzed by medical professionals. For autonomous driving however, I guess what you ultimately want is just the most likely prediction and, crucially, the corresponding uncertainty of this prediction. Can we extract this from the proposed method?

- If we take the mean of the prior-net Gaussian as our sample, I guess we would produce the most likely segmentation? And I guess sigma of this Gaussian is then a measure of the corresponding uncertainty? How about uncertainty estimates for the pixel-wise predictions, could you extract those as well somehow? Just treat the M maps of predicted class scores like you would when using MC-dropout or ensembles (e.g. take the sample variance as a measure of the epistemic uncertainty), or could you get this directly from the Gaussian?
