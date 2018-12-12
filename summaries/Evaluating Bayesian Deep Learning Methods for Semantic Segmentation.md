##### [18-12-06] [paper25]
- Evaluating Bayesian Deep Learning Methods for Semantic Segmentation [[pdf]](https://arxiv.org/abs/1811.12709) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Evaluating%20Bayesian%20Deep%20Learning%20Methods%20for%20Semantic%20Segmentation.pdf)
- *Jishnu Mukhoti, Yarin Gal*
- `2018-11-30`

****

### General comments on paper quality:
- Quite well-written and interesting paper. Not particularly heavy to read.

### Paper overview:
- The authors present three metrics designed to evaluate and compare different Bayesian DL methods for the task of semantic segmentation (i.e., models which also output pixel-wise uncertainty estimates).

- They train DeepLabv3+ using both MC-dropout (apply dropout also during inference, run multiple forward passes to obtain M samples, compute the sample mean and variance) and Concrete dropout ("a modification of the MC-dropout method where the network tunes the dropout rates as part of the optimization process"), and then compare these two methods on Cityscapes using their suggested metrics. They thus hope to provide quantitative benchmarks which can be used for future comparisons. 

- Their three presented metrics are (higher values are better):
- - p(accurate | certain) = n_a_c/(n_a_c + n_i_c).
- - p(uncertain | inaccurate) = n_i_u/(n_i_c + n_i_u).
- - PAvPU = (n_a_c + n_i_u)/(n_a_c + n_a_u + n_i_c + n_i_u)
- - Where:
- - - n_a_c: number of accurate and certain patches.
- - - n_a_u: number of accurate and uncertain patches.
- - - n_i_c: number of inaccurate and certain patches.
- - - n_i_u: number of inaccurate and uncertain patches.

- They compute these metrics on patches of size 4x4 pixels (I didn't quite get their reasoning for why this makes more sense than studying this pixel-wise), where a patch is defined as accurate if more than 50% of the pixels in the patch are correctly classified. Similarly, a patch is defined as uncertain if its average pixel-wise uncertainty is above a given threshold. They set this uncertainty threshold to the average uncertainty value on Cityscapes val (which I found somewhat strange, since they then also do all of their evaluations on Cityscapes val).

- They found that MC-dropout outperformed concrete dropout with respect to all three metrics.
 
### Comments:
- The intended contribution is great, we definitely need to define metrics which can be used to benchmark different uncertainty estimating models. I am not 100% happy with the presentation of their suggested metrics though: 
- - What would be the ideal values for these metrics?
- - Can the metrics be ranked in terms of importance? 
- - What is the "best" value for the uncertainty threshold, and how should it be chosen?
