##### [20-01-14] [paper78]
- Noise-contrastive estimation: A new estimation principle for unnormalized statistical models [[pdf]](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Noise-contrastive%20estimation:%20A%20new%20estimation%20principle%20for%20unnormalized%20statistical%20models.pdf)
- *Michael Gutmann, Aapo Hyvarinen*
- `2009, AISTATS 2010`

****

### General comments on paper quality:
- Well-written and interesting paper.

### Comments:
- The original paper for Noise Contrastive Estimation (NCE). Somewhat dated of course,  but still interesting and well-written. Provides a quite neat introduction to NCE.

- They use a VERY simple problem to compare the performance of NCE to MLE with importance sampling, contrastive divergence (CD) and score-matching (and MLE, which gives the reference performance. MLE requires an analytical expression for the normalizing constant). CD has the best performance, but NCE is apparently more computationally efficient. I do not think such a simple problem say too much though.

- They then also apply NCE on a (by today's standards) very simple unsupervised image modeling problem. It seems to perform as expected. 
