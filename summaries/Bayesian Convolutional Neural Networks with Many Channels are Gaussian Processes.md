##### [18-10-25] [paper15]
- Bayesian Convolutional Neural Networks with Many Channels are Gaussian Processes [[pdf]](https://arxiv.org/abs/1810.05148) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Bayesian%20Convolutional%20Neural%20Networks%20with%20Many%20Channels%20are%20Gaussian%20Processes_.pdf)
- *Roman Novak, Lechao Xiao, Jaehoon Lee, Yasaman Bahri, Daniel A. Abolafia, Jeffrey Pennington, Jascha Sohl-Dickstein*
- `2018-10-11`

****

### General comments on paper quality:
- Fairly well-written but rather heavy paper to read, I probably don't have the necessary background to fully appreciate its contributions.

### Paper overview:
- There is a known correspondence between fully Bayesian (with Gaussian prior), *infinitely wide*, fully connected, deep feedforward neural networks and Gaussian processes. The authors here derive an analogous correspondence between fully Bayesian (with Gaussian prior), deep CNNs with *infinitely many channels* and Gaussian Processes. 

- They also propose a method to find this corresponding GP (which has a 0 mean function), by estimating its kernel (which might be computationally impractical to compute analytically, or might have an unknown analytic form) using a Monte Carlo method. They show that this estimated kernel converges to the analytic kernel in probability as the number of channels.

### Comments:
- I always find these kind of papers interesting as they try to improve our understanding of the theoretical properties of neural networks. However, it's still not particularly clear to me what this GP correspondence in the infinite limit of fully Bayesian networks actually tells us about finite, non-Bayesian networks. 
