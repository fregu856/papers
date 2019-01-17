##### [19-01-09] [paper27]
- Relaxed Softmax: Efficient Confidence Auto-Calibration for Safe Pedestrian Detection [[pdf]](https://openreview.net/forum?id=S1lG7aTnqQ) [[poster]](http://www.robots.ox.ac.uk/~vgg/publications/2018/Neumann18c/poster.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Relaxed%20Softmax:%20Efficient%20Confidence%20Auto-Calibration%20for%20Safe%20Pedestrian%20Detection.pdf)
- *Lukas Neumann, Andrew Zisserman, Andrea Vedaldi*
- `2018-11-29, NeurIPS2018 Workshop`

****

### General comments on paper quality:
- Reasonably well-written paper. I'm not entirely convinced of the usefulness of the proposed method.

### Paper overview:
- The authors study pedestrian object detectors and evaluate the quality of their confidence score estimates using reliability diagrams (and related metrics, e.g. Expected Calibration Error). They find that a SOTA detector produces significantly over-confident predictions, i.e., that the obtained accuracy for predictions in any given confidence score interval is lower than the associated confidence score.

- To mitigate this problem, they propose a simple modification of the standard softmax layer, called *relaxed softmax*. Instead of having the network output logits z in R^{K} and computing the probability vector softmax(z) (also in R^{K}), the network instead outputs (z, alpha), where alpha > 0, and the probability vector is computed as softmax(alpha*z). Relaxed softmax is inspired by [temperature scaling](https://github.com/fregu856/papers/blob/master/summaries/On%20Calibration%20of%20Modern%20Neural%20Networks.md). 

- For quantitative evaluation, they use *Expected Calibration Error*, *Average Calibration Error* (like ECE, but each bin is assigned an equal weight) and *Maximum Calibration Error*. They compare softmax, softmax + temperature scaling, softmax + linear scaling (similar to temperature scaling), relaxed softmax and relaxed softmax + linear scaling. They utilize two datasets: Caltech and NightOwls (models are trained on the train sets, linear scaling and temperature scaling are tuned on the val sets, and all metrics are computed on the test sets).

- On Caltech, relaxed softmax + linear scaling gives the best calibration metrics, ahead of softmax + temperature scaling. On NightOwl, relaxed softmax is the winner, just ahead of relaxed softmax + linear scaling. The relaxed softmax methods also achieve somewhat worse miss rate metrics (13.26% versus 10.17% on Caltech, I'm not sure how significant of a decrease that actually is).

### Comments:
- Quite interesting paper, but I am not fully convinced. For example, I find it odd that relaxed softmax beats softmax + temperature scaling on NightOwl but not on Caltech.

- It might be that I am missing something, but I also struggle to understand some of their explanations and arguments, e.g. in the final paragraph of section 3.2. I am not quite sure if the network-outputted scale alpha actually does what they say it does.
