##### [20-03-27] [paper94]
- Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning [[pdf]](https://arxiv.org/abs/2002.06470) [[code]](https://github.com/bayesgroup/pytorch-ensembles) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Pitfalls%20of%20In-Domain%20Uncertainty%20Estimation%20and%20Ensembling%20in%20Deep%20Learning.pdf)
- *Arsenii Ashukha, Alexander Lyzhov, Dmitry Molchanov, Dmitry Vetrov*
- `2020-02-15, ICLR 2020`

****

### General comments on paper quality:
- Quite well-written and interesting paper.

### Comments:
- The number of compared methods is quite impressive.

- The paper provides further evidence for what intuitively makes A LOT of sense: "Deep ensembles dominate other methods given a fixed test-time budget. The results indicate, in particular, that exploration of different modes in the loss landscape is crucial for good predictive performance". While deep ensembles might require a larger amount of total training time, they are extremely simple to train and separate ensemble members can be trained completely in parallel. Overall then, deep ensembles is a baseline that's extremely hard to beat IMO.

- Not convinced that "calibrated log-likelihood" is an ideal metric that addresses the described flaws of commonly used metrics. For example, "...especially calibrated log-likelihood is highly correlated with accuracy" does not seem ideal. Also, how would you generalize it to regression? 
