##### [19-02-13] [paper42]
-  Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning [[pdf]](https://arxiv.org/abs/1902.03932) [[code]](https://github.com/ruqizhang/csgmcmc) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Cyclical%20Stochastic%20Gradient%20MCMC%20for%20Bayesian%20Deep%20Learning.pdf)
- *Ruqi Zhang, Chunyuan Li, Jianyi Zhang, Changyou Chen, Andrew Gordon Wilson*
- `2019-02-11`

****

### General comments on paper quality:
- Well-written and VERY interesting paper (did find a couple of typos though).

### Comments:
- Very interesting method. I have however done some experiments using their code, and I find that samples from the same cycle produce very similar predictions. Thus I am somewhat skeptical that the method actually is significantly better than snapshot-ensembling, or just regular ensembling for that matter. The results in table 3 do seem to suggest that there is something to gain from collecting more than just one sample per cycle though, right? I need to do more experiments and investigate this further.

- Must admit that I struggled to understand much of section 4, I am thus not really sure how impressive their theoretical results actually are.
