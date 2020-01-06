##### [20-01-06] [paper75]
- Maximum Entropy Generators for Energy-Based Models [[pdf]](https://arxiv.org/abs/1901.08508) [[code]](https://github.com/ritheshkumar95/energy_based_generative_models) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Maximum%20Entropy%20Generators%20for%20Energy-Based%20Models.pdf)
- *Rithesh Kumar, Sherjil Ozair, Anirudh Goyal, Aaron Courville, Yoshua Bengio*
- `2019-01-24`

****

### General comments on paper quality:
Quite well-written and interesting paper.

### Comments:
The general idea, learning an energy-based model p_theta by drawing samples from an approximating distribution (that minimizes the KL divergence w.r.t p_theta) instead of generating approximate samples from p_theta using MCMC, is interesting and intuitively makes quite a lot of sense IMO. 

Since the paper was written prior to the recent work on MCMC-based learning ([Learning Non-Convergent Non-Persistent Short-Run MCMC Toward Energy-Based Model](https://github.com/fregu856/papers/blob/master/summaries/Learning%20Non-Convergent%20Non-Persistent%20Short-Run%20MCMC%20Toward%20Energy-Based%20Model.md), [Implicit Generation and Generalization in Energy-Based Models](https://github.com/fregu856/papers/blob/master/summaries/Implicit%20Generation%20and%20Generalization%20in%20Energy-Based%20Models.md), [On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models](https://github.com/fregu856/papers/blob/master/summaries/On%20the%20Anatomy%20of%20MCMC-Based%20Maximum%20Likelihood%20Learning%20of%20Energy-Based%20Models.md))
