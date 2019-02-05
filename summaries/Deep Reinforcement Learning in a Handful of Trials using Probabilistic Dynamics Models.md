##### [19-02-05] [paper38]
- Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models [[pdf]](https://arxiv.org/abs/1805.12114) [[poster]](https://kchua.github.io/misc/poster.pdf) [[video]](https://youtu.be/3d8ixUMSiL8) [[code]](https://github.com/kchua/handful-of-trials) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Reinforcement%20Learning%20in%20a%20Handful%20of%20Trials%20using%20Probabilistic%20Dynamics%20Models.pdf)
- *Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine*
- `2018-05-30, NeurIPS2018`

****

### General comments on paper quality:
- Well-written and very interesting paper. It applies relatively common methods for uncertainty estimation (ensemble of probabilistic NNs) to an interesting problem in RL and shows promising results.

### Paper overview:
- The authors present a model-based RL algorithm called _Probabilistic Ensembles with Trajectory Sampling (PETS)_, that (at least roughly) matches the asymptotic performance of SOTA model-free algorithms on four control tasks, while requiring significantly fewer samples (model-based algorithms generally have much better sample efficiency, but worse asymptotic performance than the best model-free algorithms).

- They use an ensemble of probabilistic NNs (Probabilistic Ensemble, PE) to learn a probabilistic dynamics model, p_theta(s_{t+1} | s_t, a_t), where s_t is the state and a_t is the taken action at time t. 

- A probabilistic NN outputs the parameters of a probability distribution, in this case by outputting the mean, mu(s_t, a_t), and diagonal covariance matrix, SIGMA(s_t, a_t), of a Gaussian, enabling estimation of aleatoric (data) uncertainty. To also estimate epistemic (model) uncertainty, they train an ensemble of B probabilistic NNs.

- The B ensemble models are trained on separate (but overlapping) datasets: for each ensemble model, a dataset is created by drawing N examples with replacement from the original dataset D (which also contains N examples).

- The B ensemble models are then used in the trajectory sampling step, where P state particles s_t_p are propagated forward in time by iteratively sampling s_{t+1}_p ~ p_theta_b(s_{t+1}_p | s_t_p, a_t)). I.e., each ensemble model outputs a distribution, and we sample particles from these B distributions. This results in P trajectory samples, The authors used P=20, B=5 in all their experiments.

- Based on these P state trajectory samples, s_{t:t+T}_p (which we hope approximate the true distribution over trajectories s_{t:t+T}), MPC is finally used to compute the next action a_t. 

### Comments:
- Interesting method. Should be possible to benchmark various uncertainty estimation techniques using their setup, just like they compare probabilistic/deterministic ensembles and probabilistic networks. I found it quite interesting that a single probabilistic network (at least somewhat) outperformed a deterministic ensemble (perhaps this would change with a larger ensemble size though?).

- Do you actually need to perform the bootstrap procedure when training the ensemble, or would you get the same performance by simply training all B models on the same dataset D?

- I struggle somewhat to understand their method for lower/upper bounding the outputted variance during testing (appendix A.1). Do you actually need this? Also, I do not quite get the lines of code. Are max_logvar and min_logvar variables?
