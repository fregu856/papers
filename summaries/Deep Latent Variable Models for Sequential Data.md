##### [19-11-04] [paper4]
- Variational Inference MPC for Bayesian Model-based Reinforcement Learning [[pdf]](https://arxiv.org/abs/1907.04202) [[pdf with comments]](https://github.com/fregu856/papers_private/blob/master/commented_pdfs/Variational%20Inference%20MPC%20for%20Bayesian%20Model-based%20Reinforcement%20Learning.pdf)
- *Masashi Okada, Tadahiro Taniguchi*
- `2019-07-08, CoRL 2019`

****

### General comments on paper quality:
- Interesting and quite well-written paper. Quite difficult to follow all steps in their proposed approach, reading "Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review" (Sergey Levine) first would probably give a more comprehensive introduction to the framework of "control as inference".

### Comments:
- The proposed approach is interesting and quite neat, it makes intuitive sense that explicitly trying to capture multi-modality also in the space of trajectories / action sequences could improve performance, and their experimental results seem to suggest that this is indeed true.

- According to the authors, this approach could also be applied to e.g. PlaNet, and it seems reasonable to assume that it might improve performance.

- The approach does therefore definitely seem interesting, but I don't fully understand it, and I think there are other more lower-hanging possible improvements you could try to implement first.
