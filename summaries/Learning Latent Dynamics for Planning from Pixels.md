##### [19-11-22] [paper65]
- Learning Latent Dynamics for Planning from Pixels [[pdf]](https://arxiv.org/abs/1811.04551) [[code]](https://github.com/google-research/planet) [[blog]](https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20Latent%20Dynamics%20for%20Planning%20from%20Pixels.pdf)
- *Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson*
- `2018-11-12, ICML2019`

****

### General comments on paper quality:
- Very well-written and interesting paper! Very good introduction to the entire field of model-based RL I feel like.

### Comments:
- Seems quite odd to me that they spend an entire page on "Latent overshooting", but then don't actually use it for their RSSM model?

- It's not entirely clear to me how this approach differs from "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models" (PETS), apart from the fact that PETS actually has access to the state (so, they don't need to apply VAE stuff to construct a latent state representation).

- The provided code seems like it could be very useful. Is it easy to use? The model seems to train on just 1 GPU in just 1 day anyway, which is good.
