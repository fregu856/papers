##### [19-11-29] [paper67]
- Dream to Control: Learning Behaviors by Latent Imagination [[pdf]](https://openreview.net/forum?id=S1lOTC4tDS) [[webpage]](https://dreamrl.github.io/) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Dream%20to%20Control%20Learning%20Behaviors%20by%20Latent%20Imagination.pdf)
- *Anonymous*
- `2019-09`

****


### General comments on paper quality:
- Interesting and very well-written paper. A recommended read, even if you just want to gain an improved understanding of state-of-the-art RL in general and the PlaNet paper ("Learning Latent Dynamics for Planning from Pixels") in particular.

### Comments:
- Very similar to PlaNet, the difference is that they here learn an actor-critic model on-top of the learned dynamics, instead of doing planning using MPC.

- The improvement over PlaNet, in terms of experimental results, seems significant.

- Since they didn't actually use the latent overshooting in the PlaNet paper, I assume they don't use it here either?
