##### [20-01-15] [paper79]
- Generative Modeling by Estimating Gradients of the Data Distribution [[pdf]](https://arxiv.org/abs/1907.05600) [[code]](https://github.com/ermongroup/ncsn) [[poster]](https://yang-song.github.io/papers/NeurIPS2019/ncsn-poster.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Generative%20Modeling%20by%20Estimating%20Gradients%20of%20the%20Data%20Distribution.pdf)
- *Yang Song, Stefano Ermon*
- `2019-07-12, NeurIPS 2019`

****

### General comments on paper quality:
- Well-written and quite interesting paper.

### Comments:
- The examples in section 3 are neat and quite pedagogical.

- I would probably need to read a couple of papers covering the basics of score matching, and then come back and read this paper again to fully appreciate it.

- Like they write, their training method could be used to train an EBM (by replacing their score network with the gradient of the energy in the EBM). This would then be just like "denoising score matching", but combining multiple noise levels in a combined objective? 

- I suppose that their annealed Langevin approach could also be used to sample from an EBM. This does however seem very computationally expensive, as they run T=100 steps of Langevin dynamics for each of the L=10 noise levels?
