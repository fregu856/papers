##### [20-09-21] [paper105]
- Uncertainty Estimation Using a Single Deep Deterministic Neural Network [[pdf]](https://arxiv.org/abs/2003.02037) [[code]](https://github.com/y0ast/deterministic-uncertainty-quantification) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20Estimation%20Using%20a%20Single%20Deep%20Deterministic%20Neural%20Network.pdf)
- *Joost van Amersfoort, Lewis Smith, Yee Whye Teh, Yarin Gal*
- `2020-03-04, ICML 2020`

****

### General comments on paper quality:
- Well-written and quite interesting paper.

### Comments:
- Interesting and neat idea, it definitely makes some intuitive sense.

- In the end though, I was not overly impressed. Once they used the more realistic setup on the CIFAR10 experiment (not using a third dataset to tune lambda), the proposed method was outperformed by ensembling (also using quite few networks). Yes, their method is more computationally efficient at test time (which is indeed very important in many applications), but it also seems quite a lot less convenient to train, involves setting a couple of important hyperparameters and so on. Interesting method and a step in the right direction though.
