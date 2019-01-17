##### [19-01-17] [paper28]
- How Does Batch Normalization Help Optimization? [[pdf]](https://arxiv.org/abs/1805.11604) [[poster]](http://people.csail.mit.edu/tsipras/batchnorm_poster.pdf) [[video]](https://youtu.be/ZOabsYbmBRM) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/How%20Does%20Batch%20Normalization%20Help%20Optimization%3F.pdf)
- *Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, Aleksander Madry*
- `2018-10-27, NeurIPS2018`

****

(I had to make this summary less detailed than usual)

### General comments on paper quality:
- Well-written and interesting paper. A recommended read if you have ever been given the explanation that batch normalization works because it reduces the internal covariate shift (ICS). 

### Paper overview:
- The abstract summarizes the paper very well:
- - _Batch Normalization (BatchNorm) is a widely adopted technique that enables faster and more stable training of deep neural networks (DNNs). Despite its pervasiveness, the exact reasons for BatchNorm's effectiveness are still poorly understood. The popular belief is that this effectiveness stems from controlling the change of the layers' input distributions during training to reduce the so-called "internal covariate shift". In this work, we demonstrate that such distributional stability of layer inputs has little to do with the success of BatchNorm. Instead, we uncover a more fundamental impact of BatchNorm on the training process: it makes the optimization landscape significantly smoother. This smoothness induces a more predictive and stable behavior of the gradients, allowing for faster training._

- _In this work, we have investigated the roots of BatchNorm’s effectiveness as a technique for training deep neural networks. We find that the widely believed connection between the performance of BatchNorm and the internal covariate shift is tenuous, at best. In particular, we demonstrate that existence of internal covariate shift, at least when viewed from the - generally adopted – distributional stability perspective, is not a good predictor of training performance. Also, we show that, from an optimization viewpoint, BatchNorm might not be even reducing that shift._

- _Instead, we identify a key effect that BatchNorm has on the training process: it reparametrizes the underlying optimization problem to make it more stable (in the sense of loss Lipschitzness) and smooth (in the sense of “effective” β-smoothness of the loss). This implies that the gradients used in
training are more predictive and well-behaved, which enables faster and more effective optimization._

- _We also show that this smoothing effect is not unique to BatchNorm. In fact, several other natural normalization strategies have similar impact and result in a comparable performance gain._
