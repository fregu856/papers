##### [18-12-05] [paper24]
- On Calibration of Modern Neural Networks [[pdf]](https://arxiv.org/abs/1706.04599) [[code]](https://github.com/gpleiss/temperature_scaling) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20Calibration%20of%20Modern%20Neural%20Networks.pdf)
- *Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger*
- `2017-08-03, ICML2017`

****

### General comments on paper quality:
- Well-written and quite interesting paper. Not a particularly heavy read.

### Paper overview:
- The authors study the concept of confidence calibration in classification models. A model is said to be calibrated (or well-calibrated) if the confidence score corresponding to its prediction actually is representative of the true correctness likelihood, i.e., if the model outputs a confidence score of 0.75 for 1000 examples, roughly 750 of those should be correctly classified by the model.

- They empirically find that modern neural networks (e.g. ResNets) usually are poorly calibrated, outputting overconfident predictions (whereas old networks, e.g. LeNet, usually were well-calibrated). They e.g. find that while increasing network depth or width often improves the classification accuracy, it also has a negative effect on model calibration.

- The authors then describe a few post-processing methods designed to improve model calibration, all of which require a validation set (you fix the network weights, learn to modify the outputted confidence score based on the validation set and then hope for the model to stay well-calibrated also on the test set). They also introduce a very simple calibration method, named *temperature scaling*, in which you learn (optimize on the validation set) a single scalar T, which is used to scale the logits z outputted by the model (new_conf_score = max_k{softmax(z/T)_k}).

- They compare these calibration methods on six different image classification datasets (e.g. ImageNet and CIFAR100) and 4 document classification datasets, using different CNNs (e.g. ResNet and DenseNet). Surprising to the authors, they find the simple temperature scaling method to achieve the best overall performance (most well-calibrated confidence scores), often having a significant positive effect on calibration. 

### Comments:
- Quite interesting paper, and the effectiveness of temperature scaling is actually quite impressive. Since the authors assume that the train, val and test sets are drawn from the same data distribution, it would however be interesting to evaluate the calibration also on out-of-distribution data. If we train a model on MNIST train, use temperature scaling on MNIST val (and thus obtain quite well-calibrated confidence scores on MNIST test), would it then also be more well-calibrated on e.g. notMNIST?
