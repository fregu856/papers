# About

I categorize, annotate and write comments for all research papers I read as a PhD student.

#### Categories:

[Uncertainty Estimation], [Ensembling], [Stochastic Gradient MCMC], [Variational Inference], [Out-of-Distribution Detection], [Theoretical Properties of Deep Learning], [VAEs], [Normalizing Flows], [Autonomous Driving], [Medical ML], [Object Detection], [3D Object Detection], [3D Multi-Object Tracking], [3D Human Pose Estimation], [Visual Tracking], [Sequence Modeling], [Reinforcement Learning], [System Identification], [Energy-Based Models], [Neural Processes], [Neural ODEs], [Transformers], [Implicit Neural Representations]


### Papers:

- [Papers Read in 2022](#papers-read-in-2022)
- [Papers Read in 2021](#papers-read-in-2021)
- [Papers Read in 2020](#papers-read-in-2020)
- [Papers Read in 2019](#papers-read-in-2019)
- [Papers Read in 2018](#papers-read-in-2018)

#### Papers Read in 2022:

##### [22-03-04] [paper199]
- NUQ: Nonparametric Uncertainty Quantification for Deterministic Neural Networks
 [[pdf]](https://arxiv.org/abs/2202.03101) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/NUQ:%20Nonparametric%20Uncertainty%20Quantification%20for%20Deterministic%20Neural%20Networks.pdf)
- `2022-02, NeurIPS 2022`
- [Uncertainty Estimation] [Out-of-Distribution Detection]
```
Interesting paper. I found it difficult to understand Section 2, I wouldn't really be able to implement their proposed NUQ method. Only image classification, but their experimental evaluation is still quite extensive. And, they obtain strong performance.
```

##### [22-03-03] [paper198]
- On the Practicality of Deterministic Epistemic Uncertainty
 [[pdf]](https://openreview.net/forum?id=W3-hiLnUYl) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20the%20Practicality%20of%20Deterministic%20Epistemic%20Uncertainty.pdf)
- `2021-09`
- [Uncertainty Estimation]
```
Interesting and well-written paper. Their evaluation with the corrupted datasets makes sense I think. The results are interesting,  the fact that ensembling/MC-dropout consistently outperforms the other methods. Another reminder of how strong of a baseline ensembling is when it comes to uncertainty estimation? Also, I think that their proposed rAULC is more or less equivalent to AUSE (area under the sparsification error curve)?
```

##### [22-03-03] [paper197]
- Transformers Can Do Bayesian Inference
 [[pdf]](https://arxiv.org/abs/2112.10510) [[code]](https://github.com/automl/TransformersCanDoBayesianInference) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Transformers%20Can%20Do%20Bayesian%20Inference.pdf)
- *Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, Frank Hutter*
- `2021-12-20, ICLR 2022`
- [Transformers]
```
Quite interesting and well-written paper. I did however find it difficult to properly understand everything, it feels like a lot of details are omitted (I wouldn't really know how to actually implement this in practice). It's difficult for me to judge how impressive the results are or how practically useful this approach actually might be, what limitations are there? Overall though, it does indeed seem quite interesting.
```

##### [22-03-02] [paper196]
- A Deep Bayesian Neural Network for Cardiac Arrhythmia Classification with Rejection from ECG Recordings
 [[pdf]](https://arxiv.org/abs/2203.00512) [[code]](https://github.com/hsd1503/ecg_uncertainty) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Deep%20Bayesian%20Neural%20Network%20for%20Cardiac%20Arrhythmia%20Classification%20with%20Rejection%20from%20ECG%20Recordings.pdf)
- *Wenrui Zhang, Xinxin Di, Guodong Wei, Shijia Geng, Zhaoji Fu, Shenda Hong*
- `2022-02-26`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Medical ML]](#medical-ml)
```
Somewhat interesting paper. They use a softmax model with MC-dropout to compute uncertainty estimates. The evaluation is not very extensive, they mostly just check that the classification accuracy improves as they reject more and more samples based on a uncertainty threshold.
```

##### [22-02-26] [paper195]
- Out of Distribution Data Detection Using Dropout Bayesian Neural Networks
 [[pdf]](https://arxiv.org/abs/2202.08985) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Out%20of%20Distribution%20Data%20Detection%20Using%20Dropout%20Bayesian%20Neural%20Networks.pdf)
- *Andre T. Nguyen, Fred Lu, Gary Lopez Munoz, Edward Raff, Charles Nicholas, James Holt*
- `2022-02-18, AAAI 2022`
- [[Out-of-Distribution Detection]](#out-of-distribution-detection)
```
Quite interesting and well-written paper. It seemed quite niche at first, but I think their analysis could potentially be useful.
```

##### [22-02-26] [paper194]
- Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks
 [[pdf]](https://arxiv.org/abs/1706.02690) [[code]](https://github.com/facebookresearch/odin) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Enhancing%20The%20Reliability%20of%20Out-of-distribution%20Image%20Detection%20in%20Neural%20Networks.pdf)
- *Shiyu Liang, Yixuan Li, R. Srikant*
- `2017-06-08, ICLR 2018`
- [[Out-of-Distribution Detection]](#out-of-distribution-detection)
```
Quite interesting and well-written paper. Two simple modifications of the "maximum softmax score" baseline, and the performance is consistently improved. The input perturbation method is quite interesting. Intuitively, it's not entirely clear to me why it actually works.
```

##### [22-02-25] [paper193]
- Confidence-based Out-of-Distribution Detection: A Comparative Study and Analysis
 [[pdf]](https://arxiv.org/abs/2107.02568) [[code]](https://github.com/christophbrgr/ood_detection_framework) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Confidence-based%20Out-of-Distribution%20Detection:%20A%20Comparative%20Study%20and%20Analysis.pdf)
- *Christoph Berger, Magdalini Paschali, Ben Glocker, Konstantinos Kamnitsas*
- `2021-07-06, MICCAI Workshops 2021`
- [[Out-of-Distribution Detection]](#out-of-distribution-detection) [[Medical ML]](#medical-ml)
```
Interesting and well-written paper. Interesting that Mahalanobis works very well on the CIFAR10 vs SVHN but not on the medical imaging dataset. I don't quite get how/why the ODIN method works, I'll probably have to read that paper.
```

##### [22-02-25] [paper192]
- Deep Learning Through the Lens of Example Difficulty
 [[pdf]](https://openreview.net/forum?id=WWRBHhH158K) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Learning%20Through%20the%20Lens%20of%20Example%20Difficulty.pdf)
- *Robert John Nicholas Baldock, Hartmut Maennel, Behnam Neyshabur*
- `2021-05-21, NeurIPS 2021`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning)
```
Quite interesting and well-written paper. The definition of "prediction depth" in Section 2.1 makes sense, and it definitely seems reasonable that this could correlate with example difficulty / prediction confidence in some way. Section 3 and 4, and all the figures, contain a lot of info it seems, I'd probably need to read the paper again to properly understand/appreciate everything.
```

##### [22-02-24] [paper191]
- UncertaINR: Uncertainty Quantification of End-to-End Implicit Neural Representations for Computed Tomography
 [[pdf]](https://arxiv.org/abs/2202.10847) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/UncertaINR:%20Uncertainty%20Quantification%20of%20End-to-End%20Implicit%20Neural%20Representations%20for%20Computed%20Tomography.pdf)
- *Francisca Vasconcelos, Bobby He, Nalini Singh, Yee Whye Teh*
- `2022-02-22`
- [[Implicit Neural Representations]](#implicit-neural-representations) [[Uncertainty Estimation]](#uncertainty-estimation) [[Medical ML]](#medical-ml)
```
Interesting and well-written paper. I wasn't very familiar with CT image reconstruction, but they do a good job explaining everything. Interesting that MC-dropout seems important for getting well-calibrated predictions.
```

##### [22-02-21] [paper190]
- Can You Trust Predictive Uncertainty Under Real Dataset Shifts in Digital Pathology?
 [[pdf]](https://backend.orbit.dtu.dk/ws/portalfiles/portal/218217360/MICCAI2020.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Can%20you%20trust%20predictive%20uncertainty%20under%20real%20dataset%20shifts%20in%20digital%20pathology%3F.pdf)
- *Jeppe Thagaard, Søren Hauberg, Bert van der Vegt, Thomas Ebstrup, Johan D. Hansen, Anders B. Dahl*
- `2020-09, MICCAI 2020`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Out-of-Distribution Detection]](#out-of-distribution-detection) [[Medical ML]](#medical-ml)
```
Quite interesting and well-written paper. They compare MC-dropout, ensemlbing and mixup (and with a standard softmax classifer as the baseline). Nothing groundbreaking, but the studied application (classification of pathology slides for cancer) is very interesting. The FPR95 metrics for OOD detection in Table 4 are terrible for ensembling, but the classification accuracy (89.7) is also pretty much the same as for D_test_int in Tabe 3 (90.1)? So, it doesn't really matter that the model isn't capable of distinguishing this "OOD" data from in-distribution? 
```

##### [22-02-21] [paper189]
- Robust Uncertainty Estimates with Out-of-Distribution Pseudo-Inputs Training
 [[pdf]](https://arxiv.org/abs/2201.05890) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Robust%20uncertainty%20estimates%20with%20out-of-distribution%20pseudo-inputs%20training.pdf)
- *Pierre Segonne, Yevgen Zainchkovskyy, Søren Hauberg*
- `2022-01-15`
- [[Uncertainty Estimation]](#uncertainty-estimation)
```
Somewhat interesting paper. I didn't quite understand everything, so it could be more interesting than I think. The fact that their pseudo-input generation process "relies on the availability of a differentiable density estimate of the data" seems like a big limitation? For regression, they only applied their method to very low-dimensional input data (1D toy regression and UCI benchmarks), but would this work for image-based tasks?
```

##### [22-02-19] [paper188]
- Contrastive Training for Improved Out-of-Distribution Detection
 [[pdf]](https://arxiv.org/abs/2007.05566) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Contrastive%20Training%20for%20Improved%20Out-of-Distribution%20Detection.pdf)
- *Jim Winkens, Rudy Bunel, Abhijit Guha Roy, Robert Stanforth, Vivek Natarajan, Joseph R. Ledsam, Patricia MacWilliams, Pushmeet Kohli, Alan Karthikesalingam, Simon Kohl, Taylan Cemgil, S. M. Ali Eslami, Olaf Ronneberger*
- `2020-07-10`
- [[Out-of-Distribution Detection]](#out-of-distribution-detection)
```
Quite interesting and very well-written paper. They take the method from the Mahalanobis paper ("A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks") (however, they fit Gaussians only to the features at the second-to-last network layer, and they don't use the input pre-processing either) and consistently improve OOD detection performance by incorporating contrastive training. Specifically, they first train the network using just the SimCLR loss for a large number of epochs, and then also add the standard classification loss. I didn't quite get why the label smoothing is necessary, but according to Table 2 it's responsible for a large portion of the performance gain.
```

##### [22-02-19] [paper187]
- A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks
 [[pdf]](https://arxiv.org/abs/1807.03888) [[code]](https://github.com/pokaxpoka/deep_Mahalanobis_detector) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Simple%20Unified%20Framework%20for%20Detecting%20Out-of-Distribution%20Samples%20and%20Adversarial%20Attacks.pdf)
- *Kimin Lee, Kibok Lee, Honglak Lee, Jinwoo Shin*
- `2018-07-10, NeurIPS 2018`
- [[Out-of-Distribution Detection]](#out-of-distribution-detection)
```
Well-written and interesting paper. The proposed method is simple and really neat: fit class-conditional Gaussians in the feature space of a pre-trained classifier (basically just LDA on the feature vectors), and then use the Mahalanobis distance to these Gaussians as the confidence score for input x. They then also do this for the features at multiple levels of the network and combine these confidence scores into one. I don't quite get why the "input pre-processing" in Section 2.2 (adding noise to test samples) works, in Table 1 it significantly improves the performance.
```

##### [22-02-19] [paper186]
- Noise Contrastive Priors for Functional Uncertainty
 [[pdf]](https://arxiv.org/abs/1807.09289) [[code]](https://github.com/brain-research/ncp) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Noise%20Contrastive%20Priors%20for%20Functional%20Uncertainty.pdf)
- *Danijar Hafner, Dustin Tran, Timothy Lillicrap, Alex Irpan, James Davidson*
- `2018-07-24, UAI 2019`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Out-of-Distribution Detection]](#out-of-distribution-detection)
```
Quite interesting and well-written paper. Only experiments on a toy 1D regression problem, and flight delay prediction in which the input is 8D. The approach of just adding noise to the input x to get OOD samples would probably not work very well e.g. for image-based problems?
```

##### [22-02-18] [paper185]
- Does Your Dermatology Classifier Know What It Doesn't Know? Detecting the Long-Tail of Unseen Conditions
 [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S1361841521003194?via%3Dihub) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Does%20your%20dermatology%20classifier%20know%20what%20it%20doesn't%20know%3F%20Detecting%20the%20long-tail%20of%20unseen%20conditions.pdf)
- *Abhijit Guha Roy, Jie Ren, Shekoofeh Azizi, Aaron Loh, Vivek Natarajan, Basil Mustafa, Nick Pawlowski, Jan Freyberg, Yuan Liu, Zach Beaver, Nam Vo, Peggy Bui, Samantha Winter, Patricia MacWilliams, Greg S. Corrado, Umesh Telang, Yun Liu, Taylan Cemgil, Alan Karthikesalingam, Balaji Lakshminarayanan, Jim Winkens*
- `2021-04-08, Medical Image Analysis (January 2022)`
- [[Out-of-Distribution Detection]](#out-of-distribution-detection) [[Medical ML]](#medical-ml)
```
Well-written and interesting paper. Quite long, so it took a bit longer than usual to read it. Section 1 and 2 gives a great overview of OOD detection in general, and how it can be used specifically in this dermatology setting. I can definitely recommend reading Section 2 (Related work). They assume access to some outlier data during training, so their approach is similar to the "Outlier exposure" method (specifically in this dermatology setting, they say that this is a fair assumption). Their method is an improvement of the "reject bucket" (add an extra class which you assign to all outlier training data points), in their proposed method they also use fine-grained classification of the outlier skin conditions. Then they also use an ensemble of 5 models, and also a more diverse ensemble (in which they combine models trained with different representation learning techniques). This diverse ensemble obtains the best performance.
```

##### [22-02-16] [paper184]
- Being a Bit Frequentist Improves Bayesian Neural Networks
 [[pdf]](https://arxiv.org/abs/2106.10065) [[code]](https://github.com/wiseodd/bayesian_ood_training) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Being%20a%20Bit%20Frequentist%20Improves%20Bayesian%20Neural%20Networks.pdf)
- *Agustinus Kristiadi, Matthias Hein, Philipp Hennig*
- `2021-06-18, AISTATS 2022`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Out-of-Distribution Detection]](#out-of-distribution-detection)
```
Interesting and well-written paper. The proposed method makes intuitive sense, trying to incorporate the "OOD training" method (i.e., to use some kind of OOD data during training, similar to e.g. the "Deep Anomaly Detection with Outlier Exposure" paper) into the Bayesian deep learning approach. The experimental results do seem quite promising.
```

##### [22-02-15] [paper183]
- Mixtures of Laplace Approximations for Improved Post-Hoc Uncertainty in Deep Learning
 [[pdf]](https://arxiv.org/abs/2111.03577) [[code]](https://github.com/AlexImmer/Laplace) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Mixtures%20of%20Laplace%20Approximations%20for%20Improved%20Post-Hoc%20Uncertainty%20in%20Deep%20Learning.pdf)
- *Runa Eschenhagen, Erik Daxberger, Philipp Hennig, Agustinus Kristiadi*
- `2021-11-95, NeurIPS Workshops 2021`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Out-of-Distribution Detection]](#out-of-distribution-detection)
```
Well-written and interesting paper. Short paper of just 3 pages, but with an extensive appendix which I definitely recommend going through. The method, training an ensemble and then applying the Laplace approximation to each network, is very simple and intuitively makes a lot of sense. I didn't realize that this would have basically the same test-time speed as ensembling (since they utilize that probit approximation), that's very neat. It also seems to consistently outperform ensembling a bit across almost all tasks and metrics.
```

##### [22-02-15] [paper182]
- Pessimistic Bootstrapping for Uncertainty-Driven Offline Reinforcement Learning
 [[pdf]](https://openreview.net/forum?id=Y4cs1Z3HnqL) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Pessimistic%20Bootstrapping%20for%20Uncertainty-Driven%20Offline%20Reinforcement%20Learning.pdf)
- *Chenjia Bai, Lingxiao Wang, Zhuoran Yang, Zhi-Hong Deng, Animesh Garg, Peng Liu, Zhaoran Wang*
- `2021-09-29, ICLR 2022`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Reinforcement Learning]](#reinforcement-learning)
```
Well-written and somewhat interesting paper. I'm not overly familiar with RL, which makes it a bit difficult for me to properly evaluate the paper's contributions. They use standard ensembles for uncertainty estimation combined with an OOD sampling regularization. I thought that the OOD sampling could be interesting, but it seems very specific to RL. I'm sure this paper is quite interesting for people doing RL, but I don't think it's overly useful for me.
```

##### [22-02-15] [paper181]
- On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks
 [[pdf]](https://openreview.net/forum?id=aPOpXlnV1T) [[code]](https://sites.google.com/view/pitfalls-uncertainty?authuser=0) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20the%20Pitfalls%20of%20Heteroscedastic%20Uncertainty%20Estimation%20with%20Probabilistic%20Neural%20Networks.pdf)
- *Maximilian Seitzer, Arash Tavakoli, Dimitrije Antic, Georg Martius*
- `2021-09-29, ICLR 2022`
- [[Uncertainty Estimation]](#uncertainty-estimation)
```
Quite interesting and very well-written paper, I enjoyed reading it. Their analysis of fitting Gaussian regression models via the NLL is quite interesting, I didn't really expect to learn something new about this. I've seen Gaussian models outperform standard regression (L2 loss) w.r.t. accuracy in some applications/datasets, and it being the other way around in others. In the first case, I've then attributed the success of the Gaussian model to the "learned loss attenuation". The analysis in this paper could perhaps explain why you get this performance boost only in certain applications. Their beta-NLL loss could probably be quite useful, seems like a convenient tool to have.
```

##### [22-02-15] [paper180]
- Sample Efficient Deep Reinforcement Learning via Uncertainty Estimation
 [[pdf]](https://openreview.net/forum?id=vrW3tvDfOJQ) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Sample%20Efficient%20Deep%20Reinforcement%20Learning%20via%20Uncertainty%20Estimation.pdf)
- *Vincent Mai, Kaustubh Mani, Liam Paull*
- `2021-09-29, ICLR 2022`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Reinforcement Learning]](#reinforcement-learning)
```
Well-written and somewhat interesting paper. I'm not overly familiar with reinforcement learning, which makes it a bit difficult for me to properly evaluate the paper's contributions, but to me it seems like fairly straightforward method modifications? To use ensembles of Gaussian models (instead of ensembles of models trained using the L2 loss) makes sense. The BIV method I didn't quite get, it seems rather ad hoc? I also don't quite get exactly how it's used in equation (10), is the ensemble of Gaussian models trained _jointly_ using this loss? I don't really know if this could be useful outside of RL.
```

##### [22-02-14] [paper179]
- Laplace Redux -- Effortless Bayesian Deep Learning
 [[pdf]](https://arxiv.org/abs/2106.14806) [[code]](https://github.com/AlexImmer/Laplace) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Laplace%20Redux%20--%20Effortless%20Bayesian%20Deep%20Learning.pdf)
- *Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, Philipp Hennig*
- `2021-06-28, NeurIPS 2021`
- [[Uncertainty Estimation]](#uncertainty-estimation)
```
Interesting and very well-written paper, I enjoyed reading it. I still think that ensembling probably is quite difficult to beat purely in terms of uncertainty estimation quality, but this definitely seems like a useful tool in many situations. It's not clear to me if the analytical expression for regression in "4. Approximate Predictive Distribution" is applicable also if the variance is input-dependent?
```

##### [22-02-12] [paper178]
- Benchmarking Uncertainty Quantification on Biosignal Classification Tasks under Dataset Shift
 [[pdf]](https://arxiv.org/abs/2112.09196?context=cs) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Benchmarking%20Uncertainty%20Quantification%20on%20Biosignal%20Classification%20Tasks%20under%20Dataset%20Shift.pdf)
- *Tong Xia, Jing Han, Cecilia Mascolo*
- `2021-12-16, AAAI Workshops 2022`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Out-of-Distribution Detection]](#out-of-distribution-detection) [[Medical ML]](#medical-ml)
```
Well-written and interesting paper. They synthetically create dataset shifts (e.g. by adding Gaussian noise to the data) of increasing intensity and study whether or not the uncertainty increases as the accuracy degrades. They compare regular softmax, temperature scaling, MC-dropout, ensembling and a simple variational inference method. Their conclusion is basically that ensembling slightly outperforms the other methods, but that no method performs overly well. I think these type of studies are really useful.
```

##### [22-02-12] [paper177]
- Deep Evidential Regression
 [[pdf]](https://arxiv.org/abs/1910.02600) [[code]](https://github.com/aamini/evidential-deep-learning) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Evidential%20Regression.pdf)
- *Alexander Amini, Wilko Schwarting, Ava Soleimany, Daniela Rus*
- `2019-10-07, NeurIPS 2020`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Out-of-Distribution Detection]](#out-of-distribution-detection)
```
Well-written and interesting paper. This is a good paper to read before "Natural Posterior Network: Deep Bayesian Predictive Uncertainty for Exponential Family Distributions". Their proposed method seems to have similar / slightly worse performance than a small ensemble, so the only real advantage is that it's faster at time-time? This is of course very important in many applications, but not in all. The performance also seems quite sensitive to the choice of lambda in the combined loss function (Equation (10)), according to Figure S2 in the appendix?
```

##### [22-02-11] [paper176]
- On Out-of-distribution Detection with Energy-based Models
 [[pdf]](https://arxiv.org/abs/2107.08785) [[code]](https://github.com/selflein/EBM-OOD-Detection) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20Out-of-distribution%20Detection%20with%20Energy-based%20Models.pdf)
- *Sven Elflein, Bertrand Charpentier, Daniel Zügner, Stephan Günnemann*
- `2021-07-03, ICML Workshops 2021`
- [[Out-of-Distribution Detection]](#out-of-distribution-detection) [[Energy-Based Models]](#energy-based-models)
```
Well-written and quite interesting paper. A short paper, just 4 pages. They don't study the method from the "Energy-based Out-of-distribution Detection" paper as I had expected, but it was still a quite interesting read. The results in Section 4.2 seem interesting, especially for experiment 3, but I'm not sure that I properly understand everything.
```

##### [22-02-10] [paper175]
- Natural Posterior Network: Deep Bayesian Predictive Uncertainty for Exponential Family Distributions
 [[pdf]](https://openreview.net/forum?id=tV3N0DWMxCg) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Natural%20Posterior%20Network:%20Deep%20Bayesian%20Predictive%20Uncertainty%20for%20Exponential%20Family%20Distributions.pdf)
- *Bertrand Charpentier, Oliver Borchert, Daniel Zügner, Simon Geisler, Stephan Günnemann*
- `2021-09-29, ICLR 2022`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Out-of-Distribution Detection]](#out-of-distribution-detection)
```
Interesting and well-written paper. I didn't quite understand all the details, I'll have to read a couple of related/background papers to be able to properly appreciate and evaluate the proposed method. I definitely feel like I would like to read up on this family of methods. Extensive experimental evaluation, and the results seem promising overall.
```

##### [22-02-09] [paper174]
- Energy-based Out-of-distribution Detection
 [[pdf]](https://arxiv.org/abs/2010.03759) [[code]](https://github.com/wetliu/energy_ood) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Energy-based%20Out-of-distribution%20Detection.pdf)
- *Weitang Liu, Xiaoyun Wang, John D. Owens, Yixuan Li*
- `2020-10-08, NeurIPS 2020`
- [[Out-of-Distribution Detection]](#out-of-distribution-detection) [[Energy-Based Models]](#energy-based-models)
```
Interesting and well-written paper. The proposed method is quite clearly explained and makes intuitive sense (at least if you're familiar with EBMs). Compared to using the softmax score, the performance does seem to improve consistently. Seems like fine-tuning on an "auxiliary outlier dataset" is required to get really good performance though, which you can't really assume to have access to in real-world problems, I suppose?
```

##### [22-02-09] [paper173]
- VOS: Learning What You Don't Know by Virtual Outlier Synthesis
 [[pdf]](https://arxiv.org/abs/2202.01197) [[code]](https://github.com/deeplearning-wisc/vos) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/VOS:%20Learning%20What%20You%20Don't%20Know%20by%20Virtual%20Outlier%20Synthesis.pdf)
- *Xuefeng Du, Zhaoning Wang, Mu Cai, Yixuan Li*
- `2022-02-02, ICLR 2022`
- [[Out-of-Distribution Detection]](#out-of-distribution-detection)
```
Interesting and quite well-written paper. I did find it somewhat difficult to understand certain parts though, they could perhaps be explained more clearly. The results seem quite impressive (they do consistently outperform all baselines), but I find it interesting that the "Gaussian noise" baseline in Table 2 performs that well? I should probably have read "Energy-based Out-of-distribution Detection" before reading this paper.
```

#### Papers Read in 2021:

##### [21-12-16] [paper172]
- Efficiently Modeling Long Sequences with Structured State Spaces
 [[pdf]](https://arxiv.org/abs/2111.00396) [[code]](https://github.com/HazyResearch/state-spaces) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Efficiently%20Modeling%20Long%20Sequences%20with%20Structured%20State%20Spaces.pdf)
- *Albert Gu, Karan Goel, Christopher Ré*
- `2021-10-31, ICLR 2022`
- [[Sequence Modeling]](#sequence-modeling)
```
Very interesting and quite well-written paper. Kind of neat/fun to see state-space models being used. The experimental results seem very impressive!? I didn't fully understand everything in Section 3. I had to read Section 3.4 a couple of times to understand how the parameterization actually works in practice (you have H state-space models, one for each feature dimension, so that you can map a sequence of feature vectors to another sequence of feature vectors) (and you can then also have multiple such layers of state-space models, mapping sequence --> sequence --> sequence --> ....).
```

##### [21-12-09] [paper171]
- Periodic Activation Functions Induce Stationarity
 [[pdf]](https://arxiv.org/abs/2110.13572) [[code]](https://github.com/AaltoML/PeriodicBNN) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Periodic%20Activation%20Functions%20Induce%20Stationarity.pdf)
- *Lassi Meronen, Martin Trapp, Arno Solin*
- `2021-10-26, NeurIPS 2021`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Out-of-Distribution Detection]](#out-of-distribution-detection)
```
Quite interesting and well-written paper. Quite a heavy read, probably need to be rather familiar with GPs to properly understand/appreciate everything. Definitely check Appendix D, it gives a better understanding of how the proposed method is applied in practice. I'm not quite sure how strong/impressive the experimental results actually are. Also seems like the method could be a bit inconvenient to implement/use?
```

##### [21-12-03] [paper170]
- Reliable and Trustworthy Machine Learning for Health Using Dataset Shift Detection
 [[pdf]](https://arxiv.org/abs/2110.14019) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Reliable%20and%20Trustworthy%20Machine%20Learning%20for%20Health%20Using%20Dataset%20Shift%20Detection.pdf)
- *Chunjong Park, Anas Awadalla, Tadayoshi Kohno, Shwetak Patel*
- `2021-10-26, NeurIPS 2021`
- [[Out-of-Distribution Detection]](#out-of-distribution-detection) [[Medical ML]](#medical-ml)
```
Interesting and very well-written paper. Gives a good overview of the field and contains a lot of seemingly useful references. The evaluation is very comprehensive. The user study is quite neat.
```

##### [21-12-02] [paper169]
- An Information-theoretic Approach to Distribution Shifts
 [[pdf]](https://arxiv.org/abs/2106.03783) [[code]](https://github.com/mfederici/dsit) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/An%20Information-theoretic%20Approach%20to%20Distribution%20Shifts.pdf)
- *Marco Federici, Ryota Tomioka, Patrick Forré*
- `2021-06-07, NeurIPS 2021`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning)
```
Quite well-written paper overall that seemed interesting, but I found it very difficult to properly understand everything. Thus, I can't really tell how interesting/significant their analysis actually is.
```

##### [21-11-25] [paper168]
- On the Importance of Gradients for Detecting Distributional Shifts in the Wild
 [[pdf]](https://arxiv.org/abs/2110.00218) [[code]](https://github.com/deeplearning-wisc/gradnorm_ood) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20the%20Importance%20of%20Gradients%20for%20Detecting%20Distributional%20Shifts%20in%20the%20Wild.pdf)
- *Rui Huang, Andrew Geng, Yixuan Li*
- `2021-10-01, NeurIPS 2021`
- [[Out-of-Distribution Detection]](#out-of-distribution-detection)
```
Quite interesting and well-written paper. The experimental results do seem promising. However, I don't quite get why the proposed method intuitively makes sense, why is it better to only use the parameters of the final network layer?
```

##### [21-11-18] [paper167]
- Masked Autoencoders Are Scalable Vision Learners
 [[pdf]](https://arxiv.org/abs/2111.06377) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners.pdf)
- *Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick*
- `2021-11-11`
```
Interesting and well-written paper. The proposed method is simple and makes a lot of intuitive sense, which is rather satisfying. After page 4, there's mostly just detailed ablations and results.
```

##### [21-11-11] [paper166]
- Transferring Inductive Biases through Knowledge Distillation
 [[pdf]](https://arxiv.org/abs/2006.00555) [[code]](https://github.com/samiraabnar/Reflect) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Transferring%20Inductive%20Biases%20through%20Knowledge%20Distillation.pdf)
- *Samira Abnar, Mostafa Dehghani, Willem Zuidema*
- `2020-05-31`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning)
```
Quite well-written and somewhat interesting paper. I'm not very familiar with this area. I didn't spend too much time trying to properly evaluate the significance of the findings.
```

##### [21-10-28] [paper165]
- Deep Classifiers with Label Noise Modeling and Distance Awareness
 [[pdf]](https://arxiv.org/abs/2110.02609#) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Classifiers%20with%20Label%20Noise%20Modeling%20and%20Distance%20Awareness.pdf)
- *Vincent Fortuin, Mark Collier, Florian Wenzel, James Allingham, Jeremiah Liu, Dustin Tran, Balaji Lakshminarayanan, Jesse Berent, Rodolphe Jenatton, Effrosyni Kokiopoulou*
- `2021-10-06`
- [[Uncertainty Estimation]](#uncertainty-estimation)
```
Quite interesting and well-written paper. I find the distance-awareness property more interesting than modelling of input/class-dependent label noise, so the proposed method (HetSNGP) is perhaps not overly interesting compared to the SNGP baseline.
```

##### [21-10-21] [paper164]
- Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets
 [[pdf]](https://mathai-iclr.github.io/papers/papers/MATHAI_29_paper.pdf) [[code]](https://github.com/openai/grok) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Grokking:%20Generalization%20Beyond%20Overfitting%20On%20Small%20Algorithmic%20Datasets.pdf)
- *Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra*
- `2021-05, ICLR Workshops 2021`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning)
```
Somewhat interesting paper. The phenomena observed in Figure 1, that validation accuracy suddenly increases long after almost perfect fitting of the training data has been achieved is quite interesting. I didn't quite understand the datasets they use (binary operation tables).
```

##### [21-10-14] [paper163]
- Learning to Simulate Complex Physics with Graph Networks
 [[pdf]](https://arxiv.org/abs/2002.09405) [[code]](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20to%20Simulate%20Complex%20Physics%20with%20Graph%20Networks.pdf)
- *Alvaro Sanchez-Gonzalez, Jonathan Godwin, Tobias Pfaff, Rex Ying, Jure Leskovec, Peter W. Battaglia*
- `2020-02-21, ICML 2020`
```
Quite well-written and somewhat interesting paper. Cool application and a bunch of neat videos. This is not really my area, so I didn't spend too much time/energy trying to fully understand everything.
```

##### [21-10-12] [paper162]
- Neural Unsigned Distance Fields for Implicit Function Learning
 [[pdf]](https://arxiv.org/abs/2010.13938) [[code]](https://github.com/jchibane/ndf/) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Unsigned%20Distance%20Fields%20for%20Implicit%20Function%20Learning.pdf)
- *Julian Chibane, Aymen Mir, Gerard Pons-Moll*
- `2020-10-26, NeurIPS 2020`
- [[Implicit Neural Representations]](#implicit-neural-representations)
```
Interesting and very well-written paper, I really enjoyed reading it! The paper also gives a good understanding of neural implicit representations in general.
```

##### [21-10-08] [paper161]
- Probabilistic 3D Human Shape and Pose Estimation from Multiple Unconstrained Images in the Wild
 [[pdf]](https://arxiv.org/abs/2103.10978) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Probabilistic%203D%20Human%20Shape%20and%20Pose%20Estimation%20from%20Multiple%20Unconstrained%20Images%20in%20the%20Wild.pdf)
- *Akash Sengupta, Ignas Budvytis, Roberto Cipolla*
- `2021-03-19, CVPR 2021`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)
```
Well-written and quite interesting paper. I read it mainly as background for "Hierarchical Kinematic Probability Distributions for 3D Human Shape and Pose Estimation from Images in the Wild" which is written by exactly the same authors. In this paper, they predict a single Gaussian distribution for the pose (instead of hierarchical matrix-Fisher distributions). Also, they mainly focus on the body shape. They also use silhouettes + 2D keypoint heatmaps as input (instead of edge-filters + 2D keypoint heatmaps).
```

##### [21-10-08] [paper160]
- Synthetic Training for Accurate 3D Human Pose and Shape Estimation in the Wild
 [[pdf]](https://arxiv.org/abs/2009.10013) [[code]](https://github.com/akashsengupta1997/STRAPS-3DHumanShapePose) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Synthetic%20Training%20for%20Accurate%203D%20Human%20Pose%20and%20Shape%20Estimation%20in%20the%20Wild.pdf)
- *Akash Sengupta, Ignas Budvytis, Roberto Cipolla*
- `2020-09-21, BMVC 2020`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)
```
Well-written and farily interesting paper. I read it mainly as background for "Hierarchical Kinematic Probability Distributions for 3D Human Shape and Pose Estimation from Images in the Wild" which is written by exactly the same authors. In this paper, they just use direct regression. They also use silhouettes + 2D keypoint heatmaps as input (instead of edge-filters + 2D keypoint heatmaps).
```

##### [21-10-07] [paper159]
- Learning Motion Priors for 4D Human Body Capture in 3D Scenes
 [[pdf]](https://arxiv.org/abs/2108.10399) [[code]](https://github.com/sanweiliti/LEMO) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20Motion%20Priors%20for%204D%20Human%20Body%20Capture%20in%203D%20Scenes.pdf)
- *Siwei Zhang, Yan Zhang, Federica Bogo, Marc Pollefeys, Siyu Tang*
- `2021-08-23, ICCV 2021`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)
```
Well-written and quite interesting paper. I didn't fully understand everything though, and it feels like I probably don't know this specific setting/problem well enough to fully appreciate the paper. 
```

##### [21-10-07] [paper158]
- Hierarchical Kinematic Probability Distributions for 3D Human Shape and Pose Estimation from Images in the Wild
 [[pdf]](https://arxiv.org/abs/2110.00990) [[code]](https://github.com/akashsengupta1997/HierarchicalProbabilistic3DHuman) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Hierarchical%20Kinematic%20Probability%20Distributions%20for%203D%20Human%20Shape%20and%20Pose%20Estimation%20from%20Images%20in%20the%20Wild.pdf)
- *Akash Sengupta, Ignas Budvytis, Roberto Cipolla*
- `2021-10-03, ICCV 2021`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)
```
Well-written and very interesting paper, I enjoyed reading it. The hierarchical distribution prediction approach makes sense and consistently outperforms the independent baseline. Using matrix-Fisher distributions makes sense. The synthetic training framework and the input representation of edge-filters + 2D keypoint heatmaps are both interesting.
```

##### [21-10-06] [paper157]
- SMD-Nets: Stereo Mixture Density Networks
 [[pdf]](https://arxiv.org/abs/2104.03866) [[code]](https://github.com/fabiotosi92/SMD-Nets) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/SMD-Nets:%20Stereo%20Mixture%20Density%20Networks.pdf)
- *Fabio Tosi, Yiyi Liao, Carolin Schmitt, Andreas Geiger*
- `2021-04-08, CVPR 2021`
- [[Uncertainty Estimation]](#uncertainty-estimation)
```
Well-written and interesting paper. Quite easy to read and follow, the method is clearly explained and makes intuitive sense.
```

##### [21-10-04] [paper156]
- We are More than Our Joints: Predicting how 3D Bodies Move
 [[pdf]](https://arxiv.org/abs/2012.00619) [[code]](https://github.com/yz-cnsdqz/MOJO-release) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/We%20are%20More%20than%20Our%20Joints:%20Predicting%20how%203D%20Bodies%20Move.pdf)
- *Yan Zhang, Michael J. Black, Siyu Tang*
- `2020-12-01, CVPR 2021`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)
```
Well-written and fairly interesting paper. The marker-based representation, instead of using skeleton joints, makes sense. The recursive projection scheme also makes sense, but seems very slow (2.27 sec/frame)? I didn't quite get all the details for their DCT representation of the latent space.
```

##### [21-10-03] [paper155]
- imGHUM: Implicit Generative Models of 3D Human Shape and Articulated Pose
 [[pdf]](https://arxiv.org/abs/2108.10842) [[code]](https://github.com/google-research/google-research/tree/master/imghum) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/imGHUM:%20Implicit%20Generative%20Models%20of%203D%20Human%20Shape%20and%20Articulated%20Pose.pdf)
- *Thiemo Alldieck, Hongyi Xu, Cristian Sminchisescu*
- `2021-08-24, ICCV 2021`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation) [[Implicit Neural Representations]](#implicit-neural-representations)
```
Interesting and very well-written paper, I really enjoyed reading it. Interesting combination of implicit representations and 3D human modelling. The "inclusive human modelling" application is neat and important.
```

##### [21-10-03] [paper154]
- DI-Fusion: Online Implicit 3D Reconstruction with Deep Priors
 [[pdf]](https://arxiv.org/abs/2012.05551) [[code]](https://github.com/huangjh-pub/di-fusion) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/DI-Fusion:%20Online%20Implicit%203D%20Reconstruction%20with%20Deep%20Priors.pdf)
- *Jiahui Huang, Shi-Sheng Huang, Haoxuan Song, Shi-Min Hu*
- `2020-12-10, CVPR 2021`
- [[Implicit Neural Representations]](#implicit-neural-representations)
```
Well-written and interesting paper, I enjoyed reading it. Neat application of implicit representations. The paper also gives a quite good overview of online 3D reconstruction in general.
```

##### [21-10-02] [paper153]
- Contextually Plausible and Diverse 3D Human Motion Prediction
 [[pdf]](https://arxiv.org/abs/1912.08521) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Contextually%20Plausible%20and%20Diverse%203D%20Human%20Motion%20Prediction.pdf)
- *Sadegh Aliakbarian, Fatemeh Sadat Saleh, Lars Petersson, Stephen Gould, Mathieu Salzmann*
- `2019-12-18, ICCV 2021`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)
```
Well-written and quite interesting paper. The main idea, using a learned conditional prior p(z|c) instead of just p(z), makes sense and was shown beneficial also in "HuMoR: 3D Human Motion Model for Robust Pose Estimation". I'm however somewhat confused by their specific implementation in Section 4, doesn't seem like a standard cVAE implementation?
```

##### [21-10-01] [paper152]
- Local Implicit Grid Representations for 3D Scenes
 [[pdf]](https://arxiv.org/abs/2003.08981) [[code]](https://github.com/tensorflow/graphics/tree/master/tensorflow_graphics/projects/local_implicit_grid) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Local%20Implicit%20Grid%20Representations%20for%203D%20Scenes.pdf)
- *Chiyu Max Jiang, Avneesh Sud, Ameesh Makadia, Jingwei Huang, Matthias Nießner, Thomas Funkhouser*
- `2020-03-19, CVPR 2020`
- [[Implicit Neural Representations]](#implicit-neural-representations)
```
Well-written and quite interesting paper. Interesting application, being able to reconstruct full 3D scenes from sparse point clouds. I didn't fully understand everything, as I don't have a particularly strong graphics background.
```

##### [21-09-29] [paper151]
- Information Dropout: Learning Optimal Representations Through Noisy Computation
 [[pdf]](https://arxiv.org/abs/1611.01353) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Information%20Dropout:%20Learning%20Optimal%20Representations%20Through%20Noisy%20Computation.pdf)
- *Alessandro Achille, Stefano Soatto*
- `2016-11-04`
```
Well-written and somewhat interesting paper overall. I'm not overly familiar with the topics of the paper, and didn't fully understand everything. Some results and insights seem quite interesting/neat, but I'm not sure exactly what the main takeaways should be, or how significant they actually are.
```

##### [21-09-24] [paper150]
- Encoder-decoder with Multi-level Attention for 3D Human Shape and Pose Estimation
 [[pdf]](https://arxiv.org/abs/2109.02303) [[code]](https://github.com/ziniuwan/maed) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Encoder-decoder%20with%20Multi-level%20Attention%20for%203D%20Human%20Shape%20and%20Pose%20Estimation.pdf)
- *Ziniu Wan, Zhengjia Li, Maoqing Tian, Jianbo Liu, Shuai Yi, Hongsheng Li*
- `2021-09-06, ICCV 2021`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)
```
Well-written and fairly interesting paper. Quite a lot of details on the attention architecture, which I personally don't find overly interesting. The experimental results are quite impressive, but I would like to see a comparison in terms of computational cost at test-time. It sounds like their method is rather slow.
```

##### [21-09-23] [paper149]
- Physics-based Human Motion Estimation and Synthesis from Videos
 [[pdf]](https://arxiv.org/abs/2109.09913) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Physics-based%20Human%20Motion%20Estimation%20and%20Synthesis%20from%20Videos.pdf)
- *Kevin Xie, Tingwu Wang, Umar Iqbal, Yunrong Guo, Sanja Fidler, Florian Shkurti*
- `2021-09-21, ICCV 2021`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)
```
Well-written and quite interesting paper. The general idea, refining frame-by-frame pose estimates via physical constraints, intuitively makes a lot of sense. I did however find it quite difficult to understand all the details in Section 3.
```

##### [21-09-21] [paper148]
- Hierarchical VAEs Know What They Don't Know
 [[pdf]](https://arxiv.org/abs/2102.08248) [[code]](https://github.com/JakobHavtorn/hvae-oodd) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Hierarchical%20VAEs%20Know%20What%20They%20Don't%20Know.pdf)
- *Jakob D. Havtorn, Jes Frellsen, Søren Hauberg, Lars Maaløe*
- `2021-02-16, ICML 2021`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[VAEs]](#vaes)
```
Very well-written and quite interesting paper, I enjoyed reading it. Everything is quite well-explained, it's relatively easy to follow. The paper provides a good overview of the out-of-distribution detection problem and current methods.
```

##### [21-09-17] [paper147]
- Human Pose Regression with Residual Log-likelihood Estimation
 [[pdf]](https://arxiv.org/abs/2107.11291) [[code]](https://github.com/Jeff-sjtu/res-loglikelihood-regression) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Human%20Pose%20Regression%20with%20Residual%20Log-likelihood%20Estimation.pdf)
- *Jiefeng Li, Siyuan Bian, Ailing Zeng, Can Wang, Bo Pang, Wentao Liu, Cewu Lu*
- `2021-07-23, ICCV 2021`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)
```
Quite interesting paper, but also quite strange/confusing. I don't think the proposed method is explained particularly well, at least I found it quite difficult to properly understand what they actually are doing.

In the end it seems like they are learning a global loss function that is very similar to doing probabilistic regression with a Gauss/Laplace model of p(y|x) (with learned mean and variance)? See Figure 4 in the Appendix.

And while it's true that their performance is much better than for direct regression with an L2/L1 loss (see e.g. Table 1), they only compare with Gauss/Laplace probabilistic regression once (Table 7) and in that case the Laplace model is actually quite competitive?
```

##### [21-09-15] [paper146]
- NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
 [[pdf]](https://arxiv.org/abs/2003.08934) [[code]](https://github.com/bmild/nerf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/NeRF:%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields%20for%20View%20Synthesis.pdf)
- *Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng*
- `2020-03-19, ECCV 2020`
- [[Implicit Neural Representations]](#implicit-neural-representations)
```
Extremely well-written and interesting paper. I really enjoyed reading it, and I would recommend anyone interested in computer vision to read it as well.

All parts of the proposed method are clearly explained and relatively easy to understand, including the volume rendering techniques which I was unfamiliar with.
```

##### [21-09-08] [paper145]
- Revisiting the Calibration of Modern Neural Networks
 [[pdf]](https://arxiv.org/abs/2106.07998) [[code]](https://github.com/google-research/robustness_metrics/tree/master/robustness_metrics/projects/revisiting_calibration) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Revisiting%20the%20Calibration%20of%20Modern%20Neural%20Networks.pdf)
- *Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby, Dustin Tran, Mario Lucic*
- `2021-06-15, NeurIPS 2021`
- [[Uncertainty Estimation]](#uncertainty-estimation)
```
Well-written paper. Everything is quite clearly explained and easy to understand. Quite enjoyable to read overall. 

Thorough experimental evaluation. Quite interesting findings.
```

##### [21-09-02] [paper144]
- Differentiable Particle Filtering via Entropy-Regularized Optimal Transport
 [[pdf]](https://arxiv.org/abs/2102.07850) [[code]](https://github.com/JTT94/filterflow) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Differentiable%20Particle%20Filtering%20via%20Entropy-Regularized%20Optimal%20Transport.pdf)
- *Adrien Corenflos, James Thornton, George Deligiannidis, Arnaud Doucet*
- `2021-02-15, ICML 2021`

##### [21-09-02] [paper143]
- Character Controllers Using Motion VAEs
 [[pdf]](https://arxiv.org/abs/2103.14274) [[code]](https://github.com/electronicarts/character-motion-vaes) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Character%20Controllers%20Using%20Motion%20VAEs.pdf)
- *Hung Yu Ling, Fabio Zinno, George Cheng, Michiel van de Panne*
- `2021-03-26, SIGGRAPH 2020`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)

##### [21-08-27] [paper142]
- DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation
 [[pdf]](https://arxiv.org/abs/1901.05103) [[code]](https://github.com/facebookresearch/DeepSDF) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/DeepSDF:%20Learning%20Continuous%20Signed%20Distance%20Functions%20for%20Shape%20Representation.pdf)
- *Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, Steven Lovegrove*
- `2019-01-16, CVPR 2019`
- [[Implicit Neural Representations]](#implicit-neural-representations)

##### [21-06-19] [paper141]
- Generating Multiple Hypotheses for 3D Human Pose Estimation with Mixture Density Network
 [[pdf]](https://arxiv.org/abs/1904.05547) [[code]](https://github.com/chaneyddtt/Generating-Multiple-Hypotheses-for-3D-Human-Pose-Estimation-with-Mixture-Density-Network) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Generating%20Multiple%20Hypotheses%20for%203D%20Human%20Pose%20Estimation%20with%20Mixture%20Density%20Network.pdf)
- *Chen Li, Gim Hee Lee*
- `2019-04-11, CVPR 2019`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)

##### [21-06-19] [paper140]
- Expressive Body Capture: 3D Hands, Face, and Body from a Single Image
 [[pdf]](https://arxiv.org/abs/1904.05866) [[code]](https://github.com/vchoutas/smplify-x) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Expressive%20Body%20Capture:%203D%20Hands%2C%20Face%2C%20and%20Body%20from%20a%20Single%20Image.pdf)
- *Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, Michael J. Black*
- `2019-04-11, CVPR 2019`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)
```
Very well-written and quite interesting paper. Gives a good understanding of the SMPL model and the SMPLify method.
```

##### [21-06-18] [paper139]
- Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image
 [[pdf]](https://arxiv.org/abs/1607.08128) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Keep%20it%20SMPL:%20Automatic%20Estimation%20of%203D%20Human%20Pose%20and%20Shape%20from%20a%20Single%20Image.pdf)
- *Federica Bogo, Angjoo Kanazawa, Christoph Lassner, Peter Gehler, Javier Romero, Michael J. Black*
- `2016-07-27, ECCV 2016`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)

##### [21-06-18] [paper138]
- Beyond Static Features for Temporally Consistent 3D Human Pose and Shape from a Video
 [[pdf]](https://arxiv.org/abs/2011.08627) [[code]](https://github.com/hongsukchoi/TCMR_RELEASE) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Beyond%20Static%20Features%20for%20Temporally%20Consistent%203D%20Human%20Pose%20and%20Shape%20from%20a%20Video.pdf)
- *Hongsuk Choi, Gyeongsik Moon, Ju Yong Chang, Kyoung Mu Lee*
- `2020-11-17, CVPR 2021`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)

##### [21-06-17] [paper137]
- Exemplar Fine-Tuning for 3D Human Model Fitting Towards In-the-Wild 3D Human Pose Estimation
 [[pdf]](https://arxiv.org/abs/2004.03686) [[code]](https://github.com/facebookresearch/eft) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Exemplar%20Fine-Tuning%20for%203D%20Human%20Model%20Fitting%20Towards%20In-the-Wild%203D%20Human%20Pose%20Estimation.pdf)
- *Hanbyul Joo, Natalia Neverova, Andrea Vedaldi*
- `2020-04-07`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)

##### [21-06-17] [paper136]
- Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop
 [[pdf]](https://arxiv.org/abs/1909.12828) [[code]](https://github.com/nkolot/SPIN) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20to%20Reconstruct%203D%20Human%20Pose%20and%20Shape%20via%20Model-fitting%20in%20the%20Loop.pdf)
- *Nikos Kolotouros, Georgios Pavlakos, Michael J. Black, Kostas Daniilidis*
- `2019-09-27, ICCV 2019`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)

##### [21-06-16] [paper135]
- A simple yet effective baseline for 3d human pose estimation
 [[pdf]](https://arxiv.org/abs/1705.03098) [[code]](https://github.com/una-dinosauria/3d-pose-baseline) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20simple%20yet%20effective%20baseline%20for%203d%20human%20pose%20estimation.pdf)
- *Julieta Martinez, Rayat Hossain, Javier Romero, James J. Little*
- `2017-05-08, ICCV 2017`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)

##### [21-06-16] [paper134]
- Estimating Egocentric 3D Human Pose in Global Space
 [[pdf]](https://arxiv.org/abs/2104.13454) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Estimating%20Egocentric%203D%20Human%20Pose%20in%20Global%20Space.pdf)
- *Jian Wang, Lingjie Liu, Weipeng Xu, Kripasindhu Sarkar, Christian Theobalt*
- `2021-04-27, ICCV 2021`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)

##### [21-06-15] [paper133]
- End-to-end Recovery of Human Shape and Pose
 [[pdf]](https://arxiv.org/abs/1712.06584) [[code]](https://github.com/akanazawa/hmr) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/End-to-end%20Recovery%20of%20Human%20Shape%20and%20Pose.pdf)
- *Angjoo Kanazawa, Michael J. Black, David W. Jacobs, Jitendra Malik*
- `2017-12-18, CVPR 2018`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)

##### [21-06-14] [paper132]
- 3D Multi-bodies: Fitting Sets of Plausible 3D Human Models to Ambiguous Image Data
 [[pdf]](https://arxiv.org/abs/2011.00980) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/3D%20Multi-bodies:%20Fitting%20Sets%20of%20Plausible%203D%20Human%20Models%20to%20Ambiguous%20Image%20Data.pdf)
- *Benjamin Biggs, Sébastien Ehrhadt, Hanbyul Joo, Benjamin Graham, Andrea Vedaldi, David Novotny*
- `2020-11-02, NeurIPS 2020`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)

##### [21-06-04] [paper131]
- HuMoR: 3D Human Motion Model for Robust Pose Estimation
 [[pdf]](https://arxiv.org/abs/2105.04668) [[code]](https://geometry.stanford.edu/projects/humor/) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/HuMoR:%203D%20Human%20Motion%20Model%20for%20Robust%20Pose%20Estimation.pdf)
- *Davis Rempe, Tolga Birdal, Aaron Hertzmann, Jimei Yang, Srinath Sridhar, Leonidas J. Guibas*
- `2021-05-10, ICCV 2021`
- [[3D Human Pose Estimation]](#3d-human-pose-estimation)

##### [21-05-07] [paper130]
- PixelTransformer: Sample Conditioned Signal Generation
 [[pdf]](https://arxiv.org/abs/2103.15813) [[code]](https://github.com/shubhtuls/PixelTransformer) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/PixelTransformer:%20Sample%20Conditioned%20Signal%20Generation.pdf)
- *Shubham Tulsiani, Abhinav Gupta*
- `2021-03-29, ICML 2021`
- [[Neural Processes]](#neural-processes) [[Transformers]](#transformers)

##### [21-04-29] [paper129]
- Stiff Neural Ordinary Differential Equations
 [[pdf]](https://arxiv.org/abs/2103.15341) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Stiff%20Neural%20Ordinary%20Differential%20Equations.pdf)
- *Suyong Kim, Weiqi Ji, Sili Deng, Yingbo Ma, Christopher Rackauckas*
- `2021-03-29`
- [[Neural ODEs]](#neural-odes)

##### [21-04-16] [paper128]
- Learning Mesh-Based Simulation with Graph Networks
 [[pdf]](https://arxiv.org/abs/2010.03409) [[code]](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20Mesh-Based%20Simulation%20with%20Graph%20Networks.pdf)
- *Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, Peter W. Battaglia*
- `2020-10-07, ICLR 2021`

##### [21-04-09] [paper127]
- Q-Learning in enormous action spaces via amortized approximate maximization
 [[pdf]](https://arxiv.org/abs/2001.08116) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Q-Learning%20in%20enormous%20action%20spaces%20via%20amortized%20approximate%20maximization.pdf)
- *Tom Van de Wiele, David Warde-Farley, Andriy Mnih, Volodymyr Mnih*
- `2020-01-22`
- [[Reinforcement Learning]](#reinforcement-learning)

##### [21-04-01] [paper126]
- Loss Surface Simplexes for Mode Connecting Volumes and Fast Ensembling
 [[pdf]](https://arxiv.org/abs/2102.13042) [[code]](https://github.com/g-benton/loss-surface-simplexes) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Loss%20Surface%20Simplexes%20for%20Mode%20Connecting%20Volumes%20and%20Fast%20Ensembling.pdf)
- *Gregory W. Benton, Wesley J. Maddox, Sanae Lotfi, Andrew Gordon Wilson*
- `2021-02-25, ICML 2021`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Ensembling]](#ensembling)

##### [21-03-26] [paper125]
- Your GAN is Secretly an Energy-based Model and You Should use Discriminator Driven Latent Sampling
 [[pdf]](https://arxiv.org/abs/2003.06060) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Your%20GAN%20is%20Secretly%20an%20Energy-based%20Model%20and%20You%20Should%20Use%20Discriminator%20Driven%20Latent%20Sampling.pdf)
- *Tong Che, Ruixiang Zhang, Jascha Sohl-Dickstein, Hugo Larochelle, Liam Paull, Yuan Cao, Yoshua Bengio*
- `2020-03-12, NeurIPS 2020`
- [[Energy-Based Models]](#energy-based-models)

##### [21-03-19] [paper124]
- Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability
 [[pdf]](https://arxiv.org/abs/2103.00065) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Gradient%20Descent%20on%20Neural%20Networks%20Typically%20Occurs%20at%20the%20Edge%20of%20Stability.pdf)
- *Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, Ameet Talwalkar*
- `2021-02-26, ICLR 2021`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning)

##### [21-03-12] [paper123]
- Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
 [[pdf]](https://arxiv.org/abs/2006.09882) [[code]](https://github.com/facebookresearch/swav) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Unsupervised%20Learning%20of%20Visual%20Features%20by%20Contrasting%20Cluster%20Assignments.pdf)
- *Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, Armand Joulin*
- `2020-06-17, NeurIPS 2020`

##### [21-03-04] [paper122]
- Infinitely Deep Bayesian Neural Networks with Stochastic Differential Equations
 [[pdf]](https://arxiv.org/abs/2102.06559) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Infinitely%20Deep%20Bayesian%20Neural%20Networks%20with%20Stochastic%20Differential%20Equations.pdf)
- *Winnie Xu, Ricky T.Q. Chen, Xuechen Li, David Duvenaud*
- `2021-02-12`
- [[Neural ODEs]](#neural-odes) [[Uncertainty Estimation]](#uncertainty-estimation)

##### [21-02-26] [paper121]
- Neural Relational Inference for Interacting Systems
 [[pdf]](https://arxiv.org/abs/1802.04687) [[code]](https://github.com/ethanfetaya/NRI) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Relational%20Inference%20for%20Interacting%20Systems.pdf)
- *Thomas Kipf, Ethan Fetaya, Kuan-Chieh Wang, Max Welling, Richard Zemel*
- `2018-02-13, ICML 2018`

##### [21-02-19] [paper120]
- Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision
 [[pdf]](https://arxiv.org/abs/2102.05918) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Scaling%20Up%20Visual%20and%20Vision-Language%20Representation%20Learning%20With%20Noisy%20Text%20Supervision.pdf)
- *Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig*
- `2021-02-11, ICML 2021`

##### [21-02-12] [paper119]
- On the Origin of Implicit Regularization in Stochastic Gradient Descent
 [[pdf]](https://arxiv.org/abs/2101.12176) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20the%20Origin%20of%20Implicit%20Regularization%20in%20Stochastic%20Gradient%20Descent.pdf)
- *Samuel L. Smith, Benoit Dherin, David G. T. Barrett, Soham De*
- `2021-01-28, ICLR 2021`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning)

##### [21-02-05] [paper118]
- Meta Pseudo Labels
 [[pdf]](https://arxiv.org/abs/2003.10580) [[code]](https://github.com/google-research/google-research/tree/master/meta_pseudo_labels) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Meta%20Pseudo%20Labels.pdf)
- *Hieu Pham, Zihang Dai, Qizhe Xie, Minh-Thang Luong, Quoc V. Le*
- `2020-03-23, CVPR 2021`

##### [21-01-29] [paper117]
- No MCMC for Me: Amortized Sampling for Fast and Stable Training of Energy-Based Models
 [[pdf]](https://arxiv.org/abs/2010.04230) [[code]](https://github.com/wgrathwohl/VERA) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/No%20MCMC%20for%20me:%20Amortized%20sampling%20for%20fast%20and%20stable%20training%20of%20energy-based%20models.pdf)
- *Will Grathwohl, Jacob Kelly, Milad Hashemi, Mohammad Norouzi, Kevin Swersky, David Duvenaud*
- `2020-10-08, ICLR 2021`
- [[Energy-Based Models]](#energy-based-models)

##### [21-01-22] [paper116]
- Getting a CLUE: A Method for Explaining Uncertainty Estimates
 [[pdf]](https://arxiv.org/abs/2006.06848) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Getting%20a%20CLUE:%20A%20Method%20for%20Explaining%20Uncertainty%20Estimates.pdf)
- *Javier Antorán, Umang Bhatt, Tameem Adel, Adrian Weller, José Miguel Hernández-Lobato*
- `2020-06-11, ICLR 2021`
- [[Uncertainty Estimation]](#uncertainty-estimation)

##### [21-01-15] [paper115]
- Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
 [[pdf]](https://arxiv.org/abs/2006.16236) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Transformers%20are%20RNNs:%20Fast%20Autoregressive%20Transformers%20with%20Linear%20Attention.pdf)
- *Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret*
- `2020-06-29, ICML 2020`
- [[Transformers]](#transformers)

#### Papers Read in 2020:

##### [20-12-18] [paper114]
- Score-Based Generative Modeling through Stochastic Differential Equations
 [[pdf]](https://arxiv.org/abs/2011.13456) [[code]](https://github.com/yang-song/score_sde) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Score-Based%20Generative%20Modeling%20through%20Stochastic%20Differential%20Equations.pdf)
- *Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole*
- `2020-11-26, ICLR 2021`
- [[Neural ODEs]](#neural-odes)

##### [20-12-14] [paper113]
- Dissecting Neural ODEs
 [[pdf]](https://arxiv.org/abs/2002.08071) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Dissecting%20Neural%20ODEs.pdf)
- *Stefano Massaroli, Michael Poli, Jinkyoo Park, Atsushi Yamashita, Hajime Asama*
- `2020-02-19, NeurIPS 2020`
- [[Neural ODEs]](#neural-odes)

##### [20-11-27] [paper112]
- Rethinking Attention with Performers
 [[pdf]](https://arxiv.org/abs/2009.14794) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Rethinking%20Attention%20with%20Performers.pdf)
- *Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller*
- `2020-10-30, ICLR 2021`
- [[Transformers]](#transformers)

##### [20-11-23] [paper111]
- Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images
 [[pdf]](https://arxiv.org/abs/2011.10650) [[code]](https://github.com/openai/vdvae) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Very%20Deep%20VAEs%20Generalize%20Autoregressive%20Models%20and%20Can%20Outperform%20Them%20on%20Images.pdf)
- *Rewon Child*
- `2020-11-20, ICLR 2021`
- [[VAEs]](#vaes)

##### [20-11-13] [paper110]
- VAEBM: A Symbiosis between Variational Autoencoders and Energy-based Models
 [[pdf]](https://arxiv.org/abs/2010.00654) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/VAEBM:%20A%20Symbiosis%20between%20Variational%20Autoencoders%20and%20Energy-based%20Models.pdf)
- *Zhisheng Xiao, Karsten Kreis, Jan Kautz, Arash Vahdat*
- `2020-10-01, ICLR 2021`
- [[Energy-Based Models]](#energy-based-models) [[VAEs]](#vaes)

##### [20-11-06] [paper109]
- Approximate Inference Turns Deep Networks into Gaussian Processes
 [[pdf]](https://arxiv.org/abs/1906.01930) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Approximate%20Inference%20Turns%20Deep%20Networks%20into%20Gaussian%20Processes.pdf)
- *Mohammad Emtiyaz Khan, Alexander Immer, Ehsan Abedi, Maciej Korzepa*
- `2019-06-05, NeurIPS 2019`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning)

##### [20-10-16] [paper108]
- Implicit Gradient Regularization [[pdf]](https://arxiv.org/abs/2009.11162) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Implicit%20Gradient%20Regularization.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Implicit%20Gradient%20Regularization.md)
- *David G.T. Barrett, Benoit Dherin*
- `2020-09-23`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning)

##### [20-10-09] [paper107]
- Satellite Conjunction Analysis and the False Confidence Theorem [[pdf]](https://royalsocietypublishing.org/doi/pdf/10.1098/rspa.2018.0565) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Satellite%20conjunction%20analysis%20and%20the%20false%20confidence%20theorem.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Satellite%20Conjunction%20Analysis%20and%20the%20False%20Confidence%20Theorem.md)
- *Michael Scott Balch, Ryan Martin, Scott Ferson*
- `2018-03-21`

##### [20-09-24] [paper106]
- Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness [[pdf]](https://arxiv.org/abs/2006.10108) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Simple%20and%20Principled%20Uncertainty%20Estimation%20with%20Deterministic%20Deep%20Learning%20via%20Distance%20Awareness.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Simple%20and%20Principled%20Uncertainty%20Estimation%20with%20Deterministic%20Deep%20Learning%20via%20Distance%20Awareness.md)
- *Jeremiah Zhe Liu, Zi Lin, Shreyas Padhy, Dustin Tran, Tania Bedrax-Weiss, Balaji Lakshminarayanan*
- `2020-06-17, NeurIPS 2020`
- [[Uncertainty Estimation]](#uncertainty-estimation)

##### [20-09-21] [paper105]
- Uncertainty Estimation Using a Single Deep Deterministic Neural Network [[pdf]](https://arxiv.org/abs/2003.02037) [[code]](https://github.com/y0ast/deterministic-uncertainty-quantification) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20Estimation%20Using%20a%20Single%20Deep%20Deterministic%20Neural%20Network.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Uncertainty%20Estimation%20Using%20a%20Single%20Deep%20Deterministic%20Neural%20Network.md)
- *Joost van Amersfoort, Lewis Smith, Yee Whye Teh, Yarin Gal*
- `2020-03-04, ICML 2020`
- [[Uncertainty Estimation]](#uncertainty-estimation)

##### [20-09-11] [paper104]
- Gated Linear Networks [[pdf]](https://arxiv.org/abs/1910.01526) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Gated%20Linear%20Networks.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Gated%20Linear%20Networks.md)
- *Joel Veness, Tor Lattimore, David Budden, Avishkar Bhoopchand, Christopher Mattern, Agnieszka Grabska-Barwinska, Eren Sezener, Jianan Wang, Peter Toth, Simon Schmitt, Marcus Hutter*
- `2020-06-11`

##### [20-09-04] [paper103]
- Denoising Diffusion Probabilistic Models [[pdf]](https://arxiv.org/abs/2006.11239) [[code]](https://github.com/hojonathanho/diffusion) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Denoising%20Diffusion%20Probabilistic%20Models.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Denoising%20Diffusion%20Probabilistic%20Models.md)
- *Jonathan Ho, Ajay Jain, Pieter Abbeel*
- `20-06-19`
- [[Energy-Based Models]](#energy-based-models)

##### [20-06-18] [paper102]
- Joint Training of Variational Auto-Encoder and Latent Energy-Based Model [[pdf]](https://arxiv.org/abs/2006.06059) [[code]](https://hthth0801.github.io/jointLearning/) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Joint%20Training%20of%20Variational%20Auto-Encoder%20and%20Latent%20Energy-Based%20Model.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Joint%20Training%20of%20Variational%20Auto-Encoder%20and%20Latent%20Energy-Based%20Model.md)
- *Tian Han, Erik Nijkamp, Linqi Zhou, Bo Pang, Song-Chun Zhu, Ying Nian Wu*
- `2020-06-10, CVPR 2020`
- [[VAEs]](#vaes) [[Energy-Based Models]](#energy-based-models)

##### [20-06-12] [paper101]
- End-to-End Object Detection with Transformers [[pdf]](https://arxiv.org/abs/2005.12872) [[code]](https://github.com/facebookresearch/detr) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/End-to-End%20Object%20Detection%20with%20Transformers.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/End-to-End%20Object%20Detection%20with%20Transformers.md)
- *Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko*
- `2020-05-26, ECCV 2020`
- [[Object Detection]](#object-detection)

##### [20-06-05] [paper100]
- Efficient and Scalable Bayesian Neural Nets with Rank-1 Factors [[pdf]](https://arxiv.org/abs/2005.07186) [[code]](https://github.com/google/edward2) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Efficient%20and%20Scalable%20Bayesian%20Neural%20Nets%20with%20Rank-1%20Factors.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Efficient%20and%20Scalable%20Bayesian%20Neural%20Nets%20with%20Rank-1%20Factors.md)
- *Michael W. Dusenberry, Ghassen Jerfel, Yeming Wen, Yi-an Ma, Jasper Snoek, Katherine Heller, Balaji Lakshminarayanan, Dustin Tran*
- `2020-05-14, ICML 2020`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Variational Inference]](#variational-inference)

##### [20-05-27] [paper99]
- BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning [[pdf]](https://arxiv.org/abs/2002.06715) [[code]](https://github.com/google/edward2) [[video]](https://iclr.cc/virtual_2020/poster_Sklf1yrYDr.html) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/BatchEnsemble:%20An%20Alternative%20Approach%20to%20Efficient%20Ensemble%20and%20Lifelong%20Learning.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/BatchEnsemble:%20An%20Alternative%20Approach%20to%20Efficient%20Ensemble%20and%20Lifelong%20Learning.md)
- *Yeming Wen, Dustin Tran, Jimmy Ba*
- `2020-02-17, ICLR 2020`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Ensembling]](#ensembling)

##### [20-05-10] [paper98]
- Stable Neural Flows [[pdf]](https://arxiv.org/abs/2003.08063) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Stable%20Neural%20Flows%20.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Stable%20Neural%20Flows.md)
- *Stefano Massaroli, Michael Poli, Michelangelo Bin, Jinkyoo Park, Atsushi Yamashita, Hajime Asama*
- `2020-03-18`

##### [20-04-17] [paper97]
- How Good is the Bayes Posterior in Deep Neural Networks Really? [[pdf]](https://arxiv.org/abs/2002.02405) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/How%20Good%20is%20the%20Bayes%20Posterior%20in%20Deep%20Neural%20Networks%20Really%3F.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/How%20Good%20is%20the%20Bayes%20Posterior%20in%20Deep%20Neural%20Networks%20Really%3F.md)
- *Florian Wenzel, Kevin Roth, Bastiaan S. Veeling, Jakub Świątkowski, Linh Tran, Stephan Mandt, Jasper Snoek, Tim Salimans, Rodolphe Jenatton, Sebastian Nowozin*
- `2020-02-06`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Stochastic Gradient MCMC]](#stochastic-gradient-mcmc)

##### [20-04-09] [paper96]
- Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration [[pdf]](https://arxiv.org/abs/1910.12656) [[code]](https://github.com/dirichletcal/experiments_neurips) [[poster]](https://dirichletcal.github.io/documents/neurips2019/poster.pdf) [[slides]](https://dirichletcal.github.io/documents/neurips2019/slides.pdf) [[video]](https://dirichletcal.github.io/documents/neurips2019/video/Meelis_Ettekanne.mp4) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Beyond%20temperature%20scaling:%20Obtaining%20well-calibrated%20multiclass%20probabilities%20with%20Dirichlet%20calibration.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Beyond%20temperature%20scaling:%20Obtaining%20well-calibrated%20multiclass%20probabilities%20with%20Dirichlet%20calibration.md)
- *Meelis Kull, Miquel Perello-Nieto, Markus Kängsepp, Telmo Silva Filho, Hao Song, Peter Flach*
- `2019-10-28, NeurIPS 2019`
- [[Uncertainty Estimation]](#uncertainty-estimation)

##### [20-04-03] [paper95]
- Normalizing Flows: An Introduction and Review of Current Methods [[pdf]](https://arxiv.org/abs/1908.09257) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Normalizing%20Flows:%20An%20Introduction%20and%20Review%20of%20Current%20Methods.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Normalizing%20Flows:%20An%20Introduction%20and%20Review%20of%20Current%20Methods.md)
- *Ivan Kobyzev, Simon Prince, Marcus A. Brubaker*
- `2019-08-25`
- [[Normalizing Flows]](#normalizing-flows)

##### [20-03-27] [paper94]
- Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning [[pdf]](https://arxiv.org/abs/2002.06470) [[code]](https://github.com/bayesgroup/pytorch-ensembles) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Pitfalls%20of%20In-Domain%20Uncertainty%20Estimation%20and%20Ensembling%20in%20Deep%20Learning.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Pitfalls%20of%20In-Domain%20Uncertainty%20Estimation%20and%20Ensembling%20in%20Deep%20Learning.md)
- *Arsenii Ashukha, Alexander Lyzhov, Dmitry Molchanov, Dmitry Vetrov*
- `2020-02-15, ICLR 2020`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Ensembling]](#ensembling) [[Stochastic Gradient MCMC]](#stochastic-gradient-mcmc)

##### [20-03-26] [paper93]
- Conservative Uncertainty Estimation By Fitting Prior Networks [[pdf]](https://openreview.net/forum?id=BJlahxHYDS) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Conservative%20Uncertainty%20Estimation%20By%20Fitting%20Prior%20Networks.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Conservative%20Uncertainty%20Estimation%20By%20Fitting%20Prior%20Networks.md)
- *Kamil Ciosek, Vincent Fortuin, Ryota Tomioka, Katja Hofmann, Richard Turner*
- `2019-10-25, ICLR 2020`
- [[Uncertainty Estimation]](#uncertainty-estimation) 

##### [20-03-09] [paper92]
- Batch Normalization Biases Deep Residual Networks Towards Shallow Paths [[pdf]](https://arxiv.org/abs/2002.10444) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Batch%20Normalization%20Biases%20Deep%20Residual%20Networks%20Towards%20Shallow%20Path.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Batch%20Normalization%20Biases%20Deep%20Residual%20Networks%20Towards%20Shallow%20Paths.md)
- *Soham De, Samuel L. Smith*
- `2020-02-24`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning) 

##### [20-02-28] [paper91]
- Bayesian Deep Learning and a Probabilistic Perspective of Generalization [[pdf]](https://arxiv.org/abs/2002.08791) [[code]](https://github.com/izmailovpavel/understandingbdl) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Bayesian%20Deep%20Learning%20and%20a%20Probabilistic%20Perspective%20of%20Generalization.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Bayesian%20Deep%20Learning%20and%20a%20Probabilistic%20Perspective%20of%20Generalization.md)
- *Andrew Gordon Wilson, Pavel Izmailov*
- `2020-02-20`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Ensembling]](#ensembling) 

##### [20-02-21] [paper90]
- Convolutional Conditional Neural Processes [[pdf]](https://arxiv.org/abs/1910.13556) [[code]](https://github.com/cambridge-mlg/convcnp) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Convolutional%20Conditional%20Neural%20Processes.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Convolutional%20Conditional%20Neural%20Processes.md)
- *Jonathan Gordon, Wessel P. Bruinsma, Andrew Y. K. Foong, James Requeima, Yann Dubois, Richard E. Turner*
- `2019-10-29, ICLR 2020`
- [[Neural Processes]](#neural-processes)

##### [20-02-18] [paper89]
- Probabilistic 3D Multi-Object Tracking for Autonomous Driving [[pdf]](https://arxiv.org/abs/2001.05673) [[code]](https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Probabilistic%203D%20Multi-Object%20Tracking%20for%20Autonomous%20Driving.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Probabilistic%203D%20Multi-Object%20Tracking%20for%20Autonomous%20Driving.md)
- *Hsu-kuang Chiu, Antonio Prioletti, Jie Li, Jeannette Bohg*
- `2020-01-16`
- [[3D Multi-Object Tracking]](#3d-multi-object-tracking)

##### [20-02-15] [paper88]
- A Baseline for 3D Multi-Object Tracking [[pdf]](https://arxiv.org/abs/1907.03961) [[code]](https://github.com/xinshuoweng/AB3DMOT) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Baseline%20for%203D%20Multi-Object%20Tracking.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/A%20Baseline%20for%203D%20Multi-Object%20Tracking.md)
- *Xinshuo Weng, Kris Kitani*
- `2019-07-09`
- [[3D Multi-Object Tracking]](#3d-multi-object-tracking)

##### [20-02-14] [paper87]
- A Contrastive Divergence for Combining Variational Inference and MCMC [[pdf]](https://arxiv.org/abs/1905.04062) [[code]](https://github.com/franrruiz/vcd_divergence) [[slides]](https://franrruiz.github.io/contents/group_talks/EMS-Jul2019.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Contrastive%20Divergence%20for%20Combining%20Variational%20Inference%20and%20MCMC.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/A%20Contrastive%20Divergence%20for%20Combining%20Variational%20Inference%20and%20MCMC.md)
- *Francisco J. R. Ruiz, Michalis K. Titsias*
- `2019-05-10, ICML 2019`
- [[VAEs]](#vaes)

##### [20-02-13] [paper86]
- Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning [[pdf]](https://arxiv.org/abs/1710.07283) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Decomposition%20of%20Uncertainty%20in%20Bayesian%20Deep%20Learning%20for%20Efficient%20and%20Risk-sensitive%20Learning.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Decomposition%20of%20Uncertainty%20in%20Bayesian%20Deep%20Learning%20for%20Efficient%20and%20Risk-sensitive%20Learning.md)
- *Stefan Depeweg, José Miguel Hernández-Lobato, Finale Doshi-Velez, Steffen Udluft*
- `2017-10-19, ICML 2018`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Reinforcement Learning]](#reinforcement-learning)

##### [20-02-08] [paper85]
- Uncertainty Decomposition in Bayesian Neural Networks with Latent Variables [[pdf]](https://arxiv.org/abs/1706.08495) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20Decomposition%20in%20Bayesian%20Neural%20Networks%20with%20Latent%20Variables.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Uncertainty%20Decomposition%20in%20Bayesian%20Neural%20Networks%20with%20Latent%20Variables.md)
- *Stefan Depeweg, José Miguel Hernández-Lobato, Finale Doshi-Velez, Steffen Udluft*
- `2017-06-26`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Reinforcement Learning]](#reinforcement-learning)

##### [20-01-31] [paper84]
- Modelling heterogeneous distributions with an Uncountable Mixture of Asymmetric Laplacians [[pdf]](https://arxiv.org/abs/1910.12288) [[code]](https://github.com/BBVA/UMAL) [[video]](https://vimeo.com/369179175) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Modelling%20heterogeneous%20distributions%20with%20an%20Uncountable%20Mixture%20of%20Asymmetric%20Laplacians.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Modelling%20heterogeneous%20distributions%20with%20an%20Uncountable%20Mixture%20of%20Asymmetric%20Laplacians.md)
- *Axel Brando, Jose A. Rodríguez-Serrano, Jordi Vitrià, Alberto Rubio*
- `2019-10-27, NeurIPS 2019`
- [[Uncertainty Estimation]](#uncertainty-estimation)

##### [20-01-24] [paper83]
- A Primal-Dual link between GANs and Autoencoders [[pdf]](http://papers.nips.cc/paper/8333-a-primal-dual-link-between-gans-and-autoencoders) [[poster]](https://drive.google.com/file/d/1ifPldBOeuSa2Iwh3ESVGmRJQKzs9XgPv/view) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Primal-Dual%20link%20between%20GANs%20and%20Autoencoders.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/A%20Primal-Dual%20link%20between%20GANs%20and%20Autoencoders.md)
- *Hisham Husain, Richard Nock, Robert C. Williamson*
- `2019-04-26, NeurIPS 2019`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning)

##### [20-01-20] [paper82]
- A Connection Between Score Matching and Denoising Autoencoders [[pdf]](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport_1358.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Connection%20Between%20Score%20Matching%20and%20Denoising%20Autoencoders.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/A%20Connection%20Between%20Score%20Matching%20and%20Denoising%20Autoencoders.md)
- *Pascal Vincent*
- `2010-12`
- [[Energy-Based Models]](#energy-based-models)

##### [20-01-17] [paper81]
- Multiplicative Interactions and Where to Find Them [[pdf]](https://openreview.net/forum?id=rylnK6VtDH) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Multiplicative%20Interactions%20and%20Where%20to%20Find%20Them.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Multiplicative%20Interactions%20and%20Where%20to%20Find%20Them.md)
- *Siddhant M. Jayakumar, Jacob Menick, Wojciech M. Czarnecki, Jonathan Schwarz, Jack Rae, Simon Osindero, Yee Whye Teh, Tim Harley, Razvan Pascanu*
- `2019-09-25, ICLR 2020`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning) [[Sequence Modeling]](#sequence-modeling)

##### [20-01-16] [paper80]
- Estimation of Non-Normalized Statistical Models by Score Matching [[pdf]](http://www.jmlr.org/papers/v6/hyvarinen05a.html) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Estimation%20of%20Non-Normalized%20Statistical%20Models%20by%20Score%20Matching.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Estimation%20of%20Non-Normalized%20Statistical%20Models%20by%20Score%20Matching.md)
- *Aapo Hyvärinen*
- `2004-11, JMLR 6`
- [[Energy-Based Models]](#energy-based-models)

##### [20-01-15] [paper79]
- Generative Modeling by Estimating Gradients of the Data Distribution [[pdf]](https://arxiv.org/abs/1907.05600) [[code]](https://github.com/ermongroup/ncsn) [[poster]](https://yang-song.github.io/papers/NeurIPS2019/ncsn-poster.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Generative%20Modeling%20by%20Estimating%20Gradients%20of%20the%20Data%20Distribution.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Generative%20Modeling%20by%20Estimating%20Gradients%20of%20the%20Data%20Distribution.md)
- *Yang Song, Stefano Ermon*
- `2019-07-12, NeurIPS 2019`
- [[Energy-Based Models]](#energy-based-models)

##### [20-01-14] [paper78]
- Noise-contrastive estimation: A new estimation principle for unnormalized statistical models [[pdf]](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Noise-contrastive%20estimation:%20A%20new%20estimation%20principle%20for%20unnormalized%20statistical%20models.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Noise-contrastive%20estimation:%20A%20new%20estimation%20principle%20for%20unnormalized%20statistical%20models.md)
- *Michael Gutmann, Aapo Hyvärinen*
- `2009, AISTATS 2010`
- [[Energy-Based Models]](#energy-based-models)

##### [20-01-10] [paper77]
- Z-Forcing: Training Stochastic Recurrent Networks [[pdf]](https://arxiv.org/abs/1711.05411) [[code]](https://github.com/sordonia/zforcing) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Z-Forcing:%20Training%20Stochastic%20Recurrent%20Networks.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Z-Forcing:%20Training%20Stochastic%20Recurrent%20Networks.md)
- *Anirudh Goyal, Alessandro Sordoni, Marc-Alexandre Côté, Nan Rosemary Ke, Yoshua Bengio*
- `2017-11-15, NeurIPS 2017`
- [[VAEs]](#vaes) [[Sequence Modeling]](#sequence-modeling)

##### [20-01-08] [paper76]
- Practical Deep Learning with Bayesian Principles [[pdf]](https://arxiv.org/abs/1906.02506) [[code]](https://github.com/team-approx-bayes/dl-with-bayes) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Practical%20Deep%20Learning%20with%20Bayesian%20Principles.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Practical%20Deep%20Learning%20with%20Bayesian%20Principles.md)
- *Kazuki Osawa, Siddharth Swaroop, Anirudh Jain, Runa Eschenhagen, Richard E. Turner, Rio Yokota, Mohammad Emtiyaz Khan*
- `2019-06-06, NeurIPS 2019`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Variational Inference]](#variational-inference)

##### [20-01-06] [paper75]
- Maximum Entropy Generators for Energy-Based Models [[pdf]](https://arxiv.org/abs/1901.08508) [[code]](https://github.com/ritheshkumar95/energy_based_generative_models) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Maximum%20Entropy%20Generators%20for%20Energy-Based%20Models.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Maximum%20Entropy%20Generators%20for%20Energy-Based%20Models.md)
- *Rithesh Kumar, Sherjil Ozair, Anirudh Goyal, Aaron Courville, Yoshua Bengio*
- `2019-01-24`
- [[Energy-Based Models]](#energy-based-models)

#### Papers Read in 2019:

##### [19-12-22] [paper74]
- Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One [[pdf]](https://arxiv.org/abs/1912.03263) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Your%20Classifier%20is%20Secretly%20an%20Energy%20Based%20Model%20and%20You%20Should%20Treat%20it%20Like%20One.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Your%20Classifier%20is%20Secretly%20an%20Energy%20Based%20Model%20and%20You%20Should%20Treat%20it%20Like%20One.md)
- *Will Grathwohl, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, Kevin Swersky*
- `2019-12-06, ICLR 2020`
- [[Energy-Based Models]](#energy-based-models)

##### [19-12-20] [paper73]
- Noise Contrastive Estimation and Negative Sampling for Conditional Models: Consistency and Statistical Efficiency [[pdf]](https://arxiv.org/abs/1809.01812) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Noise%20Contrastive%20Estimation%20and%20Negative%20Sampling%20for%20Conditional%20Models:%20Consistency%20and%20Statistical%20Efficiency.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Noise%20Contrastive%20Estimation%20and%20Negative%20Sampling%20for%20Conditional%20Models:%20Consistency%20and%20Statistical%20Efficiency.md)
- *Zhuang Ma, Michael Collins*
- `2018-09-06, EMNLP 2018`
- [[Energy-Based Models]](#energy-based-models)

##### [19-12-20] [paper72]
- Flow Contrastive Estimation of Energy-Based Models [[pdf]](https://arxiv.org/abs/1912.00589) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Flow%20Contrastive%20Estimation%20of%20Energy-Based%20Models.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Flow%20Contrastive%20Estimation%20of%20Energy-Based%20Models.md)
- *Ruiqi Gao, Erik Nijkamp, Diederik P. Kingma, Zhen Xu, Andrew M. Dai, Ying Nian Wu*
- `2019-12-02, CVPR 2020`
- [[Energy-Based Models]](#energy-based-models) [[Normalizing Flows]](#normalizing-flows)

##### [19-12-19] [paper71]
- On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models [[pdf]](https://arxiv.org/abs/1903.12370) [[code]](https://github.com/point0bar1/ebm-anatomy)  [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20the%20Anatomy%20of%20MCMC-Based%20Maximum%20Likelihood%20Learning%20of%20Energy-Based%20Models.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/On%20the%20Anatomy%20of%20MCMC-Based%20Maximum%20Likelihood%20Learning%20of%20Energy-Based%20Models.md)
- *Erik Nijkamp, Mitch Hill, Tian Han, Song-Chun Zhu, Ying Nian Wu*
- `2019-04-29, AAAI 2020`
- [[Energy-Based Models]](#energy-based-models)

##### [19-12-15] [paper70]
- Implicit Generation and Generalization in Energy-Based Models [[pdf]](https://arxiv.org/abs/1903.08689) [[code]](https://github.com/openai/ebm_code_release) [[blog]](https://openai.com/blog/energy-based-models/) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Implicit%20Generation%20and%20Generalization%20in%20Energy-Based%20Models.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Implicit%20Generation%20and%20Generalization%20in%20Energy-Based%20Models.md)
- *Yilun Du, Igor Mordatch*
- `2019-04-20, NeurIPS 2019`
- [[Energy-Based Models]](#energy-based-models)

##### [19-12-14] [paper69]
- Learning Non-Convergent Non-Persistent Short-Run MCMC Toward Energy-Based Model [[pdf]](https://arxiv.org/abs/1904.09770) [[poster]](https://neurips.cc/Conferences/2019/Schedule?showEvent=13661) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20Non-Convergent%20Non-Persistent%20Short-Run%20MCMC%20Toward%20Energy-Based%20Model.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Learning%20Non-Convergent%20Non-Persistent%20Short-Run%20MCMC%20Toward%20Energy-Based%20Model.md)
- *Erik Nijkamp, Mitch Hill, Song-Chun Zhu, Ying Nian Wu*
- `2019-04-22, NeurIPS 2019`
- [[Energy-Based Models]](#energy-based-models)

##### [19-12-13] [paper68]
- A Tutorial on Energy-Based Learning [[pdf]](http://yann.lecun.com/exdb/publis/orig/lecun-06.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Tutorial%20on%20Energy-Based%20Learning.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/A%20Tutorial%20on%20Energy-Based%20Learning.md)
- *Yann LeCun, Sumit Chopra, Raia Hadsell, Marc Aurelio Ranzato, Fu Jie Huang*
- `2006-08-19`
- [[Energy-Based Models]](#energy-based-models)

##### [19-11-29] [paper67]
- Dream to Control: Learning Behaviors by Latent Imagination [[pdf]](https://openreview.net/forum?id=S1lOTC4tDS) [[webpage]](https://dreamrl.github.io/) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Dream%20to%20Control%20Learning%20Behaviors%20by%20Latent%20Imagination.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Dream%20to%20Control:%20Learning%20Behaviors%20by%20Latent%20Imagination.md)
- *Anonymous*
- `2019-09`

##### [19-11-26] [paper66]
- Deep Latent Variable Models for Sequential Data [[pdf]](https://backend.orbit.dtu.dk/ws/portalfiles/portal/160548008/phd475_Fraccaro_M.pdf)  [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Latent%20Variable%20Models%20for%20Sequential%20Data.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Deep%20Latent%20Variable%20Models%20for%20Sequential%20Data.md)
- *Marco Fraccaro*
- `2018-04-13, PhD Thesis`

##### [19-11-22] [paper65]
- Learning Latent Dynamics for Planning from Pixels [[pdf]](https://arxiv.org/abs/1811.04551) [[code]](https://github.com/google-research/planet) [[blog]](https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20Latent%20Dynamics%20for%20Planning%20from%20Pixels.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Learning%20Latent%20Dynamics%20for%20Planning%20from%20Pixels.md)
- *Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson*
- `2018-11-12, ICML2019`

##### [19-10-28] [paper64]
- Learning nonlinear state-space models using deep autoencoders [[pdf]](http://cse.lab.imtlucca.it/~bemporad/publications/papers/cdc18-autoencoders.pdf)  [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20nonlinear%20state-space%20models%20using%20deep%20autoencoders.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Learning%20nonlinear%20state-space%20models%20using%20deep%20autoencoders.md)
- *Daniele Masti, Alberto Bemporad*
- `2018, CDC2018`

##### [19-10-18] [paper63]
- Improving Variational Inference with Inverse Autoregressive Flow [[pdf]](https://arxiv.org/abs/1606.04934) [[code]](https://github.com/openai/iaf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Improving%20Variational%20Inference%20with%20Inverse%20Autoregressive%20Flow.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Improving%20Variational%20Inference%20with%20Inverse%20Autoregressive%20Flow.md)
- *Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling*
- `2016-06-15, NeurIPS2016`

##### [19-10-11] [paper62]
- Variational Inference with Normalizing Flows [[pdf]](https://arxiv.org/abs/1505.05770) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Variational%20Inference%20with%20Normalizing%20Flows.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Variational%20Inference%20with%20Normalizing%20Flows.md)
- *Danilo Jimenez Rezende, Shakir Mohamed*
- `2015-05-21, ICML2015`

##### [19-10-04] [paper61]
- Trellis Networks for Sequence Modeling [[pdf]](https://arxiv.org/abs/1810.06682) [[code]](https://github.com/locuslab/trellisnet) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Trellis%20Networks%20for%20Sequence%20Modeling.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Trellis%20Networks%20for%20Sequence%20Modeling.md)
- *Shaojie Bai, J. Zico Kolter, Vladlen Koltun*
- `2018-10-15, ICLR2019`

##### [19-07-11] [paper60]
- Part-A^2 Net: 3D Part-Aware and Aggregation Neural Network for Object Detection from Point Cloud [[pdf]](https://arxiv.org/abs/1907.03670) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Part-A%5E2%20Net:%203D%20Part-Aware%20and%20Aggregation%20Neural%20Network%20for%20Object%20Detection%20from%20Point%20Cloud.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Part-A%5E2%20Net:%203D%20Part-Aware%20and%20Aggregation%20Neural%20Network%20for%20Object%20Detection%20from%20Point%20Cloud.md)
- *Shaoshuai Shi, Zhe Wang, Xiaogang Wang, Hongsheng Li*
- `2019-07-08`

##### [19-07-10] [paper59]
- PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud [[pdf]](https://arxiv.org/abs/1812.04244) [[code]](https://github.com/sshaoshuai/PointRCNN) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/PointRCNN:%203D%20Object%20Proposal%20Generation%20and%20Detection%20from%20Point%20Cloud.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/PointRCNN:%203D%20Object%20Proposal%20Generation%20and%20Detection%20from%20Point%20Cloud.md)
- *Shaoshuai Shi, Xiaogang Wang, Hongsheng Li*
- `2018-12-11, CVPR2019`

##### [19-07-03] [paper58]
- Objects as Points [[pdf]](https://arxiv.org/abs/1904.07850) [[code]](https://github.com/xingyizhou/CenterNet) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Objects%20as%20Points.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Objects%20as%20Points.md)
- *Xingyi Zhou, Dequan Wang, Philipp Krähenbühl*
- `2019-04-16`

##### [19-06-12] [paper57]
- ATOM: Accurate Tracking by Overlap Maximization [[pdf]](https://arxiv.org/abs/1811.07628) [[code]](https://github.com/visionml/pytracking) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/ATOM:%20Accurate%20Tracking%20by%20Overlap%20Maximization.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/ATOM:%20Accurate%20Tracking%20by%20Overlap%20Maximization.md)
- *Martin Danelljan, Goutam Bhat, Fahad Shahbaz Khan, Michael Felsberg*
- `2018-11-19, CVPR2019`

##### [19-06-12] [paper56]
- Acquisition of Localization Confidence for Accurate Object Detection [[pdf]](https://arxiv.org/abs/1807.11590) [[code]](https://github.com/vacancy/PreciseRoIPooling) [[oral presentation]](https://youtu.be/SNCsXOFr_Ug) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Acquisition%20of%20Localization%20Confidence%20for%20Accurate%20Object%20Detection.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Acquisition%20of%20Localization%20Confidence%20for%20Accurate%20Object%20Detection.md)
- *Borui Jiang, Ruixuan Luo, Jiayuan Mao, Tete Xiao, Yuning Jiang*
- `2018-07-30, ECCV2018`

##### [19-06-05] [paper55]
- LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving [[pdf]](https://arxiv.org/abs/1903.08701) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/LaserNet:%20An%20Efficient%20Probabilistic%203D%20Object%20Detector%20for%20Autonomous%20Driving.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/LaserNet:%20An%20Efficient%20Probabilistic%203D%20Object%20Detector%20for%20Autonomous%20Driving.md)
- *Gregory P. Meyer, Ankit Laddha, Eric Kee, Carlos Vallespi-Gonzalez, Carl K. Wellington*
- `2019-03-20, CVPR2019`

##### [19-05-29] [paper54]
- Attention Is All You Need [[pdf]](https://arxiv.org/abs/1706.03762) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Attention%20Is%20All%20You%20Need.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Attention%20Is%20All%20You%20Need.md)
- *Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin*
- `2017-06-12, NeurIPS2017`

##### [19-04-05] [paper53]
- Stochastic Gradient Descent as Approximate Bayesian Inference [[pdf]](https://arxiv.org/abs/1704.04289) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Stochastic%20Gradient%20Descent%20as%20Approximate%20Bayesian%20Inference.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Stochastic%20Gradient%20Descent%20as%20Approximate%20Bayesian%20Inference.md)
- *Stephan Mandt, Matthew D. Hoffman, David M. Blei*
- `2017-04-13, Journal of Machine Learning Research 18 (2017)`

##### [19-03-29] [paper52]
- Generating High Fidelity Images with Subscale Pixel Networks and Multidimensional Upscaling [[pdf]](https://openreview.net/forum?id=HylzTiC5Km) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/GENERATING%20HIGH%20FIDELITY%20IMAGES%20WITH%20SUBSCALE%20PIXEL%20NETWORKS%20AND%20MULTIDIMENSIONAL%20UPSCALING.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Generating%20High%20Fidelity%20Images%20with%20Subscale%20Pixel%20Networks%20and%20Multidimensional%20Upscaling.md)
- *Jacob Menick, Nal Kalchbrenner*
- `2018-12-04, ICLR2019`

##### [19-03-15] [paper51]
- A recurrent neural network without chaos [[pdf]](https://arxiv.org/abs/1612.06212) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20recurrent%20neural%20network%20without%20chaos.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/A%20recurrent%20neural%20network%20without%20chaos.md)
- *Thomas Laurent, James von Brecht*
- `2016-12-19, ICLR2017`

##### [19-03-11] [paper50]
- Auto-Encoding Variational Bayes [[pdf]](https://arxiv.org/abs/1312.6114) [[pdf with comments (TODO!)]]() [[comments (TOOD!)]](https://github.com/fregu856/papers/blob/master/summaries/Auto-Encoding%20Variational%20Bayes.md)
- *Diederik P Kingma, Max Welling*
- `2014-05-01, ICLR2014`

##### [19-03-04] [paper49]
- Coupled Variational Bayes via Optimization Embedding [[pdf]](https://papers.nips.cc/paper/8177-coupled-variational-bayes-via-optimization-embedding.pdf) [[poster]](http://wyliu.com/papers/LiuNIPS18_CVB_poster.pdf) [[code]](https://github.com/Hanjun-Dai/cvb) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Coupled%20Variational%20Bayes%20via%20Optimization%20Embedding.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Coupled%20Variational%20Bayes%20via%20Optimization%20Embedding.md)
- *Bo Dai, Hanjun Dai, Niao He, Weiyang Liu, Zhen Liu, Jianshu Chen, Lin Xiao, Le Song*
- `NeurIPS2018`

##### [19-03-01] [paper48]
- Language Models are Unsupervised Multitask Learners [[pdf]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) [[blog post]](https://blog.openai.com/better-language-models/) [[code]](https://github.com/openai/gpt-2) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Language%20Models%20are%20Unsupervised%20Multitask%20Learners.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Language%20Models%20are%20Unsupervised%20Multitask%20Learners.md)
- *Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever*
- `2019-02-14`

##### [19-02-27] [paper47]
- Predictive Uncertainty Estimation via Prior Networks [[pdf]](https://arxiv.org/abs/1802.10501) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Predictive%20Uncertainty%20Estimation%20via%20Prior%20Networks.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Predictive%20Uncertainty%20Estimation%20via%20Prior%20Networks.md)
- *Andrey Malinin, Mark Gales*
- `2018-02-28, NeurIPS2018`

##### [19-02-25] [paper46]
- Evaluating model calibration in classification [[pdf]](https://arxiv.org/abs/1902.06977) [[code]](https://github.com/uu-sml/calibration) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Evaluating%20model%20calibration%20in%20classification.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Evaluating%20model%20calibration%20in%20classification.md)
- *Juozas Vaicenavicius, David Widmann, Carl Andersson, Fredrik Lindsten, Jacob Roll, Thomas B. Schön*
- `2019-02-19, AISTATS2019`

##### [19-02-22] [paper45]
- Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks [[pdf]](https://arxiv.org/abs/1901.08584) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Fine-Grained%20Analysis%20of%20Optimization%20and%20Generalization%20for%20Overparameterized%20Two-Layer%20Neural%20Networks.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Fine-Grained%20Analysis%20of%20Optimization%20and%20Generalization%20for%20Overparameterized%20Two-Layer%20Neural%20Networks.md)
- *Sanjeev Arora, Simon S. Du, Wei Hu, Zhiyuan Li, Ruosong Wang*
- `2019-01-24`

##### [19-02-17] [paper44]
- Visualizing the Loss Landscape of Neural Nets [[pdf]](https://arxiv.org/abs/1712.09913) [[code]](https://github.com/tomgoldstein/loss-landscape) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Visualizing%20the%20Loss%20Landscape%20of%20Neural%20Nets.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Visualizing%20the%20Loss%20Landscape%20of%20Neural%20Nets.md)
- *Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein*
- `2017-12-28, NeurIPS2018`

##### [19-02-14] [paper43]
-  A Simple Baseline for Bayesian Uncertainty in Deep Learning [[pdf]](https://arxiv.org/abs/1902.02476) [[code]](https://github.com/wjmaddox/swa_gaussian) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Simple%20Baseline%20for%20Bayesian%20Uncertainty%20in%20Deep%20Learning.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/A%20Simple%20Baseline%20for%20Bayesian%20Uncertainty%20in%20Deep%20Learning.md)
- *Wesley Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, Andrew Gordon Wilson*
- `2019-02-07`

##### [19-02-13] [paper42]
-  Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning [[pdf]](https://arxiv.org/abs/1902.03932) [[code]](https://github.com/ruqizhang/csgmcmc) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Cyclical%20Stochastic%20Gradient%20MCMC%20for%20Bayesian%20Deep%20Learning.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Cyclical%20Stochastic%20Gradient%20MCMC%20for%20Bayesian%20Deep%20Learning.md)
- *Ruqi Zhang, Chunyuan Li, Jianyi Zhang, Changyou Chen, Andrew Gordon Wilson*
- `2019-02-11`

##### [19-02-12] [paper41]
-  Bayesian Dark Knowledge [[pdf]](https://arxiv.org/abs/1506.04416) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Bayesian%20Dark%20Knowledge.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Bayesian%20Dark%20Knowledge.md)
- *Anoop Korattikara, Vivek Rathod, Kevin Murphy, Max Welling*
- `2015-06-07, NeurIPS2015`

##### [19-02-07] [paper40]
- Noisy Natural Gradient as Variational Inference [[pdf]](https://arxiv.org/abs/1712.02390) [[video]](https://youtu.be/bWItvHYqKl8) [[code]](https://github.com/pomonam/NoisyNaturalGradient) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Noisy%20Natural%20Gradient%20as%20Variational%20Inference.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Noisy%20Natural%20Gradient%20as%20Variational%20Inference.md)
- *Guodong Zhang, Shengyang Sun, David Duvenaud, Roger Grosse*
- `2017-12-06, ICML2018`

##### [19-02-06] [paper39]
- Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks [[pdf]](https://arxiv.org/abs/1502.05336) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Probabilistic%20Backpropagation%20for%20Scalable%20Learning%20of%20Bayesian%20Neural%20Networks.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Probabilistic%20Backpropagation%20for%20Scalable%20Learning%20of%20Bayesian%20Neural%20Networks.md)
- *José Miguel Hernández-Lobato, Ryan P. Adams*
- `2015-07-15, ICML2015`

##### [19-02-05] [paper38]
- Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models [[pdf]](https://arxiv.org/abs/1805.12114) [[poster]](https://kchua.github.io/misc/poster.pdf) [[video]](https://youtu.be/3d8ixUMSiL8) [[code]](https://github.com/kchua/handful-of-trials) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Reinforcement%20Learning%20in%20a%20Handful%20of%20Trials%20using%20Probabilistic%20Dynamics%20Models.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Deep%20Reinforcement%20Learning%20in%20a%20Handful%20of%20Trials%20using%20Probabilistic%20Dynamics%20Models.md)
- *Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine*
- `2018-05-30, NeurIPS2018`

##### [19-01-28] [paper37]
- Practical Variational Inference for Neural Networks [[pdf]](https://www.cs.toronto.edu/~graves/nips_2011.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Practical%20Variational%20Inference%20for%20Neural%20Networks.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Practical%20Variational%20Inference%20for%20Neural%20Networks.md)
- *Alex Graves*
- `NeurIPS2011`

##### [19-01-27] [paper36]
- Weight Uncertainty in Neural Networks [[pdf]](https://arxiv.org/abs/1505.05424) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Weight%20Uncertainty%20in%20Neural%20Networks.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Weight%20Uncertainty%20in%20Neural%20Networks.md)
- *Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra*
- `2015-05-20, ICML2015`

##### [19-01-26] [paper35]
- Learning Weight Uncertainty with Stochastic Gradient MCMC for Shape Classification [[pdf]](http://people.duke.edu/~cl319/doc/papers/dbnn_shape_cvpr.pdf)  [[poster]](https://zhegan27.github.io/Papers/dbnn_shape_poster.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20Weight%20Uncertainty%20with%20Stochastic%20Gradient%20MCMC%20for%20Shape%20Classification.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Learning%20Weight%20Uncertainty%20with%20Stochastic%20Gradient%20MCMC%20for%20Shape%20Classification.md)
- *Chunyuan Li, Andrew Stevens, Changyou Chen, Yunchen Pu, Zhe Gan, Lawrence Carin*
- `CVPR2016`

##### [19-01-25] [paper34]
- Meta-Learning For Stochastic Gradient MCMC [[pdf]](https://openreview.net/forum?id=HkeoOo09YX) [[code]](https://github.com/WenboGong/MetaSGMCMC) [[slides]](http://yingzhenli.net/home/pdf/uai_udl_meta_sampler.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Meta-Learning%20For%20Stochastic%20Gradient%20MCMC.pdf) [[summary (TODO!)]]()
- *Wenbo Gong, Yingzhen Li, José Miguel Hernández-Lobato*
- `2018-10-28, ICLR2019`

##### [19-01-25] [paper33]
-  A Complete Recipe for Stochastic Gradient MCMC [[pdf]](https://arxiv.org/abs/1506.04696) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Complete%20Recipe%20for%20Stochastic%20Gradient%20MCMC.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/A%20Complete%20Recipe%20for%20Stochastic%20Gradient%20MCMC.md)
- *Yi-An Ma, Tianqi Chen, Emily B. Fox*
- `2015-06-15, NeurIPS2015`

##### [19-01-24] [paper32]
- Tutorial: Introduction to Stochastic Gradient Markov Chain Monte Carlo Methods [[pdf]](https://cse.buffalo.edu/~changyou/PDF/sgmcmc_intro_without_video.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Tutorial:%20Introduction%20to%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo%20Methods.pdf)
- *Changyou Chen*
- `2016-08-10`

##### [19-01-24] [paper31]
- An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling [[pdf]](https://arxiv.org/abs/1803.01271) [[code]](https://github.com/locuslab/TCN) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/An%20Empirical%20Evaluation%20of%20Generic%20Convolutional%20and%20Recurrent%20Networks%20for%20Sequence%20Modeling.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/An%20Empirical%20Evaluation%20of%20Generic%20Convolutional%20and%20Recurrent%20Networks%20for%20Sequence%20Modeling.md)
- *Shaojie Bai, J. Zico Kolter, Vladlen Koltun*
- `2018-04-19`

##### [19-01-23] [paper30]
- Stochastic Gradient Hamiltonian Monte Carlo [[pdf]](https://arxiv.org/abs/1402.4102) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Stochastic%20Gradient%20Hamiltonian%20Monte%20Carlo.pdf) [[summary (TODO!)]]()
- *Tianqi Chen, Emily B. Fox, Carlos Guestrin*
- `2014-05-12, ICML2014`

##### [19-01-23] [paper29]
- Bayesian Learning via Stochastic Gradient Langevin Dynamics [[pdf]](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Bayesian%20Learning%20via%20Stochastic%20Gradient%20Langevin%20Dynamics.pdf) [[summary (TODO!)]]()
- *Max Welling, Yee Whye Teh*
- `ICML2011`

##### [19-01-17] [paper28]
- How Does Batch Normalization Help Optimization? [[pdf]](https://arxiv.org/abs/1805.11604) [[poster]](http://people.csail.mit.edu/tsipras/batchnorm_poster.pdf) [[video]](https://youtu.be/ZOabsYbmBRM) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/How%20Does%20Batch%20Normalization%20Help%20Optimization%3F.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/How%20Does%20Batch%20Normalization%20Help%20Optimization%3F.md)
- *Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, Aleksander Madry*
- `2018-10-27, NeurIPS2018`

##### [19-01-09] [paper27]
- Relaxed Softmax: Efficient Confidence Auto-Calibration for Safe Pedestrian Detection [[pdf]](https://openreview.net/forum?id=S1lG7aTnqQ) [[poster]](http://www.robots.ox.ac.uk/~vgg/publications/2018/Neumann18c/poster.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Relaxed%20Softmax:%20Efficient%20Confidence%20Auto-Calibration%20for%20Safe%20Pedestrian%20Detection.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Relaxed%20Softmax:%20Efficient%20Confidence%20Auto-Calibration%20for%20Safe%20Pedestrian%20Detection.md)
- *Lukas Neumann, Andrew Zisserman, Andrea Vedaldi*
- `2018-11-29, NeurIPS2018 Workshop`

#### Papers Read in 2018:

##### [18-12-12] [paper26]
- Neural Ordinary Differential Equations [[pdf]](https://arxiv.org/abs/1806.07366) [[code]](https://github.com/rtqichen/torchdiffeq) [[slides]](https://www.cs.toronto.edu/~duvenaud/talks/ode-talk-google.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Ordinary%20Differential%20Equations.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Neural%20Ordinary%20Differential%20Equations.md)
- *Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud*
- `2018-10-22, NeurIPS2018`
- [[Neural ODEs]](#neural-odes)

##### [18-12-06] [paper25]
- Evaluating Bayesian Deep Learning Methods for Semantic Segmentation [[pdf]](https://arxiv.org/abs/1811.12709) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Evaluating%20Bayesian%20Deep%20Learning%20Methods%20for%20Semantic%20Segmentation.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Evaluating%20Bayesian%20Deep%20Learning%20Methods%20for%20Semantic%20Segmentation.md)
- *Jishnu Mukhoti, Yarin Gal*
- `2018-11-30`

##### [18-12-05] [paper24]
- On Calibration of Modern Neural Networks [[pdf]](https://arxiv.org/abs/1706.04599) [[code]](https://github.com/gpleiss/temperature_scaling) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20Calibration%20of%20Modern%20Neural%20Networks.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/On%20Calibration%20of%20Modern%20Neural%20Networks.md)
- *Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger*
- `2017-08-03, ICML2017`

##### [18-11-29] [paper23]
-  Evidential Deep Learning to Quantify Classification Uncertainty [[pdf]](https://arxiv.org/abs/1806.01768) [[poster]](https://muratsensoy.github.io/NIPS18_EDL_poster.pdf) [[code example]](https://muratsensoy.github.io/uncertainty.html) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Evidential%20Deep%20Learning%20to%20Quantify%20Classification%20Uncertainty.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Evidential%20Deep%20Learning%20to%20Quantify%20Classification%20Uncertainty.md)
- *Murat Sensoy, Lance Kaplan, Melih Kandemir*
- `2018-10-31, NeurIPS2018`

##### [18-11-22] [paper22]
-  A Probabilistic U-Net for Segmentation of Ambiguous Images [[pdf]](https://arxiv.org/abs/1806.05034) [[code]](https://github.com/SimonKohl/probabilistic_unet) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Probabilistic%20U-Net%20for%20Segmentation%20of%20Ambiguous%20Images.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/A%20Probabilistic%20U-Net%20for%20Segmentation%20of%20Ambiguous%20Images.md)
- *Simon A. A. Kohl, Bernardino Romera-Paredes, Clemens Meyer, Jeffrey De Fauw, Joseph R. Ledsam, Klaus H. Maier-Hein, S. M. Ali Eslami, Danilo Jimenez Rezende, Olaf Ronneberger*
- `2018-10-29, NeurIPS2018`

##### [18-11-22] [paper21]
- When Recurrent Models Don't Need To Be Recurrent (a.k.a. Stable Recurrent Models) [[pdf]](https://arxiv.org/abs/1805.10369) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/When%20Recurrent%20Models%20Don%E2%80%99t%20Need%20To%20Be%20Recurrent.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/When%20Recurrent%20Models%20Don't%20Need%20To%20Be%20Recurrent.md)
- *John Miller, Moritz Hardt*
- `2018-05-29, ICLR2019`

##### [18-11-16] [paper20]
- Uncertainty Estimates and Multi-Hypotheses Networks for Optical Flow [[pdf]](https://arxiv.org/abs/1802.07095) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20Estimates%20and%20Multi-Hypotheses%20Networks%20for%20Optical%20Flow_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Uncertainty%20Estimates%20and%20Multi-Hypotheses%20Networks%20for%20Optical%20Flow.md)
- *Eddy Ilg, Özgün Çiçek, Silvio Galesso, Aaron Klein, Osama Makansi, Frank Hutter, Thomas Brox*
- `2018-08-06, ECCV2018`

##### [18-11-15] [paper19]
- Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV) [[pdf]](https://arxiv.org/abs/1711.11279) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Interpretability%20Beyond%20Feature%20Attribution:%20Quantitative%20Testing%20with%20Concept%20Activation%20Vectors%20(TCAV)_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Interpretability%20Beyond%20Feature%20Attribution:%20Quantitative%20Testing%20with%20Concept%20Activation%20Vectors%20(TCAV).md)
- *Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, Rory Sayres*
- `2018-06-07, ICML2018`

##### [18-11-12] [paper18]
- Large-Scale Visual Active Learning with Deep Probabilistic Ensembles [[pdf]](https://arxiv.org/abs/1811.03575) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Large-Scale%20Visual%20Active%20Learning%20with%20Deep%20Probabilistic%20Ensembles_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Large-Scale%20Visual%20Active%20Learning%20with%20Deep%20Probabilistic%20Ensembles.md)
- *Kashyap Chitta, Jose M. Alvarez, Adam Lesnikowski*
- `2018-11-08`

##### [18-11-08] [paper17]
- The Lottery Ticket Hypothesis: Finding Small, Trainable Neural Networks [[pdf]](https://arxiv.org/abs/1803.03635) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/The%20Lottery%20Ticket%20Hypothesis:%20Finding%20Small%2C%20Trainable%20Neural%20Networks_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/The%20Lottery%20Ticket%20Hypothesis:%20Finding%20Small%2C%20Trainable%20Neural%20Networks.md)
- *Jonathan Frankle, Michael Carbin*
- `2018-03-09, ICLR2019`

##### [18-10-26] [paper16]
- Towards Safe Autonomous Driving: Capture Uncertainty in the Deep Neural Network For Lidar 3D Vehicle Detection [[pdf]](https://arxiv.org/abs/1804.05132) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Towards%20Safe%20Autonomous%20Driving:%20Capture%20Uncertainty%20in%20the%20Deep%20Neural%20Network%20For%20Lidar%203D%20Vehicle%20Detection_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Towards%20Safe%20Autonomous%20Driving:%20Capture%20Uncertainty%20in%20the%20Deep%20Neural%20Network%20For%20Lidar%203D%20Vehicle%20Detection.md)
- *Di Feng, Lars Rosenbaum, Klaus Dietmayer*
- `2018-09-08, ITSC2018`

##### [18-10-25] [paper15]
- Bayesian Convolutional Neural Networks with Many Channels are Gaussian Processes [[pdf]](https://arxiv.org/abs/1810.05148) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Bayesian%20Convolutional%20Neural%20Networks%20with%20Many%20Channels%20are%20Gaussian%20Processes_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Bayesian%20Convolutional%20Neural%20Networks%20with%20Many%20Channels%20are%20Gaussian%20Processes.md)
- *Roman Novak, Lechao Xiao, Jaehoon Lee, Yasaman Bahri, Daniel A. Abolafia, Jeffrey Pennington, Jascha Sohl-Dickstein*
- `2018-10-11, ICLR2019`

##### [18-10-19] [paper14]
- Uncertainty in Neural Networks: Bayesian Ensembling [[pdf]](https://arxiv.org/abs/1810.05546) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20in%20Neural%20Networks:%20Bayesian%20Ensembling_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Uncertainty%20in%20Neural%20Networks:%20Bayesian%20Ensembling.md)
- *Tim Pearce, Mohamed Zaki, Alexandra Brintrup, Andy Neel*
- `2018-10-12, AISTATS2019 submission`

##### [18-10-18] [paper13]
- Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles [[pdf]](https://arxiv.org/abs/1612.01474) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles.md)
- *Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell*
- `2017-11-17, NeurIPS2017`

##### [18-10-18] [paper12]
- Reliable Uncertainty Estimates in Deep Neural Networks using Noise Contrastive Priors [[pdf]](https://arxiv.org/abs/1807.09289) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Reliable%20Uncertainty%20Estimates%20in%20Deep%20Neural%20Networks%20using%20Noise%20Contrastive%20Priors_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Reliable%20Uncertainty%20Estimates%20in%20Deep%20Neural%20Networks%20using%20Noise%20Contrastive%20Priors.md)
- *Danijar Hafner, Dustin Tran, Alex Irpan, Timothy Lillicrap, James Davidson*
- `2018-07-24, ICML2018 Workshop`

##### [18-10-05] [paper11]
- VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection [[pdf]](https://arxiv.org/abs/1711.06396) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/VoxelNet:%20End-to-End%20Learning%20for%20Point%20Cloud%20Based%203D%20Object%20Detection_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/VoxelNet:%20End-to-End%20Learning%20for%20Point%20Cloud%20Based%203D%20Object%20Detection.md)
- *Yin Zhou, Oncel Tuzel*
- `2017-11-17, CVPR2018`

##### [18-10-04] [paper10]
- PIXOR: Real-time 3D Object Detection from Point Clouds [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/PIXOR:%20Real-time%203D%20Object%20Detection%20from%20Point%20Clouds_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/PIXOR:%20Real-time%203D%20Object%20Detection%20from%20Point%20Clouds.md)
- *Bin Yang, Wenjie Luo, Raquel Urtasun*
- `CVPR2018`

##### [18-10-04] [paper9]
- On gradient regularizers for MMD GANs [[pdf]](https://arxiv.org/abs/1805.11565) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20gradient%20regularizers%20for%20MMD%20GANs_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/On%20gradient%20regularizers%20for%20MMD%20GANs.md)
- *Michael Arbel, Dougal J. Sutherland, Mikołaj Bińkowski, Arthur Gretton*
- `2018-05-29, NeurIPS2018`

##### [18-09-30] [paper8]
- Neural Processes [[pdf]](https://arxiv.org/abs/1807.01622) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Processes_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Neural%20Processes.md)
- *Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J. Rezende, S.M. Ali Eslami, Yee Whye Teh*
- `2018-07-04, ICML2018 Workshop`

##### [18-09-27] [paper7]
- Conditional Neural Processes [[pdf]](https://arxiv.org/abs/1807.01613) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Conditional%20Neural%20Processes_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Conditional%20Neural%20Processes.md)
- *Marta Garnelo, Dan Rosenbaum, Chris J. Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo J. Rezende, S. M. Ali Eslami*
- `2018-07-04, ICML2018`

##### [18-09-27] [paper6]
- Neural Autoregressive Flows [[pdf]](https://arxiv.org/abs/1804.00779) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Autoregressive%20Flows_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Neural%20Autoregressive%20Flows.md)
- *Chin-Wei Huang, David Krueger, Alexandre Lacoste, Aaron Courville*
- `2018-04-03, ICML2018`

##### [18-09-25] [paper5]
- Deep Confidence: A Computationally Efficient Framework for Calculating Reliable Errors for Deep Neural Networks [[pdf]](https://arxiv.org/abs/1809.09060) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Confidence:%20A%20Computationally%20Efficient%20Framework%20for%20Calculating%20Reliable%20Errors%20for%20Deep%20Neural%20Networks.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Deep%20Confidence:%20A%20Computationally%20Efficient%20Framework%20for%20Calculating%20Reliable%20Errors%20for%20Deep%20Neural%20Networks.md)
- *Isidro Cortes-Ciriano, Andreas Bender*
- `2018-09-24`

##### [18-09-25] [paper4]
- Leveraging Heteroscedastic Aleatoric Uncertainties for Robust Real-Time LiDAR 3D Object Detection [[pdf]](https://arxiv.org/abs/1809.05590) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Leveraging%20Heteroscedastic%20Aleatoric%20Uncertainties%20for%20Robust%20Real-Time%20LiDAR%203D%20Object%20Detection_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Leveraging%20Heteroscedastic%20Aleatoric%20Uncertainties%20for%20Robust%20Real-Time%20LiDAR%203D%20Object%20Detection.md)
- *Di Feng, Lars Rosenbaum, Fabian Timm, Klaus Dietmayer*
- `2018-09-14`

##### [18-09-24] [paper3]
- Lightweight Probabilistic Deep Networks [[pdf]](https://arxiv.org/abs/1805.11327) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Lightweight%20Probabilistic%20Deep%20Networks_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Lightweight%20Probabilistic%20Deep%20Networks.md)
- *Jochen Gast, Stefan Roth*
- `2018-05-29, CVPR2018`

##### [18-09-24] [paper2]
- What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? [[pdf]](https://arxiv.org/abs/1703.04977) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/What%20Uncertainties%20Do%20We%20Need%20in%20Bayesian%20Deep%20Learning%20for%20Computer%20Vision%3F_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/What%20Uncertainties%20Do%20We%20Need%20in%20Bayesian%20Deep%20Learning%20for%20Computer%20Vision%3F.md)
- *Alex Kendall, Yarin Gal*
- `2017-10-05, NeurIPS2017`

##### [18-09-20] [paper1]
- Gaussian Process Behaviour in Wide Deep Neural Networks [[pdf]](https://arxiv.org/abs/1804.11271) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Gaussian%20Process%20Behaviour%20in%20Wide%20Deep%20Neural%20Networks.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Gaussian%20Process%20Behaviour%20in%20Wide%20Deep%20Neural%20Networks.md)
- *Alexander G. de G. Matthews, Mark Rowland, Jiri Hron, Richard E. Turner, Zoubin Ghahramani*
- `2018-08-16, ICLR2018`
