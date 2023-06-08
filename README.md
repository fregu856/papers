# About

I categorize, annotate and write comments for all research papers I read (300+ papers since 2018).

#### Categories:

[Uncertainty Estimation], [Ensembling], [Stochastic Gradient MCMC], [Variational Inference], [Out-of-Distribution Detection], [Theoretical Properties of Deep Learning], [VAEs], [Normalizing Flows], [Autonomous Driving], [Medical ML], [Object Detection], [3D Object Detection], [3D Multi-Object Tracking], [3D Human Pose Estimation], [Visual Tracking], [Sequence Modeling], [Reinforcement Learning], [System Identification], [Energy-Based Models], [Neural Processes], [Neural ODEs], [Transformers], [Implicit Neural Representations], [Distribution Shifts], [ML & Ethics], [Diffusion Models], [Graph Neural Networks], [Selective Prediction].


### Papers:

- [Papers Read in 2023](#papers-read-in-2023)
- [Papers Read in 2022](#papers-read-in-2022)
- [Papers Read in 2021](#papers-read-in-2021)
- [Papers Read in 2020](#papers-read-in-2020)
- [Papers Read in 2019](#papers-read-in-2019)
- [Papers Read in 2018](#papers-read-in-2018)

#### Papers Read in 2023:

##### [23-06-03] [paper301]
- Building One-class Detector for Anything: Open-vocabulary Zero-shot OOD Detection Using Text-image Models
 [[pdf]](https://arxiv.org/abs/2305.17207) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Building%20One-class%20Detector%20for%20Anything%3A%20Open-vocabulary%20Zero-shot%20OOD%20Detection%20Using%20Text-image%20Models.pdf)
- `2023-05`
- [Out-of-Distribution Detection]
```
Well written and interesting paper. Section 2.1 provides a good background, and their proposed OOD scores in Section 2.2 make intuitive sense. The datasets and evaluation setup in Section 3 are described well. The experimental results definitely seem promising.
```

##### [23-06-02] [paper300]
- Benchmarking Common Uncertainty Estimation Methods with Histopathological Images under Domain Shift and Label Noise
 [[pdf]](https://arxiv.org/abs/2301.01054) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Benchmarking%20Common%20Uncertainty%20Estimation%20Methods%20with%20Histopathological%20Images%20under%20Domain%20Shift%20and%20Label%20Noise.pdf)
- `2023-01`
- [Uncertainty Estimation], [Medical ML]
```
Well written and fairly interesting paper. The setup with ID/OOD data (different clinics and scanners), as described in Section 3.1, is really neat. Solid evaluation. I was not overly surprised by the results/findings. Figure 3 is neat.
```

##### [23-06-02] [paper299]
- Mechanism of Feature Learning in Deep Fully Connected Networks and Kernel Machines that Recursively Learn Features
 [[pdf]](https://arxiv.org/abs/2212.13881) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Mechanism%20of%20Feature%20Learning%20in%20Deep%20Fully%20Connected%20Networks%20and%20Kernel%20Machines%20that%20Recursively%20Learn%20Features.pdf)
- `2022-12`
- [Theoretical Properties of Deep Learning]
```
Well written and quite interesting paper. Not my main area expertise, and I would have needed to read it again to properly understand everything. Certain things seem potentially interesting, especially Section 2.1 and 2.2, but I struggle a bit to formulate one main takeaway.
```

##### [23-05-31] [paper298]
- Simplified State Space Layers for Sequence Modeling
 [[pdf]](https://arxiv.org/abs/2208.04933) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Simplified%20State%20Space%20Layers%20for%20Sequence%20Modeling.pdf)
- `ICLR 2023`
- [Sequence Modeling]
```
Well written and quite interesting paper (although not my main area of interest). Did not follow all details in Section 3 and 4.
```

##### [23-05-27] [paper297]
- CARD: Classification and Regression Diffusion Models
 [[pdf]](https://arxiv.org/abs/2206.07275) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/CARD%3A%20Classification%20and%20Regression%20Diffusion%20Models.pdf)
- `NeurIPS 2022`
- [Diffusion Models], [Uncertainty Estimation]
```
Quite well written and somewhat interesting paper. I focused mainly on the regression part, I found the classification part a bit confusing. For regression they just illustrate their method on 1D toy examples, without any baseline comparisons, and then evaluate on the UCI regression benchmark. Also, they don't compare with other simple models which can handle multi-modal p(y|x) distributions, e.g. GMMs, normalizing flows or EBMs.
```

##### [23-05-27] [paper296]
- Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration
 [[pdf]](https://arxiv.org/abs/2303.11435) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Inversion%20by%20Direct%20Iteration%3A%20An%20Alternative%20to%20Denoising%20Diffusion%20for%20Image%20Restoration.pdf)
- `2023-03`
- [Diffusion Models]
```
Interesting paper. Quite a few small typos, but overall well written. The approach becomes very similar to our paper "Image Restoration with Mean-Reverting Stochastic Differential Equations". The basic idea, training a normal regression model but letting it predict iteratively, makes intuitive sense. Figure 3 is interesting, with the trade-off between perceptual and distortion metrics, that the number of steps controls this trade-off. Figure 5 is also interesting, that adding noise (epsilon > 0) is crucial for improved perceptual metrics here. However, I don't quite understand why adding noise is beneficial for super-resolution and JPEG restoration, but not for motion/defocus deblurring? Is there some fundamental difference between those tasks?
```

##### [23-05-25] [paper295]
- Consistency Models
 [[pdf]](https://arxiv.org/abs/2303.01469) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Consistency%20Models.pdf)
- `ICML 2023`
- [Diffusion Models]
```
Well written and interesting paper. Reading it raised a few questions though. It is not quite clear to me why the moving average technique is needed during training ("the EMA update and 'stopgrad' operator in Eq. (8) can greatly stabilize the training process", why is the training unstable without it?). Algo 1 also seems somewhat heuristic? And in Figure 4 it seems like while doing 2 steps instead of 1 step improves the sample quality significantly, doing 4 steps gives basically no additional performance gain? I was expecting to see the CD sample quality to converge towards that of the original diffusion model as the number of steps increases, but here a quite significant gap seems to remain?
```

##### [23-05-12] [paper294]
- Collaborative Strategies for Deploying Artificial Intelligence to Complement Physician Diagnoses of Acute Respiratory Distress Syndrome
 [[pdf]](https://www.nature.com/articles/s41746-023-00797-9) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Collaborative%20Strategies%20for%20Deploying%20Artificial%20Intelligence%20to%20Complement%20Physician%20Diagnoses%20of%20Acute%20Respiratory%20Distress%20Syndrome.pdf)
- `npj Digital Medicine, 2023`
- [Medical ML]
```
Well written and quite interesting paper. A bit different (in a good way) compared to the pure ML papers I usually read. "It could communicate alerts to the respiratory therapist or nurses without significant physician oversight, only deferring to the physician in situations where the AI model has high uncertainty. This may be particularly helpful in low-resource settings, such as Intensive Care Units (ICU) without 24-hour access to critical care trained physicians", this would require that the model actually is well calibrated though (that you really can trust the model's uncertainty), and I'm not convinced that can be expected in many practical applications.
```

##### [23-05-03] [paper293]
- I2SB: Image-to-Image Schrödinger Bridge
 [[pdf]](https://arxiv.org/abs/2302.05872) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/I2SB:%20Image-to-Image%20Schr%C3%B6dinger%20Bridge.pdf)
- `ICML 2023`
- [Diffusion Models]
```
Well-written and interesting paper. The overall approach becomes very similar to our paper "Image Restoration with Mean-Reverting Stochastic Differential Equations" (concurrent work) it seems, and I find it quite difficult to see what the main qualitative differences actually would be in practice. Would be interesting to compare the restoration performance. I didn't fully understand everything in Section 3.
```

##### [23-04-27] [paper292]
- Assaying Out-Of-Distribution Generalization in Transfer Learning
 [[pdf]](https://arxiv.org/abs/2207.09239) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Assaying%20Out-Of-Distribution%20Generalization%20in%20Transfer%20Learning.pdf)
- `NeurIPS 2022`
- [Distribution Shifts]
```
Well-written and quite interesting paper. Just image classification, but a very extensive evaluation. Contains a lot of information, and definitely presents some quite interesting takeaways. Almost a bit too much information perhaps. I really liked the formatting, with the "Takeaway boxes" at the end of each subsection.
```

##### [23-04-20] [paper291]
- A Deep Conjugate Direction Method for Iteratively Solving Linear Systems
 [[pdf]](https://arxiv.org/abs/2205.10763) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Deep%20Conjugate%20Direction%20Method%20for%20Iteratively%20Solving%20Linear%20Systems.pdf)
- `2022-05`
```
Quite well-written and somewhat interesting paper. I really struggled to understand everything properly, I definitely don't have the required background knowledge. I don't quite understand what data they train the network on, do they train separate networks for each example? Not clear to me how generally applicable this method actually is.
```

##### [23-04-15] [paper290]
- A Roadmap to Fair and Trustworthy Prediction Model Validation in Healthcare
 [[pdf]](https://arxiv.org/abs/2304.03779) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Roadmap%20to%20Fair%20and%20Trustworthy%20Prediction%20Model%20Validation%20in%20Healthcare.pdf)
- `2023-04`
- [Medical ML]
```
A different type of paper compared to what I normally read (the title sounded interesting and I was just curious to read something a bit different). A quick read, fairly interesting. Not sure if I agree with the authors though (it might of course also just be that I don't have a sufficient background understanding). "...some works consider evaluation using external data to be stringent and highly encouraged due to the difference in population characteristics in evaluation and development settings. We propose an alternative roadmap for fair and trustworthy external validation using local data from the target population...", here I would tend to agree with the first approach, not their proposed alternative.
```

##### [23-04-15] [paper289]
- Deep Anti-Regularized Ensembles Provide Reliable Out-of-Distribution Uncertainty Quantification
 [[pdf]](https://arxiv.org/abs/2304.04042) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Anti-Regularized%20Ensembles%20Provide%20Reliable%20Out-of-Distribution%20Uncertainty%20Quantification.pdf)
- `2023-04`
- [Uncertainty Estimation]
```
Well-written and fairly interesting paper. The idea is quite interesting and neat. I like the evaluation approach used in the regression experiments, with distribution shifts. Their results in Table 1 are a bit better than the baselines, but the absolute performance is still not very good. Not particularly impressed by the classification OOD detection experiments.
```

##### [23-04-15] [paper288]
- SIO: Synthetic In-Distribution Data Benefits Out-of-Distribution Detection
 [[pdf]](https://arxiv.org/abs/2303.14531) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/SIO:%20Synthetic%20In-Distribution%20Data%20Benefits%20Out-of-Distribution%20Detection.pdf)
- `2023-03`
- [Out-of-Distribution Detection]
```
Well-written and fairly interesting paper. Extremely simple idea and it seems to quite consistently improve the detection performance of various methods a bit. Another potentially useful tool. 
```

##### [23-04-05] [paper287]
- Evaluating the Fairness of Deep Learning Uncertainty Estimates in Medical Image Analysis
 [[pdf]](https://arxiv.org/abs/2303.03242) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Evaluating%20the%20Fairness%20of%20Deep%20Learning%20Uncertainty%20Estimates%20in%20Medical%20Image%20Analysis.pdf)
- `MIDL 2023`
- [Uncertainty Estimation], [Medical ML]
```
Well-written and somewhat interesting paper. The studied problem is interesting and important, but I'm not sure about the evaluation approach. "when the uncertainty threshold is reduced, thereby increasing the number of filtered uncertain predictions, the differences in the performances on the remaining confident predictions across the subgroups should be reduced", I'm not sure this is the best metric one could use. I think there are other aspects which also would be important to measure (e.g. calibration). Also, I find it difficult to interpret the results or compare methods in Figure 2 - 4.
```

##### [23-03-30] [paper286]
- PID-GAN: A GAN Framework based on a Physics-informed Discriminator for Uncertainty Quantification with Physics
 [[pdf]](https://arxiv.org/abs/2106.02993) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/PID-GAN:%20A%20GAN%20Framework%20based%20on%20a%20Physics-informed%20Discriminator%20for%20Uncertainty%20Quantification%20with%20Physics.pdf)
- `KDD 2021`
- [Uncertainty Estimation]
```
Quite well-written and somewhat interesting paper. Compared to the "PIG-GAN" baseline, their method seems to be an improvement. However, I'm not overly convinced about the general method, it sort of seems unnecessarily complicated to me. 
```

##### [23-03-23] [paper285]
- Resurrecting Recurrent Neural Networks for Long Sequences
 [[pdf]](https://arxiv.org/abs/2303.06349) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Resurrecting%20Recurrent%20Neural%20Networks%20for%20Long%20Sequences.pdf)
- `2023-03`
- [Sequence Modeling]
```
Quite well-written and quite interesting paper. I did not really have the background knowledge necessary to properly evaluate/understand/appreciate everything. The paper is quite dense, contains a lot of detailed information. Still quite interesting though, seems to provide a number of relatively interesting insights.
```

##### [23-03-16] [paper284]
- Why AI is Harder Than We Think
 [[pdf]](https://arxiv.org/abs/2104.12871) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Why%20AI%20is%20Harder%20Than%20We%20Think.pdf)
- `2021-04`
```
Interesting and well-written paper. A bit different than the papers I usually read, but in a good way. I enjoyed reading it and it made me think.
```

##### [23-03-11] [paper283]
- How to Exploit Hyperspherical Embeddings for Out-of-Distribution Detection?
 [[pdf]](https://openreview.net/forum?id=aEFaE0W5pAd) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/How%20to%20Exploit%20Hyperspherical%20Embeddings%20for%20Out-of-Distribution%20Detection%3F.pdf)
- `ICLR 2023`
- [Out-of-Distribution Detection]
```
Very well-written and quite interesting paper. Very similar to "Out-of-Distribution Detection with Deep Nearest Neighbors", just use their proposed loss in equation (7) instead of SupCon, right? Somewhat incremental I suppose, but it's also quite neat that such a simple modification consistently improves the OOD detection performance. The analysis in Section 4.3 is also quite interesting.
```

##### [23-03-11] [paper282]
- Out-of-Distribution Detection with Deep Nearest Neighbors
 [[pdf]](https://arxiv.org/abs/2204.06507) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Out-of-Distribution%20Detection%20with%20Deep%20Nearest%20Neighbors.pdf)
- `ICML 2022`
- [Out-of-Distribution Detection]
```
Interesting and very well-written paper, I enjoyed reading it. They propose a simple extension of "SSD: A Unified Framework for Self-Supervised Outlier Detection": to use kNN distance to the train feature vectors instead of Mahalanobis distance. Very simple and intuitive, and consistently improves the results.
```

##### [23-03-11] [paper281]
- SSD: A Unified Framework for Self-Supervised Outlier Detection
 [[pdf]](https://arxiv.org/abs/2103.12051) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/SSD:%20A%20Unified%20Framework%20for%20Self-Supervised%20Outlier%20Detection.pdf)
- `ICLR 2021`
- [Out-of-Distribution Detection]
```
Well-written and interesting paper. The method is simple and makes intuitive sense, yet seems to perform quite well.
```

##### [23-03-10] [paper280]
- Rethinking Out-of-distribution (OOD) Detection: Masked Image Modeling is All You Need
 [[pdf]](https://arxiv.org/abs/2302.02615) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Rethinking%20Out-of-distribution%20(OOD)%20Detection:%20Masked%20Image%20Modeling%20is%20All%20You%20Need.pdf)
- `2023-02`
- [Out-of-Distribution Detection]
```
Quite interesting, but not overly well-written paper. I don't like the "... is all you need" title, and they focus too much on selling how their method beats SOTA (Figure 1 does definitely not illustrate the performance difference in a fair way).
```

##### [23-03-10] [paper279]
- Out-of-Distribution Detection and Selective Generation for Conditional Language Models
 [[pdf]](https://openreview.net/forum?id=kJUS5nD0vPB) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Out-of-Distribution%20Detection%20and%20Selective%20Generation%20for%20Conditional%20Language%20Models.pdf)
- `ICLR 2023`
- [Out-of-Distribution Detection], [Selective Prediction]
```
Well-written and quite interesting paper. Doing "selective generation" generally makes sense. Their method seems like a quite intuitive extension of "A simple fix to Mahalanobis distance for improving near-OOD detection" (relative Mahalanobis distance) to the setting of language models. Also seems to perform quite well, but not super impressive performance compared to the baselines perhaps.
```

##### [23-03-09] [paper278]
- Learning to Reject Meets OOD Detection: Are all Abstentions Created Equal?
 [[pdf]](https://arxiv.org/abs/2301.12386) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20to%20Reject%20Meets%20OOD%20Detection:%20Are%20all%20Abstentions%20Created%20Equal%3F.pdf)
- `2023-01`
- [Out-of-Distribution Detection], [Selective Prediction]
```
Quite well-written and fairly interesting paper. I struggled to properly follow some parts. I'm not entirely convinced by their proposed approach.
```

##### [23-03-09] [paper277]
- Calibrated Selective Classification
 [[pdf]](https://openreview.net/forum?id=zFhNBs8GaV) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Calibrated%20Selective%20Classification.pdf)
- `TMLR, 2022`
- [Uncertainty Estimation], [Selective Prediction]
```
Well-written and quite interesting paper. The overall aim of "we extend selective classification to focus on improving model calibration over non-rejected instances" makes a lot of sense to me. The full proposed method (Section 4.2 - 4.5) seems a bit complicated though, but the experiments and results are definitely quite interesting. 
```

##### [23-03-09] [paper276]
- How Powerful are Graph Neural Networks?
 [[pdf]](https://arxiv.org/abs/1810.00826) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/How%20Powerful%20are%20Graph%20Neural%20Networks%3F.pdf)
- `ICLR 2019`
- [Graph Neural Networks]
```
Very well-written paper. There are topics which I generally find a lot more interesting, but I still definitely enjoyed reading this paper.
```

##### [23-03-08] [paper275]
- A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification
 [[pdf]](https://openreview.net/forum?id=YnkGMIh0gvX) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Call%20to%20Reflect%20on%20Evaluation%20Practices%20for%20Failure%20Detection%20in%20Image%20Classification.pdf)
- `ICLR 2023`
- [Out-of-Distribution Detection]
```
Interesting and well-written paper, I'm glad that I found it and decided to read it in detail. The appendix contains a lot of information (and I did not have time to go through everything). Overall, I really like what the authors set out do with this paper. But in the end, I'm not entirely convinced. The AURC metric still has some issues, I think.
```

##### [23-03-08] [paper274]
- High-Resolution Image Synthesis with Latent Diffusion Models
 [[pdf]](https://arxiv.org/abs/2112.10752) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/High-Resolution%20Image%20Synthesis%20with%20Latent%20Diffusion%20Models.pdf)
- `CVPR 2022`
- [Diffusion Models]
```
Quite interesting and well-written paper. The method is described well in Section 3. Section 4.1 is quite interesting. The rest of the results I did not go through in much detail. Update 23-05-11: Read the paper again for our reading group, pretty much exactly the same impression this second time. The overall idea is simple and neat.
```

##### [23-03-07] [paper273]
- Certifying Out-of-Domain Generalization for Blackbox Functions
 [[pdf]](https://arxiv.org/abs/2202.01679) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Certifying%20Out-of-Domain%20Generalization%20for%20Blackbox%20Functions.pdf)
- `ICML 2022`
- [Distribution Shifts]
```
Well-written and quite interesting paper, I'm just not even close to having the background necessary to be able to properly understand/appreciate/evaluate these results. Could this be used in practice? If so, how useful would it actually be? I have basically no clue.
```

##### [23-03-07] [paper272]
- Predicting Out-of-Distribution Error with the Projection Norm
 [[pdf]](https://proceedings.mlr.press/v162/yu22i.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Predicting%20Out-of-Distribution%20Error%20with%20the%20Projection%20Norm.pdf)
- `ICML 2022`
- [Distribution Shifts]
```
Well-written and quite interesting paper. The method is conceptually simple and makes some intuitive sense. I'm just not quite sure how/when this approach actually would be used in practice? They say in Section 6 that "Our method can potentially be extended to perform OOD detection", but I don't really see how that would be possible (since the method seems to require at least ~200 test samples)?
```

##### [23-03-07] [paper271]
- Variational- and Metric-based Deep Latent Space for Out-of-Distribution Detection
 [[pdf]](https://openreview.net/forum?id=ScLeuUUi9gq) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Variational-%20and%20Metric-based%20Deep%20Latent%20Space%20for%20Out-of-Distribution%20Detection.pdf)
- `UAI 2022`
- [Out-of-Distribution Detection]
```
Quite well-written and somewhat interesting paper. Seems a bit ad hoc and quite incremental overall.
```

##### [23-03-07] [paper270]
- Igeood: An Information Geometry Approach to Out-of-Distribution Detection
 [[pdf]](https://arxiv.org/abs/2203.07798) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Igeood:%20An%20Information%20Geometry%20Approach%20to%20Out-of-Distribution%20Detection.pdf)
- `ICLR 2022`
- [Out-of-Distribution Detection]
```
Quite well-written and somewhat interesting paper. The proposed method seems a bit ad hoc to me. Not overly impressive experimental results. Seems a bit incremental overall.
```

##### [23-03-03] [paper269]
- The Tilted Variational Autoencoder: Improving Out-of-Distribution Detection
 [[pdf]](https://openreview.net/forum?id=YlGsTZODyjz) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/The%20Tilted%20Variational%20Autoencoder:%20Improving%20Out-of-Distribution%20Detection.pdf)
- `ICLR 2023`
- [Out-of-Distribution Detection], [VAEs]
```
Quite well-written and somewhat interesting paper. I still don't fully understand the "Will-it-move test", not even after having read Appendix D. It seems a bit strange to me, and it requires access to OOD data. So, then you get the same type of problems as all "outlier exposure"-style methods (what if you don't have access to OOD data? And will the OOD detector actually generalize well to other OOD data than what it was tuned on)? Section 4.2.1 pretty interesting though.
```

##### [23-03-02] [paper268]
- Improving Reconstruction Autoencoder Out-of-distribution Detection with Mahalanobis Distance
 [[pdf]](https://arxiv.org/abs/1812.02765) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Improving%20Reconstruction%20Autoencoder%20Out-of-distribution%20Detection%20with%20Mahalanobis%20Distance.pdf)
- `2018-12`
- [Out-of-Distribution Detection]
```
Quite well-written and somewhat interesting paper. Short (~4 pages) and a very quick read. A simple idea that makes intuitive sense. Very basic experiments (only MNIST).
```

##### [23-03-02] [paper267]
- Denoising Diffusion Models for Out-of-Distribution Detection
 [[pdf]](https://arxiv.org/abs/2211.07740) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Denoising%20Diffusion%20Models%20for%20Out-of-Distribution%20Detection.pdf)
- `2022-11`
- [Out-of-Distribution Detection], [Diffusion Models]
```
Well-written and interesting paper, I enjoyed reading it. Very similar to "Unsupervised Out-of-Distribution Detection with Diffusion Inpainting" (reconstruction-based OOD detection using diffusion models), but using a slightly different approach. The related work is described in a really nice way, and they compare with very relevant baselines it seems. Promising performance in the experiments.
```

##### [23-03-01] [paper266]
- Conformal Prediction Beyond Exchangeability
 [[pdf]](https://arxiv.org/abs/2202.13415) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Conformal%20Prediction%20Beyond%20Exchangeability.pdf)
- `2022-02`
- [Uncertainty Estimation], [Distribution Shifts]
```
Well-written and quite interesting paper, I quite enjoyed reading it. Much longer than usual (32 pages), but didn't really take longer than usual to read (I skipped/skimmed some of the theoretical parts). Their proposed method makes intuitive sense I think, but seems like it's applicable only to problems in which some kind of prior knowledge can be used to compute weights? From the end of Section 4.3: "On the other hand, if the test point comes from a new distribution that bears no resemblance to the training data, neither our upper bound nor any other method would be able to guarantee valid coverage without further assumptions. An important open question is whether it may be possible to determine, in an adaptive way, whether coverage will likely hold for a particular data set, or whether that data set exhibits high deviations from exchangeability such that the coverage gap may be large".
```

##### [23-02-27] [paper265]
- Robust Validation: Confident Predictions Even When Distributions Shift
 [[pdf]](https://arxiv.org/abs/2008.04267) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Robust%20Validation:%20Confident%20Predictions%20Even%20When%20Distributions%20Shift.pdf)
- `2020-08`
- [Uncertainty Estimation], [Distribution Shifts]
```
Quite interesting and well-written paper. Longer (~19 pages) and more theoretical than what I usually read. I did not understand all details in Section 2 and 3. Also find it difficult to know how Algorithm 2 and 3 actually are implemented, would like see some code. Not entirely sure how useful their methods actually would be in practice, but I quite enjoyed reading the paper at least.
```

##### [23-02-24] [paper264]
- Conformal Prediction Under Covariate Shift
 [[pdf]](https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Conformal%20Prediction%20Under%20Covariate%20Shift.pdf)
- `NeurIPS 2019`
- [Uncertainty Estimation], [Distribution Shifts]
```
Quite interesting paper. It contains more theoretical results than I'm used to, and some things are sort of explained in an unnecessarily complicated way. The proposed method in Section 2 makes some intuitive sense, but I also find it a bit odd. It requires access to unlabeled test inputs, and then you'd have to train a classifier to distinguish train inputs from test inputs? Is this actually a viable approach in practice? Would it work well e.g. for image data? Not clear to me. In the paper, the method is applied to a single very simple example.
```

##### [23-02-23] [paper263]
- Unsupervised Out-of-Distribution Detection with Diffusion Inpainting
 [[pdf]](https://arxiv.org/abs/2302.10326) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Unsupervised%20Out-of-Distribution%20Detection%20with%20Diffusion%20Inpainting.pdf)
- `2023-02`
- [Out-of-Distribution Detection], [Diffusion Models]
```
Well-written and interesting paper, I enjoyed reading it. The proposed method is conceptually very simple and makes a lot of intuitive sense. As often is the case with OOD detection papers, I find it difficult to judge how strong/impressive the experimental results actually are (the method is evaluated only on quite simple/small image classification datasets), but it seems quite promising at least.
```

##### [23-02-23] [paper262]
- Adaptive Conformal Inference Under Distribution Shift
 [[pdf]](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Adaptive%20Conformal%20Inference%20Under%20Distribution%20Shift.pdf)
- `NeurIPS 2021`
- [Uncertainty Estimation], [Distribution Shifts]
```
Interesting and well-written paper. The proposed method in Section 2 is quite intuitive and clearly explained. The examples in Figure 1 and 3 are quite neat. "The methods we develop are specific to cases where Y_t is revealed at each time point. However, there are many settings in which we receive the response in a delayed fashion or in large batches." - this is true, but there are also many settings in which the method would not really be applicable. In cases which it is though, I definitely think it could make sense to use this instead of standard conformal prediction.
```

##### [23-02-23] [paper261]
- Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting
 [[pdf]](https://arxiv.org/abs/2103.07719) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Spectral%20Temporal%20Graph%20Neural%20Network%20for%20Multivariate%20Time-series%20Forecasting.pdf)
- `NeurIPS 2020`
- [Sequence Modeling], [Graph Neural Networks]
```
Quite interesting and well-written paper, not a topic that I personally find overly interesting though.
```

##### [23-02-16] [paper260]
- Neural Networks Trained with SGD Learn Distributions of Increasing Complexity
 [[pdf]](https://arxiv.org/abs/2211.11567) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Networks%20Trained%20with%20SGD%20Learn%20Distributions%20of%20Increasing%20Complexity.pdf)
- `2022-11`
- [Theoretical Properties of Deep Learning]
```
Interesting paper. I would have needed a bit more time to read it though, felt like I didn't quite have enough time to properly understand everything and evaluate the significance of the findings. Might have to go back to this paper again.
```

##### [23-02-09] [paper259]
- The Forward-Forward Algorithm: Some Preliminary Investigations
 [[pdf]](https://arxiv.org/abs/2212.13345) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/The%20Forward-Forward%20Algorithm:%20Some%20Preliminary%20Investigations.pdf)
- `2022-12`
```
Somewhat interesting, but quite odd paper. I was quite confused by multiple parts of it. This is probably partly because of my background, but I do also think that the paper could be more clearly structured.
```

##### [23-02-01] [paper258]
- Everything is Connected: Graph Neural Networks
 [[pdf]](https://arxiv.org/abs/2301.08210) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Everything%20is%20Connected:%20Graph%20Neural%20Networks.pdf)
- `2023-01`
- [Graph Neural Networks]
```
Quite interesting and well-written paper. A short survey, took just ~40 min to read. Not overly interesting, but a quite enjoyable read. Section 4, with the connection to transformers, is quite interesting.
```

##### [23-01-27] [paper257]
- Gradient Descent Happens in a Tiny Subspace
 [[pdf]](https://arxiv.org/abs/1812.04754) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Gradient%20Descent%20Happens%20in%20a%20Tiny%20Subspace.pdf)
- `2018-12`
- [Theoretical Properties of Deep Learning]
```
Quite interesting paper. Structured in a somewhat unusual way. Some kind of interesting observations. Difficult for me to judge how significant / practically impactful these observations actually are though.
```

##### [23-01-19] [paper256]
- Out-Of-Distribution Detection Is Not All You Need
 [[pdf]](https://arxiv.org/abs/2211.16158) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Out-Of-Distribution%20Detection%20Is%20Not%20All%20You%20Need.pdf)
- `AAAI 2023`
- [Out-of-Distribution Detection]
```
Quite interesting and well-written paper. How they describe limitations of OOD detection makes sense to me, I have always found the way OOD detection methods are evaluated a bit strange/arbitrary. However, I am not sure that the solution proposed in this paper actually is the solution.
```

##### [23-01-10] [paper255]
- Diffusion Models: A Comprehensive Survey of Methods and Applications
 [[pdf]](https://arxiv.org/abs/2209.00796) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Diffusion%20Models:%20A%20Comprehensive%20Survey%20of%20Methods%20and%20Applications.pdf)
- `2022-09`
- [Diffusion Models]
```
Quite interesting and well-written paper. ~28 pages, so a longer paper than usual. Section 1 and 2 (the first 9 pages) are interesting, they describe and show connections between the "denoising diffusion probabilistic models", "score-based generative models" and "stochastic differential equations" approaches. The remainder of the paper is quite but not overly interesting, I read it in less detail.
```

#### Papers Read in 2022:

##### [22-12-14] [paper254]
- Continuous Time Analysis of Momentum Methods
 [[pdf]](https://arxiv.org/abs/1906.04285) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Continuous%20Time%20Analysis%20of%20Momentum%20Methods.pdf)
- `JMLR, 2020`
- [Theoretical Properties of Deep Learning]
```
Quite well-written and somewhat interesting paper. Longer (~20 pages) and more theoretical paper than what I usually read, and I definitely didn't understand all the details, but still a fairly enjoyable read. More enjoyable than I expected at least.
```

##### [22-12-14] [paper253]
- Toward a Theory of Justice for Artificial Intelligence
 [[pdf]](https://direct.mit.edu/daed/article-pdf/151/2/218/2009164/daed_a_01911.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Toward%20a%20Theory%20of%20Justice%20for%20Artificial%20Intelligence.pdf)
- `Daedalus, 2022`
- [ML & Ethics]
```
Well-written and quite interesting paper. Describes the distributive justice principles of John Rawls' book "A theory of justice" and explores/discusses what these might imply for how "AI systems" should be regulated/deployed/etc. Doesn't really provide any overly concrete takeaways, at least not for me, but still a quite enjoyable read.
```

##### [22-12-08] [paper252]
- Talking About Large Language Models
 [[pdf]](https://arxiv.org/abs/2212.03551) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Talking%20About%20Large%20Language%20Models.pdf)
- `2022-12`
- [ML & Ethics]
```
Well-written and interesting paper. Sections 1-6 and Section 11 are very interesting. A breath of fresh air to read this in the midst of the ChatGPT hype. It contains a lot of good quotes, for example:"To ensure that we can make informed decisions about the trustworthiness and safety of the AI systems we deploy, it is advisable to keep to the fore the way those systems actually work, and thereby to avoid imputing to them capacities they lack, while making the best use of the remarkable capabilities they genuinely possess".
```

##### [22-12-06] [paper251]
- Artificial Intelligence, Humanistic Ethics
 [[pdf]](https://direct.mit.edu/daed/article-pdf/151/2/232/2009174/daed_a_01912.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Artificial%20Intelligence%2C%20Humanistic%20Ethics.pdf)
- `Daedalus, 2022`
- [ML & Ethics]
```
Well-written and interesting paper. Provides some interesting comments/critique on utilitarianism and how engineers/scientists like myself might be inclined to find that approach attractive: "The optimizing mindset prevalent among computer scientists and economists, among other powerful actors, has led to an approach focused on maximizing the fulfilment of human preferences..... But this preference-based utilitarianism is open to serious objections. This essay sketches an alternative, “humanistic” ethics for AI that is sensitive to aspects of human engagement with the ethical often missed by the dominant approach." - - - - "So ethics is reduced to an exercise in prediction and optimization: which act or policy is likely to lead to the optimal fulfilment of human preferences?" - - - - "This incommensurability calls into question the availability of some optimizing function that determines the single option that is, all things considered, most beneficial or morally right, the quest for which has animated a lot of utilitarian thinking in ethics."
```

##### [22-12-06] [paper250]
- Physics-Informed Neural Networks for Cardiac Activation Mapping
 [[pdf]](https://www.frontiersin.org/articles/10.3389/fphy.2020.00042/full) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Physics-Informed%20Neural%20Networks%20for%20Cardiac%20Activation%20Mapping.pdf)
- `Frontiers in Physics, 2020`
```
Quite well-written and somewhat interesting paper.
```

##### [22-12-05] [paper249]
- AI Ethics and its Pitfalls: Not Living up to its own Standards?
 [[pdf]](https://link.springer.com/article/10.1007/s43681-022-00173-5) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/AI%20ethics%20and%20its%20pitfalls:%20not%20living%20up%20to%20its%20own%20standards%3F.pdf)
- `AI and Ethics, 2022`
- [ML & Ethics]
```
Well-written and somewhat interesting paper. Good reminder that also the practice of ML ethics could have unintended negative consequences. Section 2.6 is quite interesting.
```

##### [22-12-02] [paper248]
- Blind Spots in AI Ethics
 [[pdf]](https://link.springer.com/article/10.1007/s43681-021-00122-8) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Blind%20spots%20in%20AI%20ethics.pdf)
- `AI and Ethics, 2022`
- [ML & Ethics]
```
Well-written and very interesting paper. I enjoyed reading it, and it made me think - which is a good thing! Contains quite a few quotes which I really liked, for example: "However, it is wrong to assume that the goal is ethical AI. Rather, the primary aim from which detailed norms can be derived should be a peaceful, sustainable, and just society. Hence, AI ethics must dare to ask the question where in an ethical society one should use AI and its inherent principle of predictive modeling and classification at all".
```

##### [22-12-01] [paper247]
- The Ethics of AI Ethics: An Evaluation of Guidelines
 [[pdf]](https://link.springer.com/article/10.1007/s11023-020-09517-8) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/The%20Ethics%20of%20AI%20Ethics:%20An%20Evaluation%20of%20Guidelines.pdf)
- `Minds and Machines, 2020`
- [ML & Ethics]
```
Well-written and interesting paper. I liked that it discussed some actual ethical theories in Section 4.2. Sections 3.2, 3.3. and 4.1 were also interesting.
```

##### [22-12-01] [paper246]
- The Uselessness of AI Ethics
 [[pdf]](https://link.springer.com/article/10.1007/s43681-022-00209-w) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/The%20uselessness%20of%20AI%20ethics.pdf)
- `AI and Ethics, 2022`
- [ML & Ethics]
```
Well-written and very interesting paper. I enjoyed reading it, and it made me think - which is a good thing!
```

##### [22-12-01] [paper245]
- Denoising Diffusion Implicit Models
 [[pdf]](https://arxiv.org/abs/2010.02502) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Denoising%20Diffusion%20Implicit%20Models.pdf)
- `ICLR 2021`
- [Diffusion Models]
```
Quite well-written and interesting paper. I did struggle to properly understand everything in Section 3 & 4, felt like I didn't quite have the necessary background knowledge. Helped a lot to go through the paper again at our reading group.
```

##### [22-11-26] [paper244]
- You Cannot Have AI Ethics Without Ethics
 [[pdf]](https://link.springer.com/article/10.1007/s43681-020-00013-4)
- `AI and Ethics, 2021`
- [ML & Ethics]
```
Well-written and quite interesting paper. Just 5 pages long, quick to read. Sort of like an opinion piece. I enjoyed reading it. Main takeaway: "Instead of trying to reinvent ethics, or adopt ethical guidelines in isolation, it is incumbent upon us to recognize the need for broadly ethical organizations. These will be the only entrants in a position to build truly ethical AI. You cannot simply have AI ethics. It requires real ethical due diligence at the organizational level—perhaps, in some cases, even industry-wide refection".
```

##### [22-11-25] [paper243]
- Expert Responsibility in AI Development
 [[pdf]](https://link.springer.com/article/10.1007/s00146-022-01498-9) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Expert%20responsibility%20in%20AI%20development.pdf)
- `AI & Society, 2022`
- [ML & Ethics]
```
Well-written and interesting paper, quite straightforward to follow and understand everything. Section 6 & 7 are interesting, with the discussion about unintended consequences of recommender algorithms (how they contribute to an impaired democratic debate).
```

##### [22-11-25] [paper242]
- The future of AI in our hands? To what extent are we as individuals morally responsible for guiding the development of AI in a desirable direction?
 [[pdf]](https://link.springer.com/article/10.1007/s43681-021-00125-5) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/The%20future%20of%20AI%20in%20our%20hands%3F%20To%20what%20extent%20are%20we%20as%20individuals%20morally%20responsible%20for%20guiding%20the%20development%20of%20AI%20in%20a%20desirable%20direction%3F.pdf)
- `AI and Ethics, 2022`
- [ML & Ethics]
```
Well-written and somewhat interesting paper. Not overly technical or difficult to read. Discusses different perspectives on who should be responsible for ensuring that the future development of "AI" actually benefits society.
```

##### [22-11-24] [paper241]
- Collocation Based Training of Neural Ordinary Differential Equations
 [[pdf]](https://www.researchgate.net/publication/353112789_Collocation_based_training_of_neural_ordinary_differential_equations) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Collocation%20based%20training%20of%20neural%20ordinary%20differential%20equations.pdf)
- `Statistical Applications in Genetics and Molecular Biology, 2021`
- [Neural ODEs]
```
Quite well-written and fairly interesting paper. Not sure how much new insight it actually provided for me, but still interesting to read papers from people working in more applied fields.
```

##### [22-11-17] [paper240]
- Prioritized Training on Points that are learnable, Worth Learning, and Not Yet Learnt
 [[pdf]](https://proceedings.mlr.press/v162/mindermann22a/mindermann22a.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Prioritized%20Training%20on%20Points%20that%20are%20learnable%2C%20Worth%20Learning%2C%20and%20Not%20Yet%20Learnt.pdf)
- `ICML 2022`
```
Well-written and quite interesting paper. The proposed method is explained well and makes intuitive sense overall, and seems to perform well in the intended setting.
```

##### [22-11-09] [paper239]
- Learning Deep Representations by Mutual Information Estimation and Maximization
 [[pdf]](https://openreview.net/forum?id=Bklr3j0cKX) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20deep%20representations%20by%20mutual%20information%20estimation%20and%20maximization.pdf)
- `ICLR 2019`
```
Quite interesting paper, but I struggled to properly understand everything. I might not have the necessary background knowledge. I find it difficult to formulate what my main takeaway from the paper would be, their proposed method seems quite similar to previous work? And also difficult to judge how significant/impressive their experimental results are?
```

##### [22-11-03] [paper238]
- Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution
 [[pdf]](https://openreview.net/forum?id=UYneFzXSJWh) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Fine-Tuning%20can%20Distort%20Pretrained%20Features%20and%20Underperform%20Out-of-Distribution.pdf)
- `ICLR 2022`
- [Theoretical Properties of Deep Learning]
```
Quite interesting and very well-written paper, I found it very easy to read and understand (to read it also took a lot less time than usual). Pretty much all the results/arguments make intuitive sense, and the proposed method (of first doing linear probing and then full fine-tuning) seems to perform well. I am not quite able to judge how significant/interesting/important these results are, but the paper was definitely an enjoyable read at least.
```

##### [22-10-26] [paper237]
- Multi-scale Feature Learning Dynamics: Insights for Double Descent
 [[pdf]](https://proceedings.mlr.press/v162/pezeshki22a.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Multi-scale%20Feature%20Learning%20Dynamics:%20Insights%20for%20Double%20Descent.pdf)
- `ICML 2022`
- [Theoretical Properties of Deep Learning]
```
Quite well-written paper. Definitely not my area of expertise, and I did not have enough time to really try and understand everything properly either. So, it is very difficult for me to judge how significant/important/interesting the analysis and experimental results actually are.
```

##### [22-10-20] [paper236]
- Pseudo-Spherical Contrastive Divergence
 [[pdf]](https://arxiv.org/abs/2111.00780) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Pseudo-Spherical%20Contrastive%20Divergence.pdf)
- `NeurIPS 2021`
- [Energy-Based Models]
```
Well-written and quite interesting paper. Not overly impressed by the experimental results, the "robustness to data contamination" problem seems a bit odd overall to me. The proposed training method is quite neat though (that it's not just a heuristic but follows from the scoring rule approach), and the flexibility offered by the hyperparameter gamma can probably be useful in practice sometimes.
```

##### [22-10-08] [paper235]
- RankFeat: Rank-1 Feature Removal for Out-of-distribution Detection
 [[pdf]](https://arxiv.org/abs/2209.08590) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/RankFeat:%20Rank-1%20Feature%20Removal%20for%20Out-of-distribution%20Detection.pdf)
- `NeurIPS 2022`
- [Out-of-Distribution Detection]
```
Quite interesting and well-written paper. The proposed method is quite neat / conceptually simple, and seems to perform very well relative to other post-hoc OOD detection scores. I don't expect the proposed score to perform well in all settings though, but it definitely seems like a useful tool.
```

##### [22-10-06] [paper234]
- Mechanistic Models Versus Machine Learning, a Fight Worth Fighting for the Biological Community?
 [[pdf]](https://royalsocietypublishing.org/doi/epdf/10.1098/rsbl.2017.0660) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Mechanistic%20models%20versus%20machine%20learning%2C%20a%20fight%20worth%20fighting%20for%20the%20biological%20community%3F.pdf)
- `Biology Letters, 2018`
```
An opinion peace, not really a technical paper. Just 3-4 pages long. Well-written and quite interesting paper though, I quite enjoyed reading it. What the authors write at the end "Fundamental biology should not choose between small-scale mechanistic understanding and large-scale prediction. It should embrace the complementary strengths of mechanistic modelling and machine learning approaches to provide, for example, the missing link between patient outcome prediction and the mechanistic understanding of disease progression" makes a lot of sense to, this is my main takeaway. I also find the statement "The training of a new generation of researchers versatile in all these fields will be vital in making this breakthrough" quite interesting, this is probably true for really making progress in medical machine learning applications as well?
```

##### [22-09-22] [paper233]
- Adversarial Examples Are Not Bugs, They Are Features
 [[pdf]](https://papers.nips.cc/paper/2019/hash/e2c420d928d4bf8ce0ff2ec19b371514-Abstract.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Adversarial%20Examples%20Are%20Not%20Bugs%2C%20They%20Are%20Features.pdf)
- `NeurIPS 2019`
```
Well-written and interesting paper, I quite enjoyed reading it. I found this quite a lot more interesting than previous papers I have read on adversarial examples. 
```

##### [22-09-15] [paper232]
- Learning to Learn by Gradient Descent by Gradient Descent
 [[pdf]](https://proceedings.neurips.cc/paper/2016/file/fb87582825f9d28a8d42c5e5e5e8b23d-Paper.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20to%20learn%20by%20gradient%20descent%20by%20gradient%20descent.pdf)
- `NeurIPS 2016`
```
Quite interesting and well-written paper. Not my area of expertise, but still a relatively enjoyable read. "After each epoch (some fixed number of learning steps) we freeze the optimizer parameters..." is quite unclear though, it seems like they never specify for how many number of steps the optimizer is trained?
```

##### [22-09-01] [paper231]
- On the Information Bottleneck Theory of Deep Learning
 [[pdf]](https://openreview.net/forum?id=ry_WPG-A-&noteId=ry_WPG-A-) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20the%20Information%20Bottleneck%20Theory%20of%20Deep%20Learning.pdf)
- `ICLR 2018`
- [Theoretical Properties of Deep Learning]
```
Well-written and quite interesting paper. I was not particularly familiar with the previous information bottleneck papers, but everything was still fairly easy to follow. The discussion/argument on openreview is strange (`This “paper” attacks our work through the following flawed and misleading statements`), i honestly don't know who is correct.
```

##### [22-06-28] [paper230]
- Aleatoric and Epistemic Uncertainty with Random Forests
 [[pdf]](https://arxiv.org/abs/2001.00893) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Aleatoric%20and%20Epistemic%20Uncertainty%20with%20Random%20Forests.pdf)
- `2020-01`
- [Uncertainty Estimation]
```
Quite well-written and somewhat interesting paper. 
```

##### [22-06-23] [paper229]
- Linear Time Sinkhorn Divergences using Positive Features
 [[pdf]](https://proceedings.neurips.cc/paper/2020/hash/9bde76f262285bb1eaeb7b40c758b53e-Abstract.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Linear%20Time%20Sinkhorn%20Divergences%20using%20Positive%20Features.pdf)
- `NeurIPS 2020`
```
Fairly well-written and somewhat interesting paper. Definitely not my area of expertise, I struggled to understand some parts of the paper, and it's difficult for me to judge how important/significant/useful the presented method actually is.
```

##### [22-06-17] [paper228]
- Greedy Bayesian Posterior Approximation with Deep Ensembles
 [[pdf]](https://openreview.net/forum?id=P1DuPJzVTN) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Greedy%20Bayesian%20Posterior%20Approximation%20with%20Deep%20Ensembles.pdf)
- `TMLR, 2022`
- [Uncertainty Estimation]
```
Quite well-written and fairly interesting paper. I was mainly just interested in reading one of the first ever TMLR accepted papers. Their final method in Algorithm 2 makes some intuitive sense, but I did not fully understand the theoretical arguments in Section 3.
```

##### [22-06-10] [paper227]
- Weakly-Supervised Disentanglement Without Compromises
 [[pdf]](https://arxiv.org/abs/2002.02886) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Weakly-Supervised%20Disentanglement%20Without%20Compromises.pdf)
- `ICML 2020`
```
Quite well-written and somewhat interesting paper. Definitely not my area of expertise (learning disentangled representations of e.g. images) and I didn't have a lot of time to read the paper, I struggled to understand big parts of the paper.
```

##### [22-06-02] [paper226]
- Shaking the Foundations: Delusions in Sequence Models for Interaction and Control
 [[pdf]](https://arxiv.org/abs/2110.10819) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Shaking%20the%20foundations:%20delusions%20in%20sequence%20models%20for%20interaction%20and%20control.pdf)
- `2021-10`
```
Quite well-written and somewhat interesting paper. Definitely not my area of expertise (causality). I didn't understand everything properly, and it's very difficult for me to judge how interesting this paper actually is.
```

##### [22-05-23] [paper225]
- When are Bayesian Model Probabilities Overconfident?
 [[pdf]](https://arxiv.org/abs/2003.04026) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/When%20are%20Bayesian%20model%20probabilities%20overconfident%3F.pdf)
- `2020-03`
```
Quite well-written and somewhat interesting paper. A bit different compared to the papers I usually read, this is written by people doing statistics. I did definitely not understand everything properly. Quite difficult for me to say what my main practical takeaway from the paper is.
```

##### [22-05-20] [paper224]
- Open-Set Recognition: a Good Closed-Set Classifier is All You Need?
 [[pdf]](https://arxiv.org/abs/2110.06207) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Open-Set%20Recognition:%20a%20Good%20Closed-Set%20Classifier%20is%20All%20You%20Need%3F.pdf)
- `ICLR 2022`
- [Out-of-Distribution Detection]
```
Well-written and quite interesting paper. Like the authors discuss, this open-set recognition problem is of course highly related to out-of-distribution detection. Their proposed benchmark (fine-grained classification datasets) is quite neat, definitely a lot mote challenging than many OOD detection datasets (this could be seen as "very near ODD" I suppose).
```

##### [22-04-08] [paper223]
- Improving Conditional Coverage via Orthogonal Quantile Regression
 [[pdf]](https://proceedings.neurips.cc/paper/2021/file/1006ff12c465532f8c574aeaa4461b16-Paper.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Improving%20Conditional%20Coverage%20via%20Orthogonal%20Quantile%20Regression.pdf)
- `NeurIPS 2021`
- [Uncertainty Estimation]
```
Well-written and somewhat interesting paper. They propose an improved quantile regression method named orthogonal QR. The method entails adding a regularization term to the quantile regression loss, encouraging the prediction interval length to be independent of the coverage identifier (intuitively, I don't quite get why this is desired). They evaluate on 9 tabular regression datasets, the same used in e.g. "Conformalized Quantile Regression". The model is just a small 3-layer neural network. Compared to standard quantile regression, their method improves something called "conditional coverage" of the prediction intervals (they want to "achieve coverage closer to the desired level evenly across all sup-populations").
```

##### [22-04-08] [paper222]
- Conformalized Quantile Regression
 [[pdf]](https://proceedings.neurips.cc/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Conformalized%20Quantile%20Regression.pdf)
- `NeurIPS 2019`
- [Uncertainty Estimation]
```
Interesting and well-written paper. I should have read this paper before reading "Efficient and Differentiable Conformal Prediction with General Function Classes". They give a pretty good introduction to both quantile regression and conformal prediction, and then propose a method that combines these two approaches. Their method is quite simple, they use conformal prediction on validation data (the "calibration set") to calibrate the prediction intervals learned by a quantile regression method? This is sort of like temperature scaling, but for prediction intervals learned by quantile regression?
```

##### [22-04-08] [paper221]
- Efficient and Differentiable Conformal Prediction with General Function Classes
 [[pdf]](https://openreview.net/forum?id=Ht85_jyihxp) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Efficient%20and%20Differentiable%20Conformal%20Prediction%20with%20General%20Function%20Classes.pdf)
- `ICLR 2022`
- [Uncertainty Estimation]
```
Quite interesting and well-written paper. Mainly consider regression problems (tabular datasets + next-state prediction in RL, low-dimensional inputs). I should have read at least one more basic paper on conformal prediction and/or quantile regression first, I didn't quite understand all the details.
```

##### [22-04-06] [paper220]
- Consistent Estimators for Learning to Defer to an Expert
 [[pdf]](http://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Consistent%20Estimators%20for%20Learning%20to%20Defer%20to%20an%20Expert.pdf)
- `ICML 2020`
- [Uncertainty Estimation], [Medical ML]
```
Somewhat interesting paper. Image and text classification. The general problem setting (that a model can either predict or defer to an expert) is interesting and the paper is well-written overall, but in the end I can't really state any specific takeaways. I didn't understand section 4 or 5 properly. I don't think I can judge the significance of their results/contributions. 
```

##### [22-04-06] [paper219]
- Uncalibrated Models Can Improve Human-AI Collaboration
 [[pdf]](https://arxiv.org/abs/2202.05983) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncalibrated%20Models%20Can%20Improve%20Human-AI%20Collaboration.pdf)
- `2022-02`
- [Medical ML]
```
Quite interesting paper. Sort of thought-provoking, an interesting perspective. I was not exactly convinced in the end though. It seems weird to me that they don't even use an ML model to provide the advice, but instead use the average response of another group of human participants. Because this means that, like they write in Section 6, the average advice accuracy is higher than the average human accuracy. So, if the advice is better than the human participants, we just want to push the human predictions towards the advice? And therefore it's beneficial to increase the confidence of the advice (and thus make it uncalibrated), because this will make more humans actually change their prediction and align it more with the advice? I might miss something here, but this sort of seems a bit trivial?
```

##### [22-04-05] [paper218]
- Exploring Covariate and Concept Shift for Detection and Calibration of Out-of-Distribution Data
 [[pdf]](https://arxiv.org/abs/2110.15231) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Exploring%20Covariate%20and%20Concept%20Shift%20for%20Detection%20and%20Calibration%20of%20Out-of-Distribution%20Data.pdf)
- `2021-11`
- [Out-of-Distribution Detection]
```
Quite interesting and well-written paper. Only image classification (CIFAR10/100). I didn't quite spend enough time to properly understand everything in Section 4, or to really judge how impressive their experimental results actually are. Seems potentially useful.
```

##### [22-04-02] [paper217]
- On the Out-of-distribution Generalization of Probabilistic Image Modelling
 [[pdf]](https://arxiv.org/abs/2109.02639) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20the%20Out-of-distribution%20Generalization%20of%20Probabilistic%20Image%20Modelling.pdf)
- `NeurIPS 2021`
- [Out-of-Distribution Detection]
```
Well-written and interesting paper, I enjoyed reading it. Everything is clearly explained and the proposed OOD detection score in Section 3.1 makes intuitive sense. The results in Table 4 seem quite impressive. I was mostly interested in the OOD detection aspect, so I didn't read Section 4 too carefully.
```

##### [22-04-02] [paper216]
- A Fine-Grained Analysis on Distribution Shift
 [[pdf]](https://arxiv.org/abs/2110.11328) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Fine-Grained%20Analysis%20on%20Distribution%20Shift.pdf)
- `ICLR 2022`
- [Distribution Shifts]
```
Somewhat interesting paper. They consider 6 different datasets, only classification tasks. The takeaways and practical tips in Section 4 seem potentially useful, but I also find them somewhat vague.
```

##### [22-04-01] [paper215]
- Transformer-Based Out-of-Distribution Detection for Clinically Safe Segmentation
 [[pdf]](https://openreview.net/forum?id=En7660i-CLJ) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Transformer-Based%20Out-of-Distribution%20Detection%20for%20Clinically%20Safe%20Segmentation.pdf)
- `MIDL 2022`
- [Medical ML], [Out-of-Distribution Detection], [Transformers]
```
Well-written and interesting paper. I was not familiar with the VQ-GAN/VAE model, so I was confused by Section 2.3 at first, but now I think that I understand most of it. Their VQ-GAN + transformer approach seems quite complex indeed, but also seems to perform well. However, they didn't really compare with any other OOD detection method. I find it somewhat difficult to tell how useful this actually could be in practice.
```

##### [22-03-31] [paper214]
- Delving into Deep Imbalanced Regression
 [[pdf]](https://arxiv.org/abs/2102.09554) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Delving%20into%20Deep%20Imbalanced%20Regression.pdf)
- `ICML 2021`
```
Well-written and somewhat interesting paper. The "health condition score" estimation problem seems potentially interesting. They only consider problems with 1D regression targets. Their two proposed methods are clearly explained. I could probably encounter this imbalanced issue at some point, and then I'll keep this paper in mind.
```

##### [22-03-31] [paper213]
- Hidden in Plain Sight: Subgroup Shifts Escape OOD Detection
 [[pdf]](https://openreview.net/forum?id=aZgiUNye2Cz) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Hidden%20in%20Plain%20Sight:%20Subgroup%20Shifts%20Escape%20OOD%20Detection.pdf)
- `MIDL 2022`
- [Medical ML], [Out-of-Distribution Detection], [Distribution Shifts]
```
Quite well-written, but somewhat confusing paper. The experiment in Table 1 seems odd to me, why would we expect or even want digit-5 images to be classified as OOD when the training data actually includes a bunch of digit-5 images (the bottom row)? And for what they write in the final paragraph of Section 3 (that the accuracy is a bit lower for the hospital 3 subgroup), this wouldn't actually be a problem in practice if the model then also is more uncertain for these examples? I.e., studying model calibration across the different subgroups would be what's actually interesting? Or am I not understanding this whole subgroup shift properly? I feel quite confused.
```

##### [22-03-30] [paper212]
- Self-Distribution Distillation: Efficient Uncertainty Estimation
 [[pdf]](https://arxiv.org/abs/2203.08295) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Self-Distribution%20Distillation:%20Efficient%20Uncertainty%20Estimation.pdf)
- `2022-03`
- [Uncertainty Estimation], [Out-of-Distribution Detection]
```
Quite well-written and somewhat interesting paper. Only consider image classification. Their method in Figure 1 is in a way more interesting than I first realized, it's not entirely clear to me why this would improve performance compared to just training a model with the standard cross-entropy loss, their method induces some type of beneficial regularization? I didn't quite get the method described in Section 4.1.
```

##### [22-03-29] [paper211]
- A Benchmark with Decomposed Distribution Shifts for 360 Monocular Depth Estimation
 [[pdf]](https://openreview.net/pdf?id=6ksR7XSRuGB) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Benchmark%20with%20Decomposed%20Distribution%20Shifts%20for%20360%20Monocular%20Depth%20Estimation.pdf)
- `NeurIPS Workshops 2021`
- [Distribution Shifts]
```
Somewhat interesting paper. A short paper of just 4-5 pages. The provided dataset could be useful for comparing methods in terms of distribution shift robustness.
```

##### [22-03-28] [paper210]
- WILDS: A Benchmark of in-the-Wild Distribution Shifts
 [[pdf]](http://proceedings.mlr.press/v139/koh21a/koh21a.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/WILDS:%20A%20Benchmark%20of%20in-the-Wild%20Distribution%20Shifts.pdf)
- `ICML 2021`
- [Distribution Shifts]
```
Well-written and quite interesting paper. Neat benchmark with a diverse set of quite interesting datasets.
```

##### [22-03-24] [paper209]
- Random Synaptic Feedback Weights Support Error Backpropagation for Deep Learning
 [[pdf]](https://www.nature.com/articles/ncomms13276) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Random%20Synaptic%20Feedback%20Weights%20Support%20Error%20Backpropagation%20for%20Deep%20Learning.pdf)
- `Nature Communications, 2016`
- [Theoretical Properties of Deep Learning]
```
Definitely not my area of expertise, but still a quite interesting paper to read. The authors are interested in the question of how error propagation-based learning algorithms potentially might be utilized in the human brain. Backpropagation is one such algorithm and is highly effective, but it "involves a precise, symmetric backward connectivity pattern" (to compute the gradient update of the current layer weight matrix, the error is multiplied with the weight matrix W of the following layer), which apparently is thought to be impossible in the brain. The authors show that backpropagation can be simplified but still offer effective learning, their feedback alignment method instead make use of "fixed, random connectivity patterns" (replace the weight matrix W with a random matrix B). Their study thus "reveals much lower architectural constraints on what is required for error propagation across layers of neurons".
```

##### [22-03-17] [paper208]
- Comparing Elementary Cellular Automata Classifications with a Convolutional Neural Network
 [[pdf]](https://www.scitepress.org/Papers/2021/101600/101600.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Comparing%20Elementary%20Cellular%20Automata%20Classifications%20with%20a%20Convolutional%20Neural%20Network.pdf)
- `ICAART 2021`
```
I'm not familiar with "Cellular automata" at all, but still a somewhat interesting paper to read. I mostly understand what they're doing (previous papers have proposed different categorizations/groupings/classifications of ECAs, and in this paper they train CNNs to predict the classes assigned by these different ECA categorizations, to compare them), but I don't really know why it's interesting/useful.
```

##### [22-03-10] [paper207]
- Structure and Distribution Metric for Quantifying the Quality of Uncertainty: Assessing Gaussian Processes, Deep Neural Nets, and Deep Neural Operators for Regression
 [[pdf]](https://arxiv.org/abs/2203.04515) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Structure%20and%20Distribution%20Metric%20for%20Quantifying%20the%20Quality%20of%20Uncertainty:%20Assessing%20Gaussian%20Processes%2C%20Deep%20Neural%20Nets%2C%20and%20Deep%20Neural%20Operators%20for%20Regression.pdf)
- `2022-03`
- [Uncertainty Estimation]
```
Somewhat interesting paper, I didn't spend too much time on it. Just simply using the correlation between squared error and predicted variance makes some sense, I guess? I don't quite get what their NDIP metric in Section 2.2 will actually measure though? Also, I don't understand their studied application at all.
```

##### [22-03-10] [paper206]
- How to Measure Deep Uncertainty Estimation Performance and Which Models are Naturally Better at Providing It
 [[pdf]](https://openreview.net/forum?id=LK8bvVSw6rn) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/How%20to%20Measure%20Deep%20Uncertainty%20Estimation%20Performance%20and%20Which%20Models%20are%20Naturally%20Better%20at%20Providing%20It.pdf)
- `2021-10`
- [Uncertainty Estimation], [Out-of-Distribution Detection]
```
Quite interesting and well-written paper. They only study image classification. The E-AURC metric which is described in Appendix C should be equivalent to AUSE, I think? Quite interesting that knowledge distillation seems to rather consistently have a positive effect on the uncertainty estimation metrics, and that ViT models seem to perform very well compared to a lot of other architectures. Otherwise, I find it somewhat difficult to draw any concrete conclusions.
```

##### [22-03-10] [paper205]
- The Deep Bootstrap Framework: Good Online Learners are Good Offline Generalizers
 [[pdf]](https://arxiv.org/abs/2010.08127) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/The%20Deep%20Bootstrap%20Framework:%20Good%20Online%20Learners%20are%20Good%20Offline%20Generalizers.pdf)
- `2020-10, ICLR 2021`
- [Theoretical Properties of Deep Learning]
```
Well-written and quite interesting paper. I didn't take the time to try and really understand all the details, but a quite enjoyable read. The proposed framework seems to make some intuitive sense and lead to some fairly interesting observations/insights, but it's difficult for me to judge how significant it actually is.
```

##### [22-03-08] [paper204]
- Selective Regression Under Fairness Criteria
 [[pdf]](https://arxiv.org/abs/2110.15403) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Selective%20Regression%20Under%20Fairness%20Criteria.pdf)
- `2021-10`
- [Uncertainty Estimation], [Selective Prediction]
```
Well-written and somewhat interesting paper. Gives a pretty good introduction to the fair regression problem, Section 2 is very well-written. Quite interesting that it can be the case that while overall performance improves with decreased coverage, the performance for a minority sub-group is degraded. I didn't quite follow everything in Section 5, the methods seem a bit niche. I'm not overly impressed by the experiments either.
```

##### [22-03-08] [paper203]
- Risk-Controlled Selective Prediction for Regression Deep Neural Network Models
 [[pdf]](http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_WCCI_2020/IJCNN/Papers/N-20828.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Risk-Controlled%20Selective%20Prediction%20for%20Regression%20Deep%20Neural%20Network%20Models.pdf)
- `IJCNN 2020`
- [Uncertainty Estimation], [Selective Prediction]
```
Interesting and well-written paper. They take the method from "Selective Classification for Deep Neural Networks" and extend it to regression. I don't really understand the details of the lemmas/theorems, but otherwise everything is clearly explained.
```

##### [22-03-08] [paper202]
- Second Opinion Needed: Communicating Uncertainty in Medical Artificial Intelligence
 [[pdf]](https://www.nature.com/articles/s41746-020-00367-3.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Second%20Opinion%20Needed:%20Communicating%20Uncertainty%20in%20Medical%20Artificial%20Intelligence.pdf)
- `NPJ Digital Medicine, 2021`
- [Uncertainty Estimation], [Medical ML]
```
Well-written and quite interesting paper. A relatively short paper of just 4 pages. They give an overview of different uncertainty estimation techniques, and provide some intuitive examples and motivation for why uncertainty estimation is important/useful within medical applications. I quite enjoyed reading the paper.
```

##### [22-03-07] [paper201]
- Selective Classification for Deep Neural Networks
 [[pdf]](https://dl.acm.org/doi/pdf/10.5555/3295222.3295241) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Selective%20Classification%20for%20Deep%20Neural%20Networks.pdf)
- `NeurIPS 2017`
- [Uncertainty Estimation], [Selective Prediction]
```
Interesting and well-written paper, I enjoyed reading it. I don't really understand the lemma/theorem in Section 3, but everything is still clearly explained.
```

##### [22-03-05] [paper200]
- SelectiveNet: A Deep Neural Network with an Integrated Reject Option
 [[pdf]](https://arxiv.org/abs/1901.09192) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/SelectiveNet:%20A%20Deep%20Neural%20Network%20with%20an%20Integrated%20Reject%20Option.pdf)
- `2019-01, ICML 2019`
- [Uncertainty Estimation], [Selective Prediction]
```
Well-written and quite interesting paper. The proposed method is quite interesting and makes some intuitive sense, but I would assume that the calibration technique in Section 5 has similar issues as temperature scaling (i.e., the calibration might still break under various data shifts)?
```

##### [22-03-04] [paper199]
- NUQ: Nonparametric Uncertainty Quantification for Deterministic Neural Networks
 [[pdf]](https://arxiv.org/abs/2202.03101) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/NUQ:%20Nonparametric%20Uncertainty%20Quantification%20for%20Deterministic%20Neural%20Networks.pdf)
- `2022-02, NeurIPS 2022`
- [Uncertainty Estimation], [Out-of-Distribution Detection]
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
- [Graph Neural Networks]
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
- [Graph Neural Networks]

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
 [[pdf]](https://arxiv.org/abs/2003.06060) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Your%20GAN%20is%20Secretly%20an%20Energy-based%20Model%20and%20You%20Should%20Use%20Discriminator%20Driven%20Latent%20Sampling.pdf)
- *Tong Che, Ruixiang Zhang, Jascha Sohl-Dickstein, Hugo Larochelle, Liam Paull, Yuan Cao, Yoshua Bengio*
- `2020-03-12, NeurIPS 2020`
- [[Energy-Based Models]](#energy-based-models)

##### [21-03-19] [paper124]
- Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability
 [[pdf]](https://arxiv.org/abs/2103.00065) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Gradient%20Descent%20on%20Neural%20Networks%20Typically%20Occurs%20at%20the%20Edge%20of%20Stability.pdf)
- *Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, Ameet Talwalkar*
- `2021-02-26, ICLR 2021`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning)

##### [21-03-12] [paper123]
- Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
 [[pdf]](https://arxiv.org/abs/2006.09882) [[code]](https://github.com/facebookresearch/swav) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Unsupervised%20Learning%20of%20Visual%20Features%20by%20Contrasting%20Cluster%20Assignments.pdf)
- *Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, Armand Joulin*
- `2020-06-17, NeurIPS 2020`

##### [21-03-04] [paper122]
- Infinitely Deep Bayesian Neural Networks with Stochastic Differential Equations
 [[pdf]](https://arxiv.org/abs/2102.06559) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Infinitely%20Deep%20Bayesian%20Neural%20Networks%20with%20Stochastic%20Differential%20Equations.pdf)
- *Winnie Xu, Ricky T.Q. Chen, Xuechen Li, David Duvenaud*
- `2021-02-12`
- [[Neural ODEs]](#neural-odes) [[Uncertainty Estimation]](#uncertainty-estimation)

##### [21-02-26] [paper121]
- Neural Relational Inference for Interacting Systems
 [[pdf]](https://arxiv.org/abs/1802.04687) [[code]](https://github.com/ethanfetaya/NRI) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Relational%20Inference%20for%20Interacting%20Systems.pdf)
- *Thomas Kipf, Ethan Fetaya, Kuan-Chieh Wang, Max Welling, Richard Zemel*
- `2018-02-13, ICML 2018`

##### [21-02-19] [paper120]
- Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision
 [[pdf]](https://arxiv.org/abs/2102.05918) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Scaling%20Up%20Visual%20and%20Vision-Language%20Representation%20Learning%20With%20Noisy%20Text%20Supervision.pdf)
- *Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig*
- `2021-02-11, ICML 2021`

##### [21-02-12] [paper119]
- On the Origin of Implicit Regularization in Stochastic Gradient Descent
 [[pdf]](https://arxiv.org/abs/2101.12176) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20the%20Origin%20of%20Implicit%20Regularization%20in%20Stochastic%20Gradient%20Descent.pdf)
- *Samuel L. Smith, Benoit Dherin, David G. T. Barrett, Soham De*
- `2021-01-28, ICLR 2021`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning)

##### [21-02-05] [paper118]
- Meta Pseudo Labels
 [[pdf]](https://arxiv.org/abs/2003.10580) [[code]](https://github.com/google-research/google-research/tree/master/meta_pseudo_labels) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Meta%20Pseudo%20Labels.pdf)
- *Hieu Pham, Zihang Dai, Qizhe Xie, Minh-Thang Luong, Quoc V. Le*
- `2020-03-23, CVPR 2021`

##### [21-01-29] [paper117]
- No MCMC for Me: Amortized Sampling for Fast and Stable Training of Energy-Based Models
 [[pdf]](https://arxiv.org/abs/2010.04230) [[code]](https://github.com/wgrathwohl/VERA) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/No%20MCMC%20for%20me:%20Amortized%20sampling%20for%20fast%20and%20stable%20training%20of%20energy-based%20models.pdf)
- *Will Grathwohl, Jacob Kelly, Milad Hashemi, Mohammad Norouzi, Kevin Swersky, David Duvenaud*
- `2020-10-08, ICLR 2021`
- [[Energy-Based Models]](#energy-based-models)

##### [21-01-22] [paper116]
- Getting a CLUE: A Method for Explaining Uncertainty Estimates
 [[pdf]](https://arxiv.org/abs/2006.06848) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Getting%20a%20CLUE:%20A%20Method%20for%20Explaining%20Uncertainty%20Estimates.pdf)
- *Javier Antorán, Umang Bhatt, Tameem Adel, Adrian Weller, José Miguel Hernández-Lobato*
- `2020-06-11, ICLR 2021`
- [[Uncertainty Estimation]](#uncertainty-estimation)

##### [21-01-15] [paper115]
- Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
 [[pdf]](https://arxiv.org/abs/2006.16236) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Transformers%20are%20RNNs:%20Fast%20Autoregressive%20Transformers%20with%20Linear%20Attention.pdf)
- *Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret*
- `2020-06-29, ICML 2020`
- [[Transformers]](#transformers)

#### Papers Read in 2020:

##### [20-12-18] [paper114]
- Score-Based Generative Modeling through Stochastic Differential Equations
 [[pdf]](https://arxiv.org/abs/2011.13456) [[code]](https://github.com/yang-song/score_sde) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Score-Based%20Generative%20Modeling%20through%20Stochastic%20Differential%20Equations.pdf)
- *Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole*
- `2020-11-26, ICLR 2021`
- [[Neural ODEs]](#neural-odes)

##### [20-12-14] [paper113]
- Dissecting Neural ODEs
 [[pdf]](https://arxiv.org/abs/2002.08071) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Dissecting%20Neural%20ODEs.pdf)
- *Stefano Massaroli, Michael Poli, Jinkyoo Park, Atsushi Yamashita, Hajime Asama*
- `2020-02-19, NeurIPS 2020`
- [[Neural ODEs]](#neural-odes)

##### [20-11-27] [paper112]
- Rethinking Attention with Performers
 [[pdf]](https://arxiv.org/abs/2009.14794) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Rethinking%20Attention%20with%20Performers.pdf)
- *Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller*
- `2020-10-30, ICLR 2021`
- [[Transformers]](#transformers)

##### [20-11-23] [paper111]
- Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images
 [[pdf]](https://arxiv.org/abs/2011.10650) [[code]](https://github.com/openai/vdvae) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Very%20Deep%20VAEs%20Generalize%20Autoregressive%20Models%20and%20Can%20Outperform%20Them%20on%20Images.pdf)
- *Rewon Child*
- `2020-11-20, ICLR 2021`
- [[VAEs]](#vaes)

##### [20-11-13] [paper110]
- VAEBM: A Symbiosis between Variational Autoencoders and Energy-based Models
 [[pdf]](https://arxiv.org/abs/2010.00654) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/VAEBM:%20A%20Symbiosis%20between%20Variational%20Autoencoders%20and%20Energy-based%20Models.pdf)
- *Zhisheng Xiao, Karsten Kreis, Jan Kautz, Arash Vahdat*
- `2020-10-01, ICLR 2021`
- [[Energy-Based Models]](#energy-based-models) [[VAEs]](#vaes)

##### [20-11-06] [paper109]
- Approximate Inference Turns Deep Networks into Gaussian Processes
 [[pdf]](https://arxiv.org/abs/1906.01930) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Approximate%20Inference%20Turns%20Deep%20Networks%20into%20Gaussian%20Processes.pdf)
- *Mohammad Emtiyaz Khan, Alexander Immer, Ehsan Abedi, Maciej Korzepa*
- `2019-06-05, NeurIPS 2019`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning)

##### [20-10-16] [paper108]
- Implicit Gradient Regularization [[pdf]](https://arxiv.org/abs/2009.11162) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Implicit%20Gradient%20Regularization.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Implicit%20Gradient%20Regularization.md)
- *David G.T. Barrett, Benoit Dherin*
- `2020-09-23`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning)

##### [20-10-09] [paper107]
- Satellite Conjunction Analysis and the False Confidence Theorem [[pdf]](https://royalsocietypublishing.org/doi/pdf/10.1098/rspa.2018.0565) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Satellite%20conjunction%20analysis%20and%20the%20false%20confidence%20theorem.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Satellite%20Conjunction%20Analysis%20and%20the%20False%20Confidence%20Theorem.md)
- *Michael Scott Balch, Ryan Martin, Scott Ferson*
- `2018-03-21`

##### [20-09-24] [paper106]
- Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness [[pdf]](https://arxiv.org/abs/2006.10108) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Simple%20and%20Principled%20Uncertainty%20Estimation%20with%20Deterministic%20Deep%20Learning%20via%20Distance%20Awareness.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Simple%20and%20Principled%20Uncertainty%20Estimation%20with%20Deterministic%20Deep%20Learning%20via%20Distance%20Awareness.md)
- *Jeremiah Zhe Liu, Zi Lin, Shreyas Padhy, Dustin Tran, Tania Bedrax-Weiss, Balaji Lakshminarayanan*
- `2020-06-17, NeurIPS 2020`
- [[Uncertainty Estimation]](#uncertainty-estimation)

##### [20-09-21] [paper105]
- Uncertainty Estimation Using a Single Deep Deterministic Neural Network [[pdf]](https://arxiv.org/abs/2003.02037) [[code]](https://github.com/y0ast/deterministic-uncertainty-quantification) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20Estimation%20Using%20a%20Single%20Deep%20Deterministic%20Neural%20Network.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Uncertainty%20Estimation%20Using%20a%20Single%20Deep%20Deterministic%20Neural%20Network.md)
- *Joost van Amersfoort, Lewis Smith, Yee Whye Teh, Yarin Gal*
- `2020-03-04, ICML 2020`
- [[Uncertainty Estimation]](#uncertainty-estimation)

##### [20-09-11] [paper104]
- Gated Linear Networks [[pdf]](https://arxiv.org/abs/1910.01526) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Gated%20Linear%20Networks.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Gated%20Linear%20Networks.md)
- *Joel Veness, Tor Lattimore, David Budden, Avishkar Bhoopchand, Christopher Mattern, Agnieszka Grabska-Barwinska, Eren Sezener, Jianan Wang, Peter Toth, Simon Schmitt, Marcus Hutter*
- `2020-06-11`

##### [20-09-04] [paper103]
- Denoising Diffusion Probabilistic Models [[pdf]](https://arxiv.org/abs/2006.11239) [[code]](https://github.com/hojonathanho/diffusion) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Denoising%20Diffusion%20Probabilistic%20Models.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Denoising%20Diffusion%20Probabilistic%20Models.md)
- *Jonathan Ho, Ajay Jain, Pieter Abbeel*
- `20-06-19`
- [[Energy-Based Models]](#energy-based-models)

##### [20-06-18] [paper102]
- Joint Training of Variational Auto-Encoder and Latent Energy-Based Model [[pdf]](https://arxiv.org/abs/2006.06059) [[code]](https://hthth0801.github.io/jointLearning/) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Joint%20Training%20of%20Variational%20Auto-Encoder%20and%20Latent%20Energy-Based%20Model.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Joint%20Training%20of%20Variational%20Auto-Encoder%20and%20Latent%20Energy-Based%20Model.md)
- *Tian Han, Erik Nijkamp, Linqi Zhou, Bo Pang, Song-Chun Zhu, Ying Nian Wu*
- `2020-06-10, CVPR 2020`
- [[VAEs]](#vaes) [[Energy-Based Models]](#energy-based-models)

##### [20-06-12] [paper101]
- End-to-End Object Detection with Transformers [[pdf]](https://arxiv.org/abs/2005.12872) [[code]](https://github.com/facebookresearch/detr) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/End-to-End%20Object%20Detection%20with%20Transformers.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/End-to-End%20Object%20Detection%20with%20Transformers.md)
- *Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko*
- `2020-05-26, ECCV 2020`
- [[Object Detection]](#object-detection)

##### [20-06-05] [paper100]
- Efficient and Scalable Bayesian Neural Nets with Rank-1 Factors [[pdf]](https://arxiv.org/abs/2005.07186) [[code]](https://github.com/google/edward2) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Efficient%20and%20Scalable%20Bayesian%20Neural%20Nets%20with%20Rank-1%20Factors.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Efficient%20and%20Scalable%20Bayesian%20Neural%20Nets%20with%20Rank-1%20Factors.md)
- *Michael W. Dusenberry, Ghassen Jerfel, Yeming Wen, Yi-an Ma, Jasper Snoek, Katherine Heller, Balaji Lakshminarayanan, Dustin Tran*
- `2020-05-14, ICML 2020`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Variational Inference]](#variational-inference)

##### [20-05-27] [paper99]
- BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning [[pdf]](https://arxiv.org/abs/2002.06715) [[code]](https://github.com/google/edward2) [[video]](https://iclr.cc/virtual_2020/poster_Sklf1yrYDr.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/BatchEnsemble:%20An%20Alternative%20Approach%20to%20Efficient%20Ensemble%20and%20Lifelong%20Learning.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/BatchEnsemble:%20An%20Alternative%20Approach%20to%20Efficient%20Ensemble%20and%20Lifelong%20Learning.md)
- *Yeming Wen, Dustin Tran, Jimmy Ba*
- `2020-02-17, ICLR 2020`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Ensembling]](#ensembling)

##### [20-05-10] [paper98]
- Stable Neural Flows [[pdf]](https://arxiv.org/abs/2003.08063) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Stable%20Neural%20Flows%20.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Stable%20Neural%20Flows.md)
- *Stefano Massaroli, Michael Poli, Michelangelo Bin, Jinkyoo Park, Atsushi Yamashita, Hajime Asama*
- `2020-03-18`

##### [20-04-17] [paper97]
- How Good is the Bayes Posterior in Deep Neural Networks Really? [[pdf]](https://arxiv.org/abs/2002.02405) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/How%20Good%20is%20the%20Bayes%20Posterior%20in%20Deep%20Neural%20Networks%20Really%3F.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/How%20Good%20is%20the%20Bayes%20Posterior%20in%20Deep%20Neural%20Networks%20Really%3F.md)
- *Florian Wenzel, Kevin Roth, Bastiaan S. Veeling, Jakub Świątkowski, Linh Tran, Stephan Mandt, Jasper Snoek, Tim Salimans, Rodolphe Jenatton, Sebastian Nowozin*
- `2020-02-06`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Stochastic Gradient MCMC]](#stochastic-gradient-mcmc)

##### [20-04-09] [paper96]
- Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration [[pdf]](https://arxiv.org/abs/1910.12656) [[code]](https://github.com/dirichletcal/experiments_neurips) [[poster]](https://dirichletcal.github.io/documents/neurips2019/poster.pdf) [[slides]](https://dirichletcal.github.io/documents/neurips2019/slides.pdf) [[video]](https://dirichletcal.github.io/documents/neurips2019/video/Meelis_Ettekanne.mp4) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Beyond%20temperature%20scaling:%20Obtaining%20well-calibrated%20multiclass%20probabilities%20with%20Dirichlet%20calibration.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Beyond%20temperature%20scaling:%20Obtaining%20well-calibrated%20multiclass%20probabilities%20with%20Dirichlet%20calibration.md)
- *Meelis Kull, Miquel Perello-Nieto, Markus Kängsepp, Telmo Silva Filho, Hao Song, Peter Flach*
- `2019-10-28, NeurIPS 2019`
- [[Uncertainty Estimation]](#uncertainty-estimation)

##### [20-04-03] [paper95]
- Normalizing Flows: An Introduction and Review of Current Methods [[pdf]](https://arxiv.org/abs/1908.09257) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Normalizing%20Flows:%20An%20Introduction%20and%20Review%20of%20Current%20Methods.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Normalizing%20Flows:%20An%20Introduction%20and%20Review%20of%20Current%20Methods.md)
- *Ivan Kobyzev, Simon Prince, Marcus A. Brubaker*
- `2019-08-25`
- [[Normalizing Flows]](#normalizing-flows)

##### [20-03-27] [paper94]
- Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning [[pdf]](https://arxiv.org/abs/2002.06470) [[code]](https://github.com/bayesgroup/pytorch-ensembles) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Pitfalls%20of%20In-Domain%20Uncertainty%20Estimation%20and%20Ensembling%20in%20Deep%20Learning.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Pitfalls%20of%20In-Domain%20Uncertainty%20Estimation%20and%20Ensembling%20in%20Deep%20Learning.md)
- *Arsenii Ashukha, Alexander Lyzhov, Dmitry Molchanov, Dmitry Vetrov*
- `2020-02-15, ICLR 2020`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Ensembling]](#ensembling) [[Stochastic Gradient MCMC]](#stochastic-gradient-mcmc)

##### [20-03-26] [paper93]
- Conservative Uncertainty Estimation By Fitting Prior Networks [[pdf]](https://openreview.net/forum?id=BJlahxHYDS) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Conservative%20Uncertainty%20Estimation%20By%20Fitting%20Prior%20Networks.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Conservative%20Uncertainty%20Estimation%20By%20Fitting%20Prior%20Networks.md)
- *Kamil Ciosek, Vincent Fortuin, Ryota Tomioka, Katja Hofmann, Richard Turner*
- `2019-10-25, ICLR 2020`
- [[Uncertainty Estimation]](#uncertainty-estimation) 

##### [20-03-09] [paper92]
- Batch Normalization Biases Deep Residual Networks Towards Shallow Paths [[pdf]](https://arxiv.org/abs/2002.10444) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Batch%20Normalization%20Biases%20Deep%20Residual%20Networks%20Towards%20Shallow%20Path.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Batch%20Normalization%20Biases%20Deep%20Residual%20Networks%20Towards%20Shallow%20Paths.md)
- *Soham De, Samuel L. Smith*
- `2020-02-24`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning) 

##### [20-02-28] [paper91]
- Bayesian Deep Learning and a Probabilistic Perspective of Generalization [[pdf]](https://arxiv.org/abs/2002.08791) [[code]](https://github.com/izmailovpavel/understandingbdl) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Bayesian%20Deep%20Learning%20and%20a%20Probabilistic%20Perspective%20of%20Generalization.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Bayesian%20Deep%20Learning%20and%20a%20Probabilistic%20Perspective%20of%20Generalization.md)
- *Andrew Gordon Wilson, Pavel Izmailov*
- `2020-02-20`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Ensembling]](#ensembling) 

##### [20-02-21] [paper90]
- Convolutional Conditional Neural Processes [[pdf]](https://arxiv.org/abs/1910.13556) [[code]](https://github.com/cambridge-mlg/convcnp) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Convolutional%20Conditional%20Neural%20Processes.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Convolutional%20Conditional%20Neural%20Processes.md)
- *Jonathan Gordon, Wessel P. Bruinsma, Andrew Y. K. Foong, James Requeima, Yann Dubois, Richard E. Turner*
- `2019-10-29, ICLR 2020`
- [[Neural Processes]](#neural-processes)

##### [20-02-18] [paper89]
- Probabilistic 3D Multi-Object Tracking for Autonomous Driving [[pdf]](https://arxiv.org/abs/2001.05673) [[code]](https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Probabilistic%203D%20Multi-Object%20Tracking%20for%20Autonomous%20Driving.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Probabilistic%203D%20Multi-Object%20Tracking%20for%20Autonomous%20Driving.md)
- *Hsu-kuang Chiu, Antonio Prioletti, Jie Li, Jeannette Bohg*
- `2020-01-16`
- [[3D Multi-Object Tracking]](#3d-multi-object-tracking)

##### [20-02-15] [paper88]
- A Baseline for 3D Multi-Object Tracking [[pdf]](https://arxiv.org/abs/1907.03961) [[code]](https://github.com/xinshuoweng/AB3DMOT) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Baseline%20for%203D%20Multi-Object%20Tracking.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/A%20Baseline%20for%203D%20Multi-Object%20Tracking.md)
- *Xinshuo Weng, Kris Kitani*
- `2019-07-09`
- [[3D Multi-Object Tracking]](#3d-multi-object-tracking)

##### [20-02-14] [paper87]
- A Contrastive Divergence for Combining Variational Inference and MCMC [[pdf]](https://arxiv.org/abs/1905.04062) [[code]](https://github.com/franrruiz/vcd_divergence) [[slides]](https://franrruiz.github.io/contents/group_talks/EMS-Jul2019.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Contrastive%20Divergence%20for%20Combining%20Variational%20Inference%20and%20MCMC.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/A%20Contrastive%20Divergence%20for%20Combining%20Variational%20Inference%20and%20MCMC.md)
- *Francisco J. R. Ruiz, Michalis K. Titsias*
- `2019-05-10, ICML 2019`
- [[VAEs]](#vaes)

##### [20-02-13] [paper86]
- Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning [[pdf]](https://arxiv.org/abs/1710.07283) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Decomposition%20of%20Uncertainty%20in%20Bayesian%20Deep%20Learning%20for%20Efficient%20and%20Risk-sensitive%20Learning.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Decomposition%20of%20Uncertainty%20in%20Bayesian%20Deep%20Learning%20for%20Efficient%20and%20Risk-sensitive%20Learning.md)
- *Stefan Depeweg, José Miguel Hernández-Lobato, Finale Doshi-Velez, Steffen Udluft*
- `2017-10-19, ICML 2018`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Reinforcement Learning]](#reinforcement-learning)

##### [20-02-08] [paper85]
- Uncertainty Decomposition in Bayesian Neural Networks with Latent Variables [[pdf]](https://arxiv.org/abs/1706.08495) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20Decomposition%20in%20Bayesian%20Neural%20Networks%20with%20Latent%20Variables.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Uncertainty%20Decomposition%20in%20Bayesian%20Neural%20Networks%20with%20Latent%20Variables.md)
- *Stefan Depeweg, José Miguel Hernández-Lobato, Finale Doshi-Velez, Steffen Udluft*
- `2017-06-26`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Reinforcement Learning]](#reinforcement-learning)

##### [20-01-31] [paper84]
- Modelling heterogeneous distributions with an Uncountable Mixture of Asymmetric Laplacians [[pdf]](https://arxiv.org/abs/1910.12288) [[code]](https://github.com/BBVA/UMAL) [[video]](https://vimeo.com/369179175) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Modelling%20heterogeneous%20distributions%20with%20an%20Uncountable%20Mixture%20of%20Asymmetric%20Laplacians.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Modelling%20heterogeneous%20distributions%20with%20an%20Uncountable%20Mixture%20of%20Asymmetric%20Laplacians.md)
- *Axel Brando, Jose A. Rodríguez-Serrano, Jordi Vitrià, Alberto Rubio*
- `2019-10-27, NeurIPS 2019`
- [[Uncertainty Estimation]](#uncertainty-estimation)

##### [20-01-24] [paper83]
- A Primal-Dual link between GANs and Autoencoders [[pdf]](http://papers.nips.cc/paper/8333-a-primal-dual-link-between-gans-and-autoencoders) [[poster]](https://drive.google.com/file/d/1ifPldBOeuSa2Iwh3ESVGmRJQKzs9XgPv/view) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Primal-Dual%20link%20between%20GANs%20and%20Autoencoders.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/A%20Primal-Dual%20link%20between%20GANs%20and%20Autoencoders.md)
- *Hisham Husain, Richard Nock, Robert C. Williamson*
- `2019-04-26, NeurIPS 2019`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning)

##### [20-01-20] [paper82]
- A Connection Between Score Matching and Denoising Autoencoders [[pdf]](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport_1358.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Connection%20Between%20Score%20Matching%20and%20Denoising%20Autoencoders.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/A%20Connection%20Between%20Score%20Matching%20and%20Denoising%20Autoencoders.md)
- *Pascal Vincent*
- `2010-12`
- [[Energy-Based Models]](#energy-based-models)

##### [20-01-17] [paper81]
- Multiplicative Interactions and Where to Find Them [[pdf]](https://openreview.net/forum?id=rylnK6VtDH) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Multiplicative%20Interactions%20and%20Where%20to%20Find%20Them.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Multiplicative%20Interactions%20and%20Where%20to%20Find%20Them.md)
- *Siddhant M. Jayakumar, Jacob Menick, Wojciech M. Czarnecki, Jonathan Schwarz, Jack Rae, Simon Osindero, Yee Whye Teh, Tim Harley, Razvan Pascanu*
- `2019-09-25, ICLR 2020`
- [[Theoretical Properties of Deep Learning]](#theoretical-properties-of-deep-learning) [[Sequence Modeling]](#sequence-modeling)

##### [20-01-16] [paper80]
- Estimation of Non-Normalized Statistical Models by Score Matching [[pdf]](http://www.jmlr.org/papers/v6/hyvarinen05a.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Estimation%20of%20Non-Normalized%20Statistical%20Models%20by%20Score%20Matching.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Estimation%20of%20Non-Normalized%20Statistical%20Models%20by%20Score%20Matching.md)
- *Aapo Hyvärinen*
- `2004-11, JMLR 6`
- [[Energy-Based Models]](#energy-based-models)

##### [20-01-15] [paper79]
- Generative Modeling by Estimating Gradients of the Data Distribution [[pdf]](https://arxiv.org/abs/1907.05600) [[code]](https://github.com/ermongroup/ncsn) [[poster]](https://yang-song.github.io/papers/NeurIPS2019/ncsn-poster.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Generative%20Modeling%20by%20Estimating%20Gradients%20of%20the%20Data%20Distribution.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Generative%20Modeling%20by%20Estimating%20Gradients%20of%20the%20Data%20Distribution.md)
- *Yang Song, Stefano Ermon*
- `2019-07-12, NeurIPS 2019`
- [[Energy-Based Models]](#energy-based-models)

##### [20-01-14] [paper78]
- Noise-contrastive estimation: A new estimation principle for unnormalized statistical models [[pdf]](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Noise-contrastive%20estimation:%20A%20new%20estimation%20principle%20for%20unnormalized%20statistical%20models.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Noise-contrastive%20estimation:%20A%20new%20estimation%20principle%20for%20unnormalized%20statistical%20models.md)
- *Michael Gutmann, Aapo Hyvärinen*
- `2009, AISTATS 2010`
- [[Energy-Based Models]](#energy-based-models)

##### [20-01-10] [paper77]
- Z-Forcing: Training Stochastic Recurrent Networks [[pdf]](https://arxiv.org/abs/1711.05411) [[code]](https://github.com/sordonia/zforcing) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Z-Forcing:%20Training%20Stochastic%20Recurrent%20Networks.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Z-Forcing:%20Training%20Stochastic%20Recurrent%20Networks.md)
- *Anirudh Goyal, Alessandro Sordoni, Marc-Alexandre Côté, Nan Rosemary Ke, Yoshua Bengio*
- `2017-11-15, NeurIPS 2017`
- [[VAEs]](#vaes) [[Sequence Modeling]](#sequence-modeling)

##### [20-01-08] [paper76]
- Practical Deep Learning with Bayesian Principles [[pdf]](https://arxiv.org/abs/1906.02506) [[code]](https://github.com/team-approx-bayes/dl-with-bayes) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Practical%20Deep%20Learning%20with%20Bayesian%20Principles.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Practical%20Deep%20Learning%20with%20Bayesian%20Principles.md)
- *Kazuki Osawa, Siddharth Swaroop, Anirudh Jain, Runa Eschenhagen, Richard E. Turner, Rio Yokota, Mohammad Emtiyaz Khan*
- `2019-06-06, NeurIPS 2019`
- [[Uncertainty Estimation]](#uncertainty-estimation) [[Variational Inference]](#variational-inference)

##### [20-01-06] [paper75]
- Maximum Entropy Generators for Energy-Based Models [[pdf]](https://arxiv.org/abs/1901.08508) [[code]](https://github.com/ritheshkumar95/energy_based_generative_models) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Maximum%20Entropy%20Generators%20for%20Energy-Based%20Models.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Maximum%20Entropy%20Generators%20for%20Energy-Based%20Models.md)
- *Rithesh Kumar, Sherjil Ozair, Anirudh Goyal, Aaron Courville, Yoshua Bengio*
- `2019-01-24`
- [[Energy-Based Models]](#energy-based-models)

#### Papers Read in 2019:

##### [19-12-22] [paper74]
- Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One [[pdf]](https://arxiv.org/abs/1912.03263) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Your%20Classifier%20is%20Secretly%20an%20Energy%20Based%20Model%20and%20You%20Should%20Treat%20it%20Like%20One.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Your%20Classifier%20is%20Secretly%20an%20Energy%20Based%20Model%20and%20You%20Should%20Treat%20it%20Like%20One.md)
- *Will Grathwohl, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, Kevin Swersky*
- `2019-12-06, ICLR 2020`
- [[Energy-Based Models]](#energy-based-models)

##### [19-12-20] [paper73]
- Noise Contrastive Estimation and Negative Sampling for Conditional Models: Consistency and Statistical Efficiency [[pdf]](https://arxiv.org/abs/1809.01812) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Noise%20Contrastive%20Estimation%20and%20Negative%20Sampling%20for%20Conditional%20Models:%20Consistency%20and%20Statistical%20Efficiency.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Noise%20Contrastive%20Estimation%20and%20Negative%20Sampling%20for%20Conditional%20Models:%20Consistency%20and%20Statistical%20Efficiency.md)
- *Zhuang Ma, Michael Collins*
- `2018-09-06, EMNLP 2018`
- [[Energy-Based Models]](#energy-based-models)

##### [19-12-20] [paper72]
- Flow Contrastive Estimation of Energy-Based Models [[pdf]](https://arxiv.org/abs/1912.00589) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Flow%20Contrastive%20Estimation%20of%20Energy-Based%20Models.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Flow%20Contrastive%20Estimation%20of%20Energy-Based%20Models.md)
- *Ruiqi Gao, Erik Nijkamp, Diederik P. Kingma, Zhen Xu, Andrew M. Dai, Ying Nian Wu*
- `2019-12-02, CVPR 2020`
- [[Energy-Based Models]](#energy-based-models) [[Normalizing Flows]](#normalizing-flows)

##### [19-12-19] [paper71]
- On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models [[pdf]](https://arxiv.org/abs/1903.12370) [[code]](https://github.com/point0bar1/ebm-anatomy)  [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20the%20Anatomy%20of%20MCMC-Based%20Maximum%20Likelihood%20Learning%20of%20Energy-Based%20Models.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/On%20the%20Anatomy%20of%20MCMC-Based%20Maximum%20Likelihood%20Learning%20of%20Energy-Based%20Models.md)
- *Erik Nijkamp, Mitch Hill, Tian Han, Song-Chun Zhu, Ying Nian Wu*
- `2019-04-29, AAAI 2020`
- [[Energy-Based Models]](#energy-based-models)

##### [19-12-15] [paper70]
- Implicit Generation and Generalization in Energy-Based Models [[pdf]](https://arxiv.org/abs/1903.08689) [[code]](https://github.com/openai/ebm_code_release) [[blog]](https://openai.com/blog/energy-based-models/) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Implicit%20Generation%20and%20Generalization%20in%20Energy-Based%20Models.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Implicit%20Generation%20and%20Generalization%20in%20Energy-Based%20Models.md)
- *Yilun Du, Igor Mordatch*
- `2019-04-20, NeurIPS 2019`
- [[Energy-Based Models]](#energy-based-models)

##### [19-12-14] [paper69]
- Learning Non-Convergent Non-Persistent Short-Run MCMC Toward Energy-Based Model [[pdf]](https://arxiv.org/abs/1904.09770) [[poster]](https://neurips.cc/Conferences/2019/Schedule?showEvent=13661) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20Non-Convergent%20Non-Persistent%20Short-Run%20MCMC%20Toward%20Energy-Based%20Model.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Learning%20Non-Convergent%20Non-Persistent%20Short-Run%20MCMC%20Toward%20Energy-Based%20Model.md)
- *Erik Nijkamp, Mitch Hill, Song-Chun Zhu, Ying Nian Wu*
- `2019-04-22, NeurIPS 2019`
- [[Energy-Based Models]](#energy-based-models)

##### [19-12-13] [paper68]
- A Tutorial on Energy-Based Learning [[pdf]](http://yann.lecun.com/exdb/publis/orig/lecun-06.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Tutorial%20on%20Energy-Based%20Learning.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/A%20Tutorial%20on%20Energy-Based%20Learning.md)
- *Yann LeCun, Sumit Chopra, Raia Hadsell, Marc Aurelio Ranzato, Fu Jie Huang*
- `2006-08-19`
- [[Energy-Based Models]](#energy-based-models)

##### [19-11-29] [paper67]
- Dream to Control: Learning Behaviors by Latent Imagination [[pdf]](https://openreview.net/forum?id=S1lOTC4tDS) [[webpage]](https://dreamrl.github.io/) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Dream%20to%20Control%20Learning%20Behaviors%20by%20Latent%20Imagination.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Dream%20to%20Control:%20Learning%20Behaviors%20by%20Latent%20Imagination.md)
- *Anonymous*
- `2019-09`

##### [19-11-26] [paper66]
- Deep Latent Variable Models for Sequential Data [[pdf]](https://backend.orbit.dtu.dk/ws/portalfiles/portal/160548008/phd475_Fraccaro_M.pdf)  [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Latent%20Variable%20Models%20for%20Sequential%20Data.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Deep%20Latent%20Variable%20Models%20for%20Sequential%20Data.md)
- *Marco Fraccaro*
- `2018-04-13, PhD Thesis`

##### [19-11-22] [paper65]
- Learning Latent Dynamics for Planning from Pixels [[pdf]](https://arxiv.org/abs/1811.04551) [[code]](https://github.com/google-research/planet) [[blog]](https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20Latent%20Dynamics%20for%20Planning%20from%20Pixels.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Learning%20Latent%20Dynamics%20for%20Planning%20from%20Pixels.md)
- *Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson*
- `2018-11-12, ICML2019`

##### [19-10-28] [paper64]
- Learning nonlinear state-space models using deep autoencoders [[pdf]](http://cse.lab.imtlucca.it/~bemporad/publications/papers/cdc18-autoencoders.pdf)  [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20nonlinear%20state-space%20models%20using%20deep%20autoencoders.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Learning%20nonlinear%20state-space%20models%20using%20deep%20autoencoders.md)
- *Daniele Masti, Alberto Bemporad*
- `2018, CDC2018`

##### [19-10-18] [paper63]
- Improving Variational Inference with Inverse Autoregressive Flow [[pdf]](https://arxiv.org/abs/1606.04934) [[code]](https://github.com/openai/iaf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Improving%20Variational%20Inference%20with%20Inverse%20Autoregressive%20Flow.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Improving%20Variational%20Inference%20with%20Inverse%20Autoregressive%20Flow.md)
- *Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling*
- `2016-06-15, NeurIPS2016`

##### [19-10-11] [paper62]
- Variational Inference with Normalizing Flows [[pdf]](https://arxiv.org/abs/1505.05770) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Variational%20Inference%20with%20Normalizing%20Flows.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Variational%20Inference%20with%20Normalizing%20Flows.md)
- *Danilo Jimenez Rezende, Shakir Mohamed*
- `2015-05-21, ICML2015`

##### [19-10-04] [paper61]
- Trellis Networks for Sequence Modeling [[pdf]](https://arxiv.org/abs/1810.06682) [[code]](https://github.com/locuslab/trellisnet) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Trellis%20Networks%20for%20Sequence%20Modeling.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Trellis%20Networks%20for%20Sequence%20Modeling.md)
- *Shaojie Bai, J. Zico Kolter, Vladlen Koltun*
- `2018-10-15, ICLR2019`

##### [19-07-11] [paper60]
- Part-A^2 Net: 3D Part-Aware and Aggregation Neural Network for Object Detection from Point Cloud [[pdf]](https://arxiv.org/abs/1907.03670) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Part-A%5E2%20Net:%203D%20Part-Aware%20and%20Aggregation%20Neural%20Network%20for%20Object%20Detection%20from%20Point%20Cloud.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Part-A%5E2%20Net:%203D%20Part-Aware%20and%20Aggregation%20Neural%20Network%20for%20Object%20Detection%20from%20Point%20Cloud.md)
- *Shaoshuai Shi, Zhe Wang, Xiaogang Wang, Hongsheng Li*
- `2019-07-08`

##### [19-07-10] [paper59]
- PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud [[pdf]](https://arxiv.org/abs/1812.04244) [[code]](https://github.com/sshaoshuai/PointRCNN) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/PointRCNN:%203D%20Object%20Proposal%20Generation%20and%20Detection%20from%20Point%20Cloud.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/PointRCNN:%203D%20Object%20Proposal%20Generation%20and%20Detection%20from%20Point%20Cloud.md)
- *Shaoshuai Shi, Xiaogang Wang, Hongsheng Li*
- `2018-12-11, CVPR2019`

##### [19-07-03] [paper58]
- Objects as Points [[pdf]](https://arxiv.org/abs/1904.07850) [[code]](https://github.com/xingyizhou/CenterNet) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Objects%20as%20Points.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Objects%20as%20Points.md)
- *Xingyi Zhou, Dequan Wang, Philipp Krähenbühl*
- `2019-04-16`

##### [19-06-12] [paper57]
- ATOM: Accurate Tracking by Overlap Maximization [[pdf]](https://arxiv.org/abs/1811.07628) [[code]](https://github.com/visionml/pytracking) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/ATOM:%20Accurate%20Tracking%20by%20Overlap%20Maximization.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/ATOM:%20Accurate%20Tracking%20by%20Overlap%20Maximization.md)
- *Martin Danelljan, Goutam Bhat, Fahad Shahbaz Khan, Michael Felsberg*
- `2018-11-19, CVPR2019`

##### [19-06-12] [paper56]
- Acquisition of Localization Confidence for Accurate Object Detection [[pdf]](https://arxiv.org/abs/1807.11590) [[code]](https://github.com/vacancy/PreciseRoIPooling) [[oral presentation]](https://youtu.be/SNCsXOFr_Ug) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Acquisition%20of%20Localization%20Confidence%20for%20Accurate%20Object%20Detection.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Acquisition%20of%20Localization%20Confidence%20for%20Accurate%20Object%20Detection.md)
- *Borui Jiang, Ruixuan Luo, Jiayuan Mao, Tete Xiao, Yuning Jiang*
- `2018-07-30, ECCV2018`

##### [19-06-05] [paper55]
- LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving [[pdf]](https://arxiv.org/abs/1903.08701) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/LaserNet:%20An%20Efficient%20Probabilistic%203D%20Object%20Detector%20for%20Autonomous%20Driving.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/LaserNet:%20An%20Efficient%20Probabilistic%203D%20Object%20Detector%20for%20Autonomous%20Driving.md)
- *Gregory P. Meyer, Ankit Laddha, Eric Kee, Carlos Vallespi-Gonzalez, Carl K. Wellington*
- `2019-03-20, CVPR2019`

##### [19-05-29] [paper54]
- Attention Is All You Need [[pdf]](https://arxiv.org/abs/1706.03762) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Attention%20Is%20All%20You%20Need.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Attention%20Is%20All%20You%20Need.md)
- *Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin*
- `2017-06-12, NeurIPS2017`

##### [19-04-05] [paper53]
- Stochastic Gradient Descent as Approximate Bayesian Inference [[pdf]](https://arxiv.org/abs/1704.04289) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Stochastic%20Gradient%20Descent%20as%20Approximate%20Bayesian%20Inference.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Stochastic%20Gradient%20Descent%20as%20Approximate%20Bayesian%20Inference.md)
- *Stephan Mandt, Matthew D. Hoffman, David M. Blei*
- `2017-04-13, Journal of Machine Learning Research 18 (2017)`

##### [19-03-29] [paper52]
- Generating High Fidelity Images with Subscale Pixel Networks and Multidimensional Upscaling [[pdf]](https://openreview.net/forum?id=HylzTiC5Km) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/GENERATING%20HIGH%20FIDELITY%20IMAGES%20WITH%20SUBSCALE%20PIXEL%20NETWORKS%20AND%20MULTIDIMENSIONAL%20UPSCALING.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Generating%20High%20Fidelity%20Images%20with%20Subscale%20Pixel%20Networks%20and%20Multidimensional%20Upscaling.md)
- *Jacob Menick, Nal Kalchbrenner*
- `2018-12-04, ICLR2019`

##### [19-03-15] [paper51]
- A recurrent neural network without chaos [[pdf]](https://arxiv.org/abs/1612.06212) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20recurrent%20neural%20network%20without%20chaos.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/A%20recurrent%20neural%20network%20without%20chaos.md)
- *Thomas Laurent, James von Brecht*
- `2016-12-19, ICLR2017`

##### [19-03-11] [paper50]
- Auto-Encoding Variational Bayes [[pdf]](https://arxiv.org/abs/1312.6114)
- *Diederik P Kingma, Max Welling*
- `2014-05-01, ICLR2014`

##### [19-03-04] [paper49]
- Coupled Variational Bayes via Optimization Embedding [[pdf]](https://papers.nips.cc/paper/8177-coupled-variational-bayes-via-optimization-embedding.pdf) [[poster]](http://wyliu.com/papers/LiuNIPS18_CVB_poster.pdf) [[code]](https://github.com/Hanjun-Dai/cvb) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Coupled%20Variational%20Bayes%20via%20Optimization%20Embedding.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Coupled%20Variational%20Bayes%20via%20Optimization%20Embedding.md)
- *Bo Dai, Hanjun Dai, Niao He, Weiyang Liu, Zhen Liu, Jianshu Chen, Lin Xiao, Le Song*
- `NeurIPS2018`

##### [19-03-01] [paper48]
- Language Models are Unsupervised Multitask Learners [[pdf]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) [[blog post]](https://blog.openai.com/better-language-models/) [[code]](https://github.com/openai/gpt-2) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Language%20Models%20are%20Unsupervised%20Multitask%20Learners.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Language%20Models%20are%20Unsupervised%20Multitask%20Learners.md)
- *Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever*
- `2019-02-14`

##### [19-02-27] [paper47]
- Predictive Uncertainty Estimation via Prior Networks [[pdf]](https://arxiv.org/abs/1802.10501) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Predictive%20Uncertainty%20Estimation%20via%20Prior%20Networks.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Predictive%20Uncertainty%20Estimation%20via%20Prior%20Networks.md)
- *Andrey Malinin, Mark Gales*
- `2018-02-28, NeurIPS2018`

##### [19-02-25] [paper46]
- Evaluating model calibration in classification [[pdf]](https://arxiv.org/abs/1902.06977) [[code]](https://github.com/uu-sml/calibration) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Evaluating%20model%20calibration%20in%20classification.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Evaluating%20model%20calibration%20in%20classification.md)
- *Juozas Vaicenavicius, David Widmann, Carl Andersson, Fredrik Lindsten, Jacob Roll, Thomas B. Schön*
- `2019-02-19, AISTATS2019`

##### [19-02-22] [paper45]
- Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks [[pdf]](https://arxiv.org/abs/1901.08584) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Fine-Grained%20Analysis%20of%20Optimization%20and%20Generalization%20for%20Overparameterized%20Two-Layer%20Neural%20Networks.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Fine-Grained%20Analysis%20of%20Optimization%20and%20Generalization%20for%20Overparameterized%20Two-Layer%20Neural%20Networks.md)
- *Sanjeev Arora, Simon S. Du, Wei Hu, Zhiyuan Li, Ruosong Wang*
- `2019-01-24`

##### [19-02-17] [paper44]
- Visualizing the Loss Landscape of Neural Nets [[pdf]](https://arxiv.org/abs/1712.09913) [[code]](https://github.com/tomgoldstein/loss-landscape) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Visualizing%20the%20Loss%20Landscape%20of%20Neural%20Nets.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Visualizing%20the%20Loss%20Landscape%20of%20Neural%20Nets.md)
- *Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein*
- `2017-12-28, NeurIPS2018`

##### [19-02-14] [paper43]
-  A Simple Baseline for Bayesian Uncertainty in Deep Learning [[pdf]](https://arxiv.org/abs/1902.02476) [[code]](https://github.com/wjmaddox/swa_gaussian) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Simple%20Baseline%20for%20Bayesian%20Uncertainty%20in%20Deep%20Learning.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/A%20Simple%20Baseline%20for%20Bayesian%20Uncertainty%20in%20Deep%20Learning.md)
- *Wesley Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, Andrew Gordon Wilson*
- `2019-02-07`

##### [19-02-13] [paper42]
-  Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning [[pdf]](https://arxiv.org/abs/1902.03932) [[code]](https://github.com/ruqizhang/csgmcmc) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Cyclical%20Stochastic%20Gradient%20MCMC%20for%20Bayesian%20Deep%20Learning.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Cyclical%20Stochastic%20Gradient%20MCMC%20for%20Bayesian%20Deep%20Learning.md)
- *Ruqi Zhang, Chunyuan Li, Jianyi Zhang, Changyou Chen, Andrew Gordon Wilson*
- `2019-02-11`

##### [19-02-12] [paper41]
-  Bayesian Dark Knowledge [[pdf]](https://arxiv.org/abs/1506.04416) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Bayesian%20Dark%20Knowledge.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Bayesian%20Dark%20Knowledge.md)
- *Anoop Korattikara, Vivek Rathod, Kevin Murphy, Max Welling*
- `2015-06-07, NeurIPS2015`

##### [19-02-07] [paper40]
- Noisy Natural Gradient as Variational Inference [[pdf]](https://arxiv.org/abs/1712.02390) [[video]](https://youtu.be/bWItvHYqKl8) [[code]](https://github.com/pomonam/NoisyNaturalGradient) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Noisy%20Natural%20Gradient%20as%20Variational%20Inference.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Noisy%20Natural%20Gradient%20as%20Variational%20Inference.md)
- *Guodong Zhang, Shengyang Sun, David Duvenaud, Roger Grosse*
- `2017-12-06, ICML2018`

##### [19-02-06] [paper39]
- Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks [[pdf]](https://arxiv.org/abs/1502.05336) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Probabilistic%20Backpropagation%20for%20Scalable%20Learning%20of%20Bayesian%20Neural%20Networks.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Probabilistic%20Backpropagation%20for%20Scalable%20Learning%20of%20Bayesian%20Neural%20Networks.md)
- *José Miguel Hernández-Lobato, Ryan P. Adams*
- `2015-07-15, ICML2015`

##### [19-02-05] [paper38]
- Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models [[pdf]](https://arxiv.org/abs/1805.12114) [[poster]](https://kchua.github.io/misc/poster.pdf) [[video]](https://youtu.be/3d8ixUMSiL8) [[code]](https://github.com/kchua/handful-of-trials) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Reinforcement%20Learning%20in%20a%20Handful%20of%20Trials%20using%20Probabilistic%20Dynamics%20Models.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Deep%20Reinforcement%20Learning%20in%20a%20Handful%20of%20Trials%20using%20Probabilistic%20Dynamics%20Models.md)
- *Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine*
- `2018-05-30, NeurIPS2018`

##### [19-01-28] [paper37]
- Practical Variational Inference for Neural Networks [[pdf]](https://www.cs.toronto.edu/~graves/nips_2011.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Practical%20Variational%20Inference%20for%20Neural%20Networks.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Practical%20Variational%20Inference%20for%20Neural%20Networks.md)
- *Alex Graves*
- `NeurIPS2011`

##### [19-01-27] [paper36]
- Weight Uncertainty in Neural Networks [[pdf]](https://arxiv.org/abs/1505.05424) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Weight%20Uncertainty%20in%20Neural%20Networks.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Weight%20Uncertainty%20in%20Neural%20Networks.md)
- *Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra*
- `2015-05-20, ICML2015`

##### [19-01-26] [paper35]
- Learning Weight Uncertainty with Stochastic Gradient MCMC for Shape Classification [[pdf]](http://people.duke.edu/~cl319/doc/papers/dbnn_shape_cvpr.pdf)  [[poster]](https://zhegan27.github.io/Papers/dbnn_shape_poster.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20Weight%20Uncertainty%20with%20Stochastic%20Gradient%20MCMC%20for%20Shape%20Classification.pdf) [[comments]](https://github.com/fregu856/papers/blob/master/summaries/Learning%20Weight%20Uncertainty%20with%20Stochastic%20Gradient%20MCMC%20for%20Shape%20Classification.md)
- *Chunyuan Li, Andrew Stevens, Changyou Chen, Yunchen Pu, Zhe Gan, Lawrence Carin*
- `CVPR2016`

##### [19-01-25] [paper34]
- Meta-Learning For Stochastic Gradient MCMC [[pdf]](https://openreview.net/forum?id=HkeoOo09YX) [[code]](https://github.com/WenboGong/MetaSGMCMC) [[slides]](http://yingzhenli.net/home/pdf/uai_udl_meta_sampler.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Meta-Learning%20For%20Stochastic%20Gradient%20MCMC.pdf)
- *Wenbo Gong, Yingzhen Li, José Miguel Hernández-Lobato*
- `2018-10-28, ICLR2019`

##### [19-01-25] [paper33]
-  A Complete Recipe for Stochastic Gradient MCMC [[pdf]](https://arxiv.org/abs/1506.04696) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Complete%20Recipe%20for%20Stochastic%20Gradient%20MCMC.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/A%20Complete%20Recipe%20for%20Stochastic%20Gradient%20MCMC.md)
- *Yi-An Ma, Tianqi Chen, Emily B. Fox*
- `2015-06-15, NeurIPS2015`

##### [19-01-24] [paper32]
- Tutorial: Introduction to Stochastic Gradient Markov Chain Monte Carlo Methods [[pdf]](https://cse.buffalo.edu/~changyou/PDF/sgmcmc_intro_without_video.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Tutorial:%20Introduction%20to%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo%20Methods.pdf)
- *Changyou Chen*
- `2016-08-10`

##### [19-01-24] [paper31]
- An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling [[pdf]](https://arxiv.org/abs/1803.01271) [[code]](https://github.com/locuslab/TCN) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/An%20Empirical%20Evaluation%20of%20Generic%20Convolutional%20and%20Recurrent%20Networks%20for%20Sequence%20Modeling.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/An%20Empirical%20Evaluation%20of%20Generic%20Convolutional%20and%20Recurrent%20Networks%20for%20Sequence%20Modeling.md)
- *Shaojie Bai, J. Zico Kolter, Vladlen Koltun*
- `2018-04-19`

##### [19-01-23] [paper30]
- Stochastic Gradient Hamiltonian Monte Carlo [[pdf]](https://arxiv.org/abs/1402.4102) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Stochastic%20Gradient%20Hamiltonian%20Monte%20Carlo.pdf)
- *Tianqi Chen, Emily B. Fox, Carlos Guestrin*
- `2014-05-12, ICML2014`

##### [19-01-23] [paper29]
- Bayesian Learning via Stochastic Gradient Langevin Dynamics [[pdf]](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Bayesian%20Learning%20via%20Stochastic%20Gradient%20Langevin%20Dynamics.pdf)
- *Max Welling, Yee Whye Teh*
- `ICML2011`

##### [19-01-17] [paper28]
- How Does Batch Normalization Help Optimization? [[pdf]](https://arxiv.org/abs/1805.11604) [[poster]](http://people.csail.mit.edu/tsipras/batchnorm_poster.pdf) [[video]](https://youtu.be/ZOabsYbmBRM) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/How%20Does%20Batch%20Normalization%20Help%20Optimization%3F.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/How%20Does%20Batch%20Normalization%20Help%20Optimization%3F.md)
- *Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, Aleksander Madry*
- `2018-10-27, NeurIPS2018`

##### [19-01-09] [paper27]
- Relaxed Softmax: Efficient Confidence Auto-Calibration for Safe Pedestrian Detection [[pdf]](https://openreview.net/forum?id=S1lG7aTnqQ) [[poster]](http://www.robots.ox.ac.uk/~vgg/publications/2018/Neumann18c/poster.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Relaxed%20Softmax:%20Efficient%20Confidence%20Auto-Calibration%20for%20Safe%20Pedestrian%20Detection.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Relaxed%20Softmax:%20Efficient%20Confidence%20Auto-Calibration%20for%20Safe%20Pedestrian%20Detection.md)
- *Lukas Neumann, Andrew Zisserman, Andrea Vedaldi*
- `2018-11-29, NeurIPS2018 Workshop`

#### Papers Read in 2018:

##### [18-12-12] [paper26]
- Neural Ordinary Differential Equations [[pdf]](https://arxiv.org/abs/1806.07366) [[code]](https://github.com/rtqichen/torchdiffeq) [[slides]](https://www.cs.toronto.edu/~duvenaud/talks/ode-talk-google.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Ordinary%20Differential%20Equations.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Neural%20Ordinary%20Differential%20Equations.md)
- *Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud*
- `2018-10-22, NeurIPS2018`
- [[Neural ODEs]](#neural-odes)

##### [18-12-06] [paper25]
- Evaluating Bayesian Deep Learning Methods for Semantic Segmentation [[pdf]](https://arxiv.org/abs/1811.12709) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Evaluating%20Bayesian%20Deep%20Learning%20Methods%20for%20Semantic%20Segmentation.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Evaluating%20Bayesian%20Deep%20Learning%20Methods%20for%20Semantic%20Segmentation.md)
- *Jishnu Mukhoti, Yarin Gal*
- `2018-11-30`

##### [18-12-05] [paper24]
- On Calibration of Modern Neural Networks [[pdf]](https://arxiv.org/abs/1706.04599) [[code]](https://github.com/gpleiss/temperature_scaling) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20Calibration%20of%20Modern%20Neural%20Networks.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/On%20Calibration%20of%20Modern%20Neural%20Networks.md)
- *Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger*
- `2017-08-03, ICML2017`

##### [18-11-29] [paper23]
-  Evidential Deep Learning to Quantify Classification Uncertainty [[pdf]](https://arxiv.org/abs/1806.01768) [[poster]](https://muratsensoy.github.io/NIPS18_EDL_poster.pdf) [[code example]](https://muratsensoy.github.io/uncertainty.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Evidential%20Deep%20Learning%20to%20Quantify%20Classification%20Uncertainty.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Evidential%20Deep%20Learning%20to%20Quantify%20Classification%20Uncertainty.md)
- *Murat Sensoy, Lance Kaplan, Melih Kandemir*
- `2018-10-31, NeurIPS2018`

##### [18-11-22] [paper22]
-  A Probabilistic U-Net for Segmentation of Ambiguous Images [[pdf]](https://arxiv.org/abs/1806.05034) [[code]](https://github.com/SimonKohl/probabilistic_unet) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Probabilistic%20U-Net%20for%20Segmentation%20of%20Ambiguous%20Images.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/A%20Probabilistic%20U-Net%20for%20Segmentation%20of%20Ambiguous%20Images.md)
- *Simon A. A. Kohl, Bernardino Romera-Paredes, Clemens Meyer, Jeffrey De Fauw, Joseph R. Ledsam, Klaus H. Maier-Hein, S. M. Ali Eslami, Danilo Jimenez Rezende, Olaf Ronneberger*
- `2018-10-29, NeurIPS2018`

##### [18-11-22] [paper21]
- When Recurrent Models Don't Need To Be Recurrent (a.k.a. Stable Recurrent Models) [[pdf]](https://arxiv.org/abs/1805.10369) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/When%20Recurrent%20Models%20Don%E2%80%99t%20Need%20To%20Be%20Recurrent.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/When%20Recurrent%20Models%20Don't%20Need%20To%20Be%20Recurrent.md)
- *John Miller, Moritz Hardt*
- `2018-05-29, ICLR2019`

##### [18-11-16] [paper20]
- Uncertainty Estimates and Multi-Hypotheses Networks for Optical Flow [[pdf]](https://arxiv.org/abs/1802.07095) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20Estimates%20and%20Multi-Hypotheses%20Networks%20for%20Optical%20Flow_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Uncertainty%20Estimates%20and%20Multi-Hypotheses%20Networks%20for%20Optical%20Flow.md)
- *Eddy Ilg, Özgün Çiçek, Silvio Galesso, Aaron Klein, Osama Makansi, Frank Hutter, Thomas Brox*
- `2018-08-06, ECCV2018`

##### [18-11-15] [paper19]
- Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV) [[pdf]](https://arxiv.org/abs/1711.11279) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Interpretability%20Beyond%20Feature%20Attribution:%20Quantitative%20Testing%20with%20Concept%20Activation%20Vectors%20(TCAV)_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Interpretability%20Beyond%20Feature%20Attribution:%20Quantitative%20Testing%20with%20Concept%20Activation%20Vectors%20(TCAV).md)
- *Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, Rory Sayres*
- `2018-06-07, ICML2018`

##### [18-11-12] [paper18]
- Large-Scale Visual Active Learning with Deep Probabilistic Ensembles [[pdf]](https://arxiv.org/abs/1811.03575) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Large-Scale%20Visual%20Active%20Learning%20with%20Deep%20Probabilistic%20Ensembles_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Large-Scale%20Visual%20Active%20Learning%20with%20Deep%20Probabilistic%20Ensembles.md)
- *Kashyap Chitta, Jose M. Alvarez, Adam Lesnikowski*
- `2018-11-08`

##### [18-11-08] [paper17]
- The Lottery Ticket Hypothesis: Finding Small, Trainable Neural Networks [[pdf]](https://arxiv.org/abs/1803.03635) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/The%20Lottery%20Ticket%20Hypothesis:%20Finding%20Small%2C%20Trainable%20Neural%20Networks_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/The%20Lottery%20Ticket%20Hypothesis:%20Finding%20Small%2C%20Trainable%20Neural%20Networks.md)
- *Jonathan Frankle, Michael Carbin*
- `2018-03-09, ICLR2019`

##### [18-10-26] [paper16]
- Towards Safe Autonomous Driving: Capture Uncertainty in the Deep Neural Network For Lidar 3D Vehicle Detection [[pdf]](https://arxiv.org/abs/1804.05132) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Towards%20Safe%20Autonomous%20Driving:%20Capture%20Uncertainty%20in%20the%20Deep%20Neural%20Network%20For%20Lidar%203D%20Vehicle%20Detection_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Towards%20Safe%20Autonomous%20Driving:%20Capture%20Uncertainty%20in%20the%20Deep%20Neural%20Network%20For%20Lidar%203D%20Vehicle%20Detection.md)
- *Di Feng, Lars Rosenbaum, Klaus Dietmayer*
- `2018-09-08, ITSC2018`

##### [18-10-25] [paper15]
- Bayesian Convolutional Neural Networks with Many Channels are Gaussian Processes [[pdf]](https://arxiv.org/abs/1810.05148) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Bayesian%20Convolutional%20Neural%20Networks%20with%20Many%20Channels%20are%20Gaussian%20Processes_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Bayesian%20Convolutional%20Neural%20Networks%20with%20Many%20Channels%20are%20Gaussian%20Processes.md)
- *Roman Novak, Lechao Xiao, Jaehoon Lee, Yasaman Bahri, Daniel A. Abolafia, Jeffrey Pennington, Jascha Sohl-Dickstein*
- `2018-10-11, ICLR2019`

##### [18-10-19] [paper14]
- Uncertainty in Neural Networks: Bayesian Ensembling [[pdf]](https://arxiv.org/abs/1810.05546) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20in%20Neural%20Networks:%20Bayesian%20Ensembling_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Uncertainty%20in%20Neural%20Networks:%20Bayesian%20Ensembling.md)
- *Tim Pearce, Mohamed Zaki, Alexandra Brintrup, Andy Neel*
- `2018-10-12, AISTATS2019 submission`

##### [18-10-18] [paper13]
- Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles [[pdf]](https://arxiv.org/abs/1612.01474) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles.md)
- *Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell*
- `2017-11-17, NeurIPS2017`

##### [18-10-18] [paper12]
- Reliable Uncertainty Estimates in Deep Neural Networks using Noise Contrastive Priors [[pdf]](https://arxiv.org/abs/1807.09289) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Reliable%20Uncertainty%20Estimates%20in%20Deep%20Neural%20Networks%20using%20Noise%20Contrastive%20Priors_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/Reliable%20Uncertainty%20Estimates%20in%20Deep%20Neural%20Networks%20using%20Noise%20Contrastive%20Priors.md)
- *Danijar Hafner, Dustin Tran, Alex Irpan, Timothy Lillicrap, James Davidson*
- `2018-07-24, ICML2018 Workshop`

##### [18-10-05] [paper11]
- VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection [[pdf]](https://arxiv.org/abs/1711.06396) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/VoxelNet:%20End-to-End%20Learning%20for%20Point%20Cloud%20Based%203D%20Object%20Detection_.pdf) [[summary]](https://github.com/fregu856/papers/blob/master/summaries/VoxelNet:%20End-to-End%20Learning%20for%20Point%20Cloud%20Based%203D%20Object%20Detection.md)
- *Yin Zhou, Oncel Tuzel*
- `2017-11-17, CVPR2018`

##### [18-10-04] [paper10]
- PIXOR: Real-time 3D Object Detection from Point Clouds [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/PIXOR:%20Real-time%203D%20Object%20Detection%20from%20Point%20Clouds_.pdf)
- *Bin Yang, Wenjie Luo, Raquel Urtasun*
- `CVPR 2018`
```
General comments on paper quality:
Fairly well-written paper, although there are a couple of typos (weird grammar). Quite interesting proposed 3D localization model.


Paper overview:
The authors present a single-stage, LiDAR-only model for 3D localization of vehicles (bird's eye view).

The most interesting/novel contribution is probably the utilized LiDAR input representation:

They discretize 3D space into a 3D voxel grid of resolution 0.1 m, each grid cell is then either assigned a value of 1 (if the grid cell contains any LiDAR points) or 0 (if the grid cell does NOT contain any LiDAR points), which results in a 3D occupancy map (e.g. a 800x700x35 tensor of 0s and 1s).
This 3D tensor is then fed as input to a conventional (2D) CNN, i.e., the height dimension plays the role of the rgb channels in an image(!).
The approach is thus very similar to models which first project the LiDAR point cloud onto a bird's eye view, in that we only need to use 2D convolutions (which is significantly more efficient than using 3D convolutions), but with the difference being that we in this approach don't need to extract any hand-crafted features in order to obtain a bird's eye view feature map. Thus, at least in theory, this approach should be comparable to bird's eye view based models in terms of efficiency, while being capable of learning a more rich bird's eye view feature representation.
The model outputs a 200x175x7 tensor, i.e., 7 values (one objectness/confidence score + cos(theta) and sin(theta) + regression targets for x, y, w, and l) for each grid cell (when the feature map is spatially down-sampled in the CNN, this corresponds to an increasingly sparser grid). The authors say that their approach doesn't use any pre-defined object anchors, but actually I would say that it uses a single anchor per grid cell (centered at the cell, with width and length set to the mean of the training set, and the yaw angle set to zero).

They use the focal loss to handle the large class imbalance between objects and background.

In inference, only anchors whose confidence score exceeds a certain threshold are decoded (i.e., rotated, translated and resized according to the regressed values), and non-maximum-suppression (NMS) is then used to get the final detections.

They evaluate their method on KITTI and compare to other entries on the bird's eye view leaderboard. They obtain very similar performance to the LiDAR-only version of MV3D and somewhat significantly worse than VoxelNet, i.e., not OVERLY impressive performance but still pretty good. The method is also significantly faster in inference than both MV3D and VoxelNet.


Comments:
Pretty interesting paper. The approach of creating a 3D occupancy map using discretization and then processing this with 2D convolutions seems rather clever indeed. One would think this should be quite efficient while also being able to learn a pretty rich feature map.

I don't think the model should be particularly difficult to extend to full 3D object detection either, you would just need to also regress values for z (relative to some default ground plane z value, i.e., we assume that all anchor 3dbboxes sit on the ground plane) and h (relative to the mean h in the training set). I think this is basically what is done in VoxelNet?

There are however some design choices which I find somewhat peculiar, e.g. the way they assign anchors (the authors just talk about "pixels") to being either positive (object) or negative (background).
```

##### [18-10-04] [paper9]
- On gradient regularizers for MMD GANs [[pdf]](https://arxiv.org/abs/1805.11565) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20gradient%20regularizers%20for%20MMD%20GANs_.pdf)
- *Michael Arbel, Dougal J. Sutherland, Mikołaj Bińkowski, Arthur Gretton*
- `2018-05-29, NeurIPS 2018`
```
General comments on paper quality:
Well-written but rather heavy paper to read, I did definitely not have the background required neither to fully understand nor to properly appreciate the proposed methods. I would probably need to do some more background reading and then come back and read this paper again.


Paper overview:
The authors propose the method Gradient-Constrained MMD and its approximation Scaled MMD, MMD GAN architectures which are trained using a novel loss function that regularizes the gradients of the critic (gradient-based regularization).

The authors experimentally evaluate their proposed architectures on the task of unsupervised image generation, on three different datasets (CIFAR-10 (32x32 images), CelebA (160x160 images) and ImageNet (64x64 images)) and using three different metrics (Inception score (IS), FID and KID). They find that their proposed losses lead to stable training and that they outperform (or at least obtain highly comparable performance to) state-of-the-art methods (e.g. Wasserstein GAN).
```

##### [18-09-30] [paper8]
- Neural Processes [[pdf]](https://arxiv.org/abs/1807.01622) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Processes_.pdf)
- *Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J. Rezende, S.M. Ali Eslami, Yee Whye Teh*
- `2018-07-04, ICML 2018 Workshop`
```
General comments on paper quality:
Quite well-written overall, although I did find a couple of typos (and perhaps you should expect more for a paper with 7 authors?). Also, I don't think the proposed method is quite as clearly explained as in the Conditional Neural Processes paper.


Paper overview:
The authors introduce Neural Processes (NPs), a family of models that aim to combine the benefits of Gaussian Processes (GPs) and neural networks (NNs). NPs are an extension of Conditional Neural Processes (CNPs) by the same main authors ([summary]).

A blog post that provides a good overview of and some intuition for NPs is found here.

NPs are very similar to CNPs, with the main difference being the introduction of a global latent variable z:

Again, each observation (x_i, y_i) (a labeled example) is fed through the encoder NN h to extract an embedding r_i. The embeddings r_i are aggregated (averaged) to a single embedding r. This r is then used to parameterize the distribution of the global latent variable z: z ~ N(mu(r), sigma(r)).
z is then sampled and fed together with each target x*_i (unlabeled example) as input to the decoder NN g, which produces a corresponding prediction y*_i.
The introduced global latent variable z is supposed to capture the global uncertainty, which allows one to sample at a global level (one function at a time) rather than just at a local output level (one y_i value for each x_i at the time). I.e., NPs are able (whereas CNPs are unable) to produce different function samples for the same conditioned observations. In e.g. the application of image completion on MNIST, this will make the model produce different digits when sampled multiple times and the model only has seen a small number of observations (as compared to CNPs which in the same setting would just output (roughly) the mean of all MNIST images).

The authors experimentally evaluate the NP approach on several different problems (1D function regression, image completion, Bayesian optimization of 1D functions using Thompson sampling, contextual bandits), they do however not really compare the NP results with those of CNPs.


Comments:
It's not immediately obvious to me what use NPs could have in the context of e.g. 3DOD, and not what practical advantage using NPs would have over using CNPs either.

I suppose the approach might be interesting if you were to train a 3DOD model on multiple datasets with the goal to maximize performance on a specific dataset (e.g., you have collected / have access to a bunch of data corresponding to different LiDAR sensor setups, and now you want to train a model for your final production car sensor setup). In this case I guess this approach might be preferable over just simply training a conventional DNN on all of the data?
```

##### [18-09-27] [paper7]
- Conditional Neural Processes [[pdf]](https://arxiv.org/abs/1807.01613) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Conditional%20Neural%20Processes_.pdf)
- *Marta Garnelo, Dan Rosenbaum, Chris J. Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo J. Rezende, S. M. Ali Eslami*
- `2018-07-04, ICML 2018`
```
General comments on paper quality:
Quite well-written. Interesting proposed method.


Paper overview:
The authors present a family of neural models called Conditional Neural Processes (CNPs), which aim to combine the benefits of Gaussian Processes (which exploit prior knowledge to quickly infer the shape of a new function at test time, but computationally scale poorly with increased dimension and dataset size) and deep neural networks (which excel at function approximation, but need to be trained from scratch for each new function).

A CNP feeds each observation (x_i, y_i) (a labeled example) through a neural network h to extract an embedding r_i. The embeddings r_i are aggregated to a single embedding r using a symmetric aggregator function (e.g. taking the mean). The embedding r is then fed together with each target x*_i (unlabeled example) as input to the neural network g, which produces a corresponding prediction y*_i. The predictions are thus made conditioned on the observations (x_i, y_i).

For example, given observations of an unknown function's value y_i at locations x_i, we would like to predict the function's value at new locations x*_i, conditioned on the observations.

To train a CNP, we have access to a training set of n observations (x_i, y_i). We then produce predictions y^_i for each x_i, conditioned on a randomly chosen subset of the observations, and minimize the negative (conditional) log likelihood. I.e., create r by embedding N randomly chosen observations (x_i, y_i) with the neural network h, then feed r as input to g and compute predictions y^_i for each x_i, and compare these with the true values y_i.

The authors experimentally evaluate the CNP approach on three different problems:

1D regression:
At every training step, they sample a curve (function) from a fixed Gaussian Process (GP), select a subset of n points (x_i, y_i) from it as observations, and a subset of m points (x'_j, y'_j) as targets.
The embedding r is created from the observations, used in g to output a prediction (y^_j, sigma^_j) (mean and variance) for each target x'_j, and compared with the true values y`_j.
In inference, they again sample a curve from the GP and are given a subset of points from it as observations. From this they create the embedding r, and are then able to output predictions (mean and variance) at arbitrary points x. Hopefully, these predictions (both in terms of mean and variance) will be close to what is outputted by a GP (with the true hyperparameters) fitted to the same observations. This is also, more or less, what they observe in their experiments.
Image completion:
Given a few pixels as observations, predict the pixel values at all pixel locations. CNP is found to outperform both a kNN and GP baseline, at least when the number of given observations is relatively small.
One-shot classification:
While CNP does NOT set a new SOTA, it is found to have comparable performance to significantly more complex models.
The authors conclude by arguing that a trained CNP is more general than conventional deep learning models, in that it encapsulates the high-level statistics of a family of functions. As such it constitutes a high-level abstraction that can be reused for multiple tasks.

Left as future work is the task of scaling up the proof-of-concept CNP architectures used in the paper, and exploring how CNPs can help tackling problems such as transfer learning and data efficiency.


Comments:
Pretty interesting approach, although it's not immediately obvious to me what use it could have in the context of 3DOD and/or uncertainty estimation (outputting both a mean and a variance is of course interesting, but you don't need to explicitly condition the model on some observations). I guess transfer learning to fine-tune performance on a specific subset of your data is the most obvious possible application.
```

##### [18-09-27] [paper6]
- Neural Autoregressive Flows [[pdf]](https://arxiv.org/abs/1804.00779) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Autoregressive%20Flows_.pdf)
- *Chin-Wei Huang, David Krueger, Alexandre Lacoste, Aaron Courville*
- `2018-04-03, ICML 2018`
```
General comments on paper quality:
Well-written and interesting paper. As I was quite unfamiliar with the material, it did however require an extra read-through.


Paper overview:
The authors introduce Neural Autoregressive Flow (NAF), a flexible method for tractably approximating rich families of distributions. Empirically, they show that NAF is able to model multi-modal distributions and outperform related methods (e.g. inverse autoregressive flow (IAF)) when applied to density estimation and variational inference.

A Normalizing Flow (NF), is an invertible function f: X --> Y expressing the transformation between two random variables (i.e., is used to translate between the distributions p_{Y}(y) and p_{X}(x)). NFs are most commonly trained to, from an input distribution p_{X}(x), produce an output distributions p_{Y}(y) that matches a target distribution p_{target}(y) (as measured by the KL divergence). E.g. in variational inference, the NF is typically used to transform a simple input distribution (e.g. standard normal) over x into a complex approximate posterior p_{Y}(y).

Research on constructing NFs, such as this work, focuses on finding ways to parameterize the NF which meet certain requirements while being maximally flexible in terms of the transformations which they can represent.

One specific (particularly successful) class of NFs are affine autoregressive flows (AAFs) (e.g. IAFs). In AFFs, the components of x and y (x_{i}, y_{i}) are given an (arbitrarily chosen) order, and y_{t} is computed as a function of x_{1:t}:

y_{t} = f(x_{1:t}) = tau(c(x_{1:t-1}), x_{t}), where:
c is an autoregressive conditioner.
tau is an invertible transformer.
In previous work, tau is taken to be a simple affine function, e.g. in IAFs:

tau(mu, sigma, x_{t}) = sigma*x_{t} + (1-sigma)*mu, where mu and sigma are outputted by the conditioner c.
In this paper, the authors replace the affine transformer tau with a neural network (yielding a more rich family of distributions with only a minor increase in computation and memory requirements), which results in the NAF method:

tau(c(x_{1:t-1}), x_{t}) = NN(x_{t}; phi = c(x_{1:t-1})), where:
NN is a (small, 1-2 hidden layers with 8/16 units) neural network that takes the scalar x_{t} as input and produces y_{t} as output, and its weights and biases phi are outputted by c(x_{1:t-1}).
To ensure that tau is strictly monotonic an thus invertible, it is sufficient to use strictly positive weights and strictly monotonic activation functions in the neural network.
The authors prove that NAFs are universal density approximators, i.e., can be used to approximate any probability distribution (over real vectors) arbitrarily well. I.e., NAFs can be used to transform any random variable into any desired random variable.

The authors empirically evaluate NAFs applied to variational inference and density estimation, and outperform IAF and MAF baselines. For example, they find that NAFs can approximate a multi-modal mixture of Gaussian distribution quite well, while AAFs only produces a uni-modal distribution.


Comments:
I probably need to do some more reading on the background material to fully understand and appreciate the results of this paper, but it definitely seems quite interesting. Could probably be useful in some application.
```

##### [18-09-25] [paper5]
- Deep Confidence: A Computationally Efficient Framework for Calculating Reliable Errors for Deep Neural Networks [[pdf]](https://arxiv.org/abs/1809.09060) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Confidence:%20A%20Computationally%20Efficient%20Framework%20for%20Calculating%20Reliable%20Errors%20for%20Deep%20Neural%20Networks.pdf)
- *Isidro Cortes-Ciriano, Andreas Bender*
- `2018-09-24`
```
General comments on paper quality:
Unfamiliar paper formatting (written by authors from a different field), but actually a well-written and interesting paper. The methods are quite clearly explained.


Paper overview:
The authors present a framework for computing "valid and efficient" confidence intervals for individual predictions made by a deep learning network, applied to the task of drug discovery (modeling bioactivity data).

To create confidence intervals, they use an ensemble of 100 networks (either obtained by independently training 100 networks, or by using Snapshot Ensembling: saving network snapshots during the training of a single network (essentially)) together with a method called conformal prediction.

More specifically, predictions are produced by the 100 networks for each example in the validation set, and for each example the sample mean y_hat and sample std sigma_hat are computed. From y_hat and sigma_hat, for each example, a non-comformity value alpha is computed (alpha = |y - y_hat|/exp(sigma_hat)). These non-conformity values are then sorted in increasing order, and the percentile corresponding to the chosen confidence level (e.g. 80%, alpha_80) is selected (I suppose what you actually do here is that you find the smallest alpha value which is larger than 80% of all alpha values?).

On the test set, for each example, again a prediction is produced by the 100 networks and y_hat, sigma_hat are computed. A confidence interval is then given by: y_hat +/- exp(sigma_hat)*alpha_80.

The key result in the paper is that the authors find that these confidence intervals are indeed valid. I.e., when they compute the percentage of examples in the test set whose true values lie within the predicted confidence interval, this fraction was equal to or greater than the selected confidence level.

They also compare the constructed confidence intervals with the ones obtained by a random forest model, and find that they have comparable efficiency (confidence intervals have a small average size -> higher efficiency).


Comments:
Since the paper comes from an unfamiliar field (and I'm not at all familiar with the used datasets etc.), I'm being somewhat careful about what conclusions I draw from the presented results, but it definitely seems interesting and as it potentially could be of practical use.

The specific method used, i.e. an ensemble of 100 models, isn't really applicable in my case since it would make inference too slow, but the approach for constructing and validating confidence intervals might actually be useful.

For example, what would happen if you replace y_hat, sigma_hat with the predicted mean and std by a single model (aleatoric uncertainty)? Would the induced confidence intervals still be valid, at least for some confidence level?

And if so, I suppose it doesn't really matter that the network in a sense is still just a black box, since you now actually are able to quantify its uncertainty on this data, and you at least have some kind of metric to compare different models with (obviously, you still don't know how much the output actually can be trusted when the network is applied to completely different data, but the fact that the confidence intervals are valid on this specific dataset is still worth something).
```

##### [18-09-25] [paper4]
- Leveraging Heteroscedastic Aleatoric Uncertainties for Robust Real-Time LiDAR 3D Object Detection [[pdf]](https://arxiv.org/abs/1809.05590) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Leveraging%20Heteroscedastic%20Aleatoric%20Uncertainties%20for%20Robust%20Real-Time%20LiDAR%203D%20Object%20Detection_.pdf)
- *Di Feng, Lars Rosenbaum, Fabian Timm, Klaus Dietmayer*
- `2018-09-14`
```
General comments on paper quality:
Fairly well-written paper. I did find a couple of typos though, and some concepts could definitely have been more carefully and clearly explained.


Paper overview:
The authors present a two-stage, LiDAR-only model for 3D object detection (trained only on the Car class on KITTI), which also models aleatoric (heteroscedastic) uncertainty by assuming a Gaussian distribution over the model output, similarly to how Kendall and Gal models aleatoric uncertainty. The 3DOD model takes as input LiDAR point clouds which have been projected to a 2D bird's eye view.

The network outputs uncertainties in both the RPN and in the refinement head, for the anchor position regression, 3D location regression and orientation regression. They do NOT model uncertainty in the classification task, but instead rely on the conventional softmax scores.

The deterministic version of the 3DOD model has fairly competitive AP3D performance on KITTI test (not as good as VoxelNet, but not bad for being a LiDAR-only method). What's actually interesting is however that modeling aleatoric uncertainty improves upon this performance with roughly 7% (Moderate class), while only increasing the inference time from 70 ms to 72 ms (TITAN X GPU).

The authors conduct some experiments to try and understand their estimated aleatoric uncertainties:

They find that the network generally outputs larger orientation uncertainty when the predicted orientation angle is far from the four most common angles {0, 90, 180, 270}.
They find that the outputted uncertainty generally increases as the softmax score decreases.
They find that the outputted uncertainty generally increases as detection distance increases.
The learned aleatoric uncertainty estimates thus seem to make intuitive sense in many cases.


Comments:
Just like in Kendall and Gal and Gast and Roth, modelling aleatoric uncertainty improves performance while adding negligible computational complexity. That this can be a practically useful tool is thus starting to become quite clear.

However, I still think we need to analyze the estimated uncertainties more carefully. Can they be used to form valid confidence intervals?

Also, it feels like this work is HEAVILY inspired by Kendall and Gal but it's not really explicitly mentioned anywhere in the paper. I personally think they could have given more credit.
```

##### [18-09-24] [paper3]
- Lightweight Probabilistic Deep Networks [[pdf]](https://arxiv.org/abs/1805.11327) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Lightweight%20Probabilistic%20Deep%20Networks_.pdf)
- *Jochen Gast, Stefan Roth*
- `2018-05-29, CVPR 2018`
```
General comments on paper quality:
Quite interesting and well written paper, I did however find a couple of the derivations (Deep uncertainty propagation using ADF & Classification with Dirichlet outputs) somewhat difficult to follow.


Paper overview:
The authors introduce two lightweight approaches to supervised learning (both regression and classification) with probabilistic deep networks:

ProbOut: replace the output layer of a network with a distribution over the output (i.e., output e.g. a Gaussian mean and variance instead of just a point estimate).
ADF: go one step further and replace all intermediate activations with distributions. Assumed density filtering (ADF) is used to propagate activation uncertainties through the network.
I.e., their approach is not a Bayesian network in the classical sense (there's no distribution over the network weights). In the terminology of Kendall and Gal, the approach only captures aleatoric (heteroscedastic) uncertainty. In fact, ProbOut is virtually identical to the approach used by Kendall and Gal to model aleatoric uncertainty (they do however use slightly different approaches for classification tasks). The authors choose to disregard epistemic (model) uncertainty in favor of improved computational performance, arguing that epistemic uncertainty is less important since it can be explained away with enough data.

While ProbOut is simple to both formulate and implement, ADF is more involved. ADF is also nearly 3x as slow in inference, while ProbOut adds negligible compute compared to standard deterministic networks.

The authors evaluate ProbOut and ADF on the task of optical flow estimation (regression) and image classification. They find that their probabilistic approaches somewhat outperform the deterministic baseline across tasks and datasets. There's however no significant difference between ProbOut and ADF.

They empirically find the estimated uncertainties from their models to be highly correlated with the actual error. They don't really mention if ProbOut or ADF is significantly better than the other in this regard.


Comments:
From the results presented in the paper, I actually find it quite difficult to see why anyone would prefer ADF over ProbOut. ProbOut seems more simple to understand and implement, is quite significantly faster in inference, and seems to have comparable task performance and capability to model aleatoric uncertainty.

Thus, I'm not quite sure how significant the contribution of this paper actually is. Essentially, they have taken the method for modeling aleatoric uncertainty from Kendall and Gal and applied this to slightly different tasks.

Also, my question from the Kendall and Gal summary still remains. Even if we assume negligible epistemic (model) uncertainty, how much can we actually trust the outputted aleatoric uncertainty?
```

##### [18-09-24] [paper2]
- What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? [[pdf]](https://arxiv.org/abs/1703.04977) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/What%20Uncertainties%20Do%20We%20Need%20in%20Bayesian%20Deep%20Learning%20for%20Computer%20Vision%3F_.pdf)
- *Alex Kendall, Yarin Gal*
- `2017-10-05, NeurIPS 2017`
```
General comments on paper quality:
Well written and interesting paper. Seems like a fairly complete introduction to Bayesian deep learning. Clearly defines aleatoric and epistemic uncertainty, and provides good intuition for what they capture and how they differ. A recommended read.


Paper overview:
The authors describe the two major types of uncertainty that one can model:

Aleatoric uncertainty: captures noise inherent in the observations, uncertainty which NOT can be reduced by collecting more data. It can also vary for different inputs to the model, if we for example are trying to estimate the distance to vehicles seen in camera images, we'd expect the distance estimate to be more noisy/uncertain for vehicles far away from the camera.
Epistemic uncertainty (a.k.a model uncertainty): accounts for uncertainty in the model parameters, uncertainty which CAN be explained away given enough data.
To model epistemic uncertainty in a neural network (NN), one puts a prior distribution over its weights W and then tries to compute the posterior p(W | X, Y). This is what typically is called a Bayesian NN (BNN). BNNs are easy to formulate, but difficult to perform inference in. Different approximate techniques exist, and the authors use Monte Carlo dropout (Use dropout during both training and testing. During testing, run multiple forward-passes and (essentially) compute the sample mean and variance).

To model aleatoric uncertainty, one assumes a (conditional) distribution over the output of the network (e.g. Gaussian with mean u(x) and sigma s(x)) and learns the parameters of this distribution (in the Gaussian case, the network outputs both u(x) and s(x)) by maximizing the corresponding likelihood function. Note that in e.g. the Gaussian case, one does NOT need extra labels to learn s(x), it is learned implicitly from the induced loss function. The authors call such a model a heteroscedastic NN.

(At least for the Gaussian case) they note that letting the model output both u(x) and s(x) allows it to intelligently attenuate the residuals in the loss function (since the residuals are divided by s(x)^2), making the model more robust to noisy data.

The novel contribution by the authors is a framework for modeling both epistemic and aleatoric uncertainty in the same model. To do this, they use Monte Carlo dropout to turn their heteroscedastic NN into a Bayesian NN (essentially: use dropout and run multiple forward passes in a model which outputs both u(x) and s(x)). They demonstrate their framework for both regression and classification tasks, as they present results for per-pixel depth regression and semantic segmentation.

For each task, they compare four different models:

Without uncertainty.
Only aleatoric uncertainty.
Only epistemic uncertainty.
Aleatoric and epistemic uncertainty.
They find that modeling both aleatoric and epistemic uncertainty results in the best performance (roughly 1-3% improvement over no uncertainty) but that the main contribution comes from modeling the aleatoric uncertainty, suggesting that epistemic uncertainty mostly can be explained away when using large datasets.

Qualitatively, for depth regression, they find that the aleatoric uncertainty is larger for e.g. great depths and reflective surfaces, which makes intuitive sense.

The authors also perform experiments where they train the models on increasingly larger datasets (1/4 of the dataset, 1/2 of the dataset and the full dataset) and compare their performance on different test datasets.

They here find that the aleatoric uncertainty remains roughly constant for the different cases, suggesting that aleatoric uncertainty NOT can be explained away with more data (as expected), but also that aleatoric uncertainty does NOT increase for out-of-data examples (examples which differs a lot from the training data).
On the other hand, they find that the epistemic uncertainty clearly decreases as the training datasets gets larger (i.e., it seems as the epistemic uncertainty CAN be explained away with enough data, as expected), and that it is significantly larger when the training and test datasets are NOT very similar (i.e., the epistemic uncertainty is larger when we train on dataset A-train and test on dataset B-test, than when we train on dataset A-train and test on dataset A-test).
This reinforces the case that while epistemic uncertainty can be explained away with enough data, it is still required to capture situations not encountered in the training data, which is particularly important for safety-critical systems (where epistemic uncertainty is required to detect situations which have never been seen by the model before).

Finally, the authors note that the aleatoric uncertainty models add negligible compute compared to deterministic models, but that the epistemic models require expensive Monte Carlo dropout sampling (50 Monte Carlo samples often results in a 50x slow-down). They thus mark finding a method for real-time epistemic uncertainty estimation as an important direction for future research.


Comments:
From the authors' experiments, it seems reasonably safe to assume that incorporating some kind of uncertainty measure can help improve model performance. It could thus definitely be of practical use (especially aleatoric uncertainty estimation, since it's quite simple to implement and computationally inexpensive).

However, it's still not quite clear to me how much you can actually trust these estimated uncertainties. The NN is still essentially a black box, so how do we know if the outputted aleatoric uncertainty estimate is "correct" in any given case? Is it possible to somehow create e.g. 95% confidence intervals from these estimated uncertainties?
```

##### [18-09-20] [paper1]
- Gaussian Process Behaviour in Wide Deep Neural Networks [[pdf]](https://arxiv.org/abs/1804.11271) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Gaussian%20Process%20Behaviour%20in%20Wide%20Deep%20Neural%20Networks.pdf)
- *Alexander G. de G. Matthews, Mark Rowland, Jiri Hron, Richard E. Turner, Zoubin Ghahramani*
- `2018-08-16, ICLR 2018`
```
General comments on paper quality:
Well written and mathematically rigorous paper that I'd recommend anyone interested in theoretical properties of deep learning to read. An interesting and pleasent read.


Paper overview:
The authors study the relationship between random, wide, fully connected, feedforward neural networks and Gaussian processes.

The network weights are assumed to be independent normally distributed with their variances sensibly scaled as the network grows.

They show that, under broad conditions, as the network is made increasingly wide, the implied random input-to-output function converges in distribution to a Gaussian process.

They also compare exact Gaussian process inference with MCMC inference for finite Bayesian neural networks. Of the six datasets considered, five show close agreement between the two models.

Because of the computational burden of the MCMC algorithms, the problems they study are however quite small in terms of both network size, data dimensionality and datset size.

Furthermore, the one dataset on which the Bayesian deep network and the Gaussian process did not make very similar predictions was the one with the highest dimensionality. The authors thus sound a note of caution about extrapolating their empirical findings too confidently to the domain of high-dimensional, large-scale datasets.

Still, the authors conclude that it seems likely that some experiments in the Bayesian deep learning literature would have given very similar results to a Gaussian process. They thus also recommend the Bayesian deep learning community to routinely compare their results to Gaussian processes (with the kernels specified in the paper).

Finally, the authors hope that their results will help to further the theoretical understanding of deep neural networks in future follow-up work.
```
