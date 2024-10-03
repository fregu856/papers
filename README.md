# About

I categorize, annotate and write comments for all research papers I read (410+ papers since 2018).

_In June 2023, I wrote the blog post [The How and Why of Reading 300 Papers in 5 Years](https://www.fregu856.com/post/phd_of_reading/) (why I think it’s important to read a lot of papers + how I organize my reading + paper statistics + a list of 30 particularly interesting papers)._

#### Categories:

[Uncertainty Estimation], [Ensembling], [Stochastic Gradient MCMC], [Variational Inference], [Out-of-Distribution Detection], [Theoretical Properties of Deep Learning], [VAEs], [Normalizing Flows], [ML for Medicine/Healthcare], [Object Detection], [3D Object Detection], [3D Multi-Object Tracking], [3D Human Pose Estimation], [Visual Tracking], [Sequence Modeling], [Reinforcement Learning], [Energy-Based Models], [Neural Processes], [Neural ODEs], [Transformers], [Implicit Neural Representations], [Distribution Shifts], [Social Consequences of ML], [Diffusion Models], [Graph Neural Networks], [Selective Prediction], [NLP], [Representation Learning], [Vision-Language Models], [Image Restoration], [Computational Pathology], [Survival Analysis], [Miscellaneous].


### Papers:

- [Papers Read in 2024](#papers-read-in-2024)
- [Papers Read in 2023](#papers-read-in-2023)
- [Papers Read in 2022](#papers-read-in-2022)
- [Papers Read in 2021](#papers-read-in-2021)
- [Papers Read in 2020](#papers-read-in-2020)
- [Papers Read in 2019](#papers-read-in-2019)
- [Papers Read in 2018](#papers-read-in-2018)

#### Papers Read in 2024:

##### [24-02-04] [paper4XX]
- 
 [[pdf]]() [[annotated pdf]]()
- ``
- [Computational Pathology]
```

```



##### [24-04-22] [paper421]
- Estrogen Receptor Gene Expression Prediction from H&E Whole Slide Images
 [[pdf]](https://www.medrxiv.org/content/10.1101/2024.04.05.24302951v1) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Estrogen%20Receptor%20Gene%20Expression%20Prediction%20from%20H%26E%20Whole%20Slide%20Images.pdf)
- `medrxiv, 2024-04`
- [Computational Pathology]
```
Short paper, just 3.5 pages, but well-written and quite interesting. The performance doesn't seem overly impressive though (Pearson's corr of 0.57, not that much lower MAE than for the "predict the dataset mean" baseline) (also, Table 3), feels like higher regression accuracy might be needed for this to actually be useful in practice. Would have been interesting to have the "using ground truth ESRI expression" upper bound also in Table 3.
```

##### [24-03-09] [paper420]
- Multimodal Histopathologic Models Stratify Hormone Receptor-Positive Early Breast Cancer
 [[pdf]](https://www.biorxiv.org/content/10.1101/2024.02.23.581806v1) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Multimodal%20Histopathologic%20Models%20Stratify%20Hormone%20Receptor-Positive%20Early%20Breast%20Cancer.pdf)
- `biorxiv, 2024-02`
- [Computational Pathology]
```
Well-written and interesting paper, I enjoyed reading it. Good background material for me as well.
```

##### [24-03-07] [paper419]
- Spatially Resolved Gene Expression Prediction from Histology Images via Bi-modal Contrastive Learning
 [[pdf]](https://openreview.net/forum?id=eT1tMdAUoc) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Spatially%20Resolved%20Gene%20Expression%20Prediction%20from%20Histology%20Images%20via%20Bi-modal%20Contrastive%20Learning.pdf)
- `NeurIPS 2023`
- [Computational Pathology]
```
Interesting and quite well-written paper. The method is definitely interesting, makes intutive sense. The experiments/results are not overly extensive/impressive perhaps (I also ran a bit short on time, so I read Section 4-5 quite quickly). Don't like the formatting of the two equations in Section 3.2.
```

##### [24-02-10] [paper418]
- An Investigation of Attention Mechanisms in Histopathology Whole-Slide-Image Analysis for Regression Objectives
 [[pdf]](https://openaccess.thecvf.com/content/ICCV2021W/CDPath/papers/Weitz_An_Investigation_of_Attention_Mechanisms_in_Histopathology_Whole-Slide-Image_Analysis_for_ICCVW_2021_paper.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/An%20Investigation%20of%20Attention%20Mechanisms%20in%20Histopathology%20Whole-Slide-Image%20Analysis%20for%20Regression%20Objectives.pdf)
- `ICCV Workshops 2021`
- [Computational Pathology]
```
Well-written and interesting paper. I'm slightly confused by the MNIST experiment though. It seems weird to add a proportion of random noise images, what would those correspond to in the histopathology application?
```

##### [24-02-06] [paper417]
- Attention-based Interpretable Regression of Gene Expression in Histology
 [[pdf]](https://arxiv.org/abs/2208.13776) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Attention-based%20Interpretable%20Regression%20of%20Gene%20Expression%20in%20Histology.pdf)
- `MICCAI Workshops 2022`
- [Computational Pathology]
```
Well-written and quite interesting paper. The method makes sense overall I think, although it seems a bit strange to use a model pretrained on imagenet as a frozen feature extractor. Not quite clear to me how many of the 45 genes the model actually can predict well, they list just four genes with correlation above 0.6 in Section 3.2, are these the only genes that reach 0.6? Difficult for me to judge if the attention heatmaps actually are reasonable from a medical perspective.
```

##### [24-02-04] [paper416]
- Regression-Based Deep-Learning Predicts Molecular Biomarkers From Pathology Slides
 [[pdf]](https://arxiv.org/abs/2304.05153) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Regression-Based%20Deep-Learning%20Predicts%20Molecular%20Biomarkers%20From%20Pathology%20Slides.pdf)
- `arxiv, 2023-04`
- [Computational Pathology]
```
Well-written and interesting paper. I didn't quite follow all medical details, but I still enjoyed reading it.
```

##### [24-09-12] [paper415]
- Screen Them All: High-Throughput Pan-Cancer Genetic and Phenotypic Biomarker Screening from H&E Whole Slide Images
 [[pdf]](https://arxiv.org/abs/2408.09554) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Screen%20Them%20All%20High-Throughput%20Pan-Cancer%20Genetic%20and%20Phenotypic%20Biomarker%20Screening%20from%20H%26E%20Whole%20Slide%20Images.pdf)
- `arxiv, 2024-08`
- [Computational Pathology]
```
Fairly interesting paper. Interesting method and problem overall, but not my favorite type of paper (a ton of very detailed results in Section 2). You need a stronger medical background in order to really understand and appreciate this. The discussion in Section 3 is quite interesting, but difficult for me to judge if the model performance actually is good enough to be used for the different potential real-world applications they describe.
```

##### [24-09-06] [paper414]
- Benchmarking Foundation Models as Feature Extractors For Weakly-Supervised Computational Pathology
 [[pdf]](https://arxiv.org/abs/2408.15823) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Benchmarking%20Foundation%20Models%20as%20Feature%20Extractors%20For%20Weakly-Supervised%20Computational%20Pathology.pdf)
- `arxiv, 2024-08`
- [Computational Pathology]
```
Interesting paper. It made me think a lot, in a good way. Definitely a solid benchmark, and a bit surprising that CONCH performs that well. "A key insight of our study is that performance of foundation models does not scale well with increasing numbers of images in the training set used for self-supervised learning... Rather, the diversity of the training set suggests to be a key factor": I agree that this probably is true, that it's time to move beyond just trying to train foundation models on more and more data, we need to actually study more carefully WHAT and WHAT TYPE of data we should use. However, I'm not sure that the quite strong claims in this paper are fully supported, it's difficult to see any clear trends in Figure 3 or Figure S5. I think we need to do various controlled experiments, training the same model on increasingly large subsets of the same underlying dataset, or step by step increase the number of patients, number of tissue types etc in the datasets. There are still many open questions, a lot of more careful analysis is still needed.
```

##### [24-08-29] [paper413]
- Benchmarking Embedding Aggregation Methods in Computational Pathology: A Clinical Data Perspective
 [[pdf]](https://arxiv.org/abs/2407.07841) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Benchmarking%20Embedding%20Aggregation%20Methods%20in%20Computational%20Pathology%20A%20Clinical%20Data%20Perspective.pdf)
- `arxiv, 2024-07`
- [Computational Pathology]
```
Well-written, solid and quite interesting paper. "Based on these findings, while it is clear that pathology FMs provide superior performance, it is not possible to recommend any particular aggregation method. We suggest using AB-MIL as a strong baseline and validate other methods on a case by case basis" is a very good summary. Would indeed be interesting to see more advanced tasks than just binary classification in this type of evaluation, for example survival analysis. Could be that spatially-aware methods actually would outperform ABMIL in such settings.
```

##### [24-08-22] [paper412]
- A Clinical Benchmark of Public Self-Supervised Pathology Foundation Models
 [[pdf]](https://arxiv.org/abs/2407.06508) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Clinical%20Benchmark%20of%20Public%20Self-Supervised%20Pathology%20Foundation%20Models.pdf)
- `arxiv, 2024-07`
- [Computational Pathology]
```
Interesting paper. I liked basically everything except the scaling law evaluations in Fig 3 and 4, I think it's very difficult to say anything conclusive when the models being compared are pretrained on completely different datasets. The comparison of their own SP22M and SP85M models is fair, since they are trained on the same datasets. The discussion is interesting. I feel like the overall sentiment of this paper is that, OK, it's time to move beyond just trying to train foundation models on more and more data (trying to be the first reaching 100k slides, 1 million slides etc.), we need to actually study more carefully WHAT and WHAT TYPE of data we should use ("this suggests that the composition of the pretraining dataset may be crucial"). And, I definitely agree with this sentiment, there are many open questions remaining.
```

##### [24-06-23] [paper411]
- Multimodal Prototyping for Cancer Survival Prediction
 [[pdf]](https://openreview.net/pdf?id=3MfvxH3Gia) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Multimodal%20Prototyping%20for%20Cancer%20Survival%20Prediction.pdf)
- `ICML 2024`
- [Computational Pathology], [Survival Analysis]
```
Well-written and quite interesting paper. Basically, they apply the prototype-based slide representation from "Morphological Prototyping for Unsupervised Slide Representation Learning in Computational Pathology" to the survival analysis model in "Modeling Dense Multimodal Interactions Between Biological Pathways and Histology for Survival Prediction" (SurvPath), two CVPR 2024 papers from the same group. The main thing is that the compact prototype-based slide representation now allows them to use standard attention without any approximations, and also to train the survival model using the Cox partial log-likelihood loss - instead of the discrete NLL with batch size = 1 used in SurvPath. In fact, if they use the NLL loss with batch size = 1, they even get slightly worse performance than SurvPath (0.621 vs 0.629)? I.e., this seems to be the main/only thing that improves over SurvPath? Still a very solid paper though, and it makes a lot of sense to combine their two previous papers.
```

##### [24-06-22] [paper410]
- Modeling Dense Multimodal Interactions Between Biological Pathways and Histology for Survival Prediction
 [[pdf]](https://arxiv.org/abs/2304.06819) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Modeling%20Dense%20Multimodal%20Interactions%20Between%20Biological%20Pathways%20and%20Histology%20for%20Survival%20Prediction.pdf)
- `CVPR 2024`
- [Computational Pathology], [Survival Analysis]
```
Well-written and quite interesting paper, I enjoyed reading it. The method makes sense overall, they describe it well. The results actually seem quite impressive, relevant baselines and they do get a relatively clear bump in performance. Difficult for me to judge how actionable the interpretability results in Section 4.5 actually are though.
```

##### [24-03-03] [paper409]
- Diffusion Models for Out-of-Distribution Detection in Digital Pathology
 [[pdf]](https://doi.org/10.1016/j.media.2024.103088) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Diffusion%20Models%20for%20Out-of-Distribution%20Detection%20in%20Digital%20Pathology.pdf)
- `Medical Image Analysis, 2024`
- [Computational Pathology], [Out-of-Distribution Detection], [Diffusion Models]
```
Interesting paper overall, but I got a bit lost in all the details. Not my favourite type of paper perhaps (I might also have been a bit too tired when reading). The overall idea of using diffusion models for reconstruction-based OOD detection is definitely interesting though.
```

##### [24-02-18] [paper408]
- Artificial Intelligence to Identify Genetic Alterations in Conventional Histopathology
 [[pdf]](https://pathsocjournals.onlinelibrary.wiley.com/doi/full/10.1002/path.5898) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Artificial%20Intelligence%20to%20Identify%20Genetic%20Alterations%20in%20Conventional%20Histopathology.pdf)
- `Journal of Pathology, 2022`
- [Computational Pathology]
```
Well-written and interesting paper, I enjoyed reading it. Gives a very good background on various biomarkers. I definitely didn't understand all the medical details, but still found it interesting and useful.
```

##### [24-02-11] [paper407]
- Transcriptome-Wide Prediction of Prostate Cancer Gene Expression From Histopathology Images Using Co-Expression-Based Convolutional Neural Networks
 [[pdf]](https://academic.oup.com/bioinformatics/article/38/13/3462/6589889) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Transcriptome-Wide%20Prediction%20of%20Prostate%20Cancer%20Gene%20Expression%20From%20Histopathology%20Images%20Using%20Co-Expression-Based%20Convolutional%20Neural%20Networks.pdf)
- `Bioinformatics, 2022`
- [Computational Pathology]
```
Well written and quite interesting paper. Helped me understand the general problem a bit better, good background material for me. I definitely didn't understand all the medical stuff, but still found e.g. everything in the Discussion interesting and useful.
```

##### [24-02-04] [paper406]
- Assessing and Enhancing Robustness of Deep Learning Models with Corruption Emulation in Digital Pathology
 [[pdf]](https://arxiv.org/abs/2310.20427) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Assessing%20and%20Enhancing%20Robustness%20of%20Deep%20Learning%20Models%20with%20Corruption%20Emulation%20in%20Digital%20Pathology.pdf)
- `arxiv, 2023-10`
- [Computational Pathology]
```
Interesting and quite well-written paper. Quite short, and basically no implementation details are given for the various corruptions. It does seem potentially useful though, for both benchmarking and augmentation. The results in Table 3 seem quite impressive.
```

##### [24-02-03] [paper405]
- Uncertainty Sets for Image Classifiers using Conformal Prediction
 [[pdf]](https://openreview.net/forum?id=eNdiU_DbM9) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20Sets%20for%20Image%20Classifiers%20using%20Conformal%20Prediction.pdf)
- `ICLR 2021`
- [Uncertainty Estimation]
```
Quite interesting and well-written paper. The proposed method mostly makes sense, and it does indeed seem to produce smaller prediction sets than the baselines. Not sure how relevant this is for me though.
```

##### [24-02-02] [paper404]
- Estimating Diagnostic Uncertainty in Artificial Intelligence Assisted Pathology Using Conformal Prediction
 [[pdf]](https://www.nature.com/articles/s41467-022-34945-8) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Estimating%20Diagnostic%20Uncertainty%20in%20Artificial%20Intelligence%20Assisted%20Pathology%20Using%20Conformal%20Prediction.pdf)
- `Nature Communications, 2022`
- [Computational Pathology], [Uncertainty Estimation]
```
Interesting paper, but I found it quite difficult to understand. I was confused by the employed conformal prediction method, it did not make sense to me that it could output empty predictions for some inputs. "efficiency, defined as the fraction of all predictions resulting in a correct single-label prediction" is not something I've seen before either. The dataset setup is neat though, with test sets from the same scanner/lab, from a different scanner, and from a different scanner and lab. Figure 2 is interesting, shows that the model becomes significantly overconfident on test set 3-5.
```

##### [24-01-21] [paper403]
- Improving Trustworthiness of AI Disease Severity Rating in Medical Imaging with Ordinal Conformal Prediction Sets
 [[pdf]](https://arxiv.org/abs/2207.02238) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Improving%20Trustworthiness%20of%20AI%20Disease%20Severity%20Rating%20in%20Medical%20Imaging%20with%20Ordinal%20Conformal%20Prediction%20Sets.pdf)
- `MICCAI 2022`
- [Uncertainty Estimation], [ML for Medicine/Healthcare]
```
Quite well-written and interesting paper, not overly impressed. I don't fully understand how the method is implemented in practice, Algorithm 1 makes sense, but how does one compute the value of lambda?
```

##### [24-01-20] [paper402]
- Hierarchical Vision Transformers for Context-Aware Prostate Cancer Grading in Whole Slide Images
 [[pdf]](https://arxiv.org/abs/2312.12619) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Hierarchical%20Vision%20Transformers%20for%20Context-Aware%20Prostate%20Cancer%20Grading%20in%20Whole%20Slide%20Images.pdf)
- `NeurIPS Workshops 2023`
- [Computational Pathology], [Transformers]
```
Workshop paper, just ~3 pages. The appendix also contains some useful info. Well-written and interesting paper. The method/model makes intuitive sense. Seemingly strong performance without any ad hoc modifications. Qiute interesting that the local model version (fine-tuning also the region-level transformer) gave such a clear performance boost.
```

##### [24-01-19] [paper401]
- Conformal Prediction Sets for Ordinal Classification
 [[pdf]](https://openreview.net/forum?id=YI4bn6aAmz) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Conformal%20Prediction%20Sets%20for%20Ordinal%20Classification.pdf)
- `NeurIPS 2023`
- [Uncertainty Estimation]
```
Quite well-written paper, interesting overall. The propsed approach is actually very simple in the end, just modify the DNN output before the softmax layer according to eq. (5), train it using standard cross-entropy, and then apply the existing conformal prediction method APS (or LAC) at test-time to output prediction sets (if I understood everything correclty). The results seem reasonable. Tables and figures could be made to look a bit better.
```

##### [24-01-19] [paper400]
- Artificial Intelligence for Diagnosis and Gleason Grading of Prostate Cancer: The PANDA Challenge
 [[pdf]](https://www.nature.com/articles/s41591-021-01620-2) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Artificial%20Intelligence%20for%20Diagnosis%20and%20Gleason%20Grading%20of%20Prostate%20Cancer%20The%20PANDA%20Challenge.pdf)
- `Nature Medicine, 2022`
- [Computational Pathology]
```
Well-written and interesting paper. Neat/cool/impressive study/challenge design, it makes sense to evaluate the top-performing mehtods on external data afterwards.
```

##### [24-06-16] [paper399]
- Morphological Prototyping for Unsupervised Slide Representation Learning in Computational Pathology
 [[pdf]](https://arxiv.org/abs/2405.11643) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Morphological%20Prototyping%20for%20Unsupervised%20Slide%20Representation%20Learning%20in%20Computational%20Pathology.pdf)
- `CVPR 2024`
- [Computational Pathology]
```
Interesting and very well-written paper, I enjoyed reading it. Figure 2 gives a great overview of their approach. The visualizations in Fig 1, 3, S1 - S4 are really neat.
```

##### [24-06-12] [paper398]
- Prediction of Recurrence Risk in Endometrial Cancer with Multimodal Deep Learning
 [[pdf]](https://www.nature.com/articles/s41591-024-02993-w) [[annotated pdf]](https://drive.google.com/file/d/1dT78m-EuN0n7h7jmtnrvi13BoQF42wQR/view?usp=sharing)
- `Nature Medicine, 2024`
- [Computational Pathology]
```
Well-written and somewhat interesting paper. Not quite my type of paper, would probably need a bit stronger medical background. The method sort of seems unnecessarily complicated to me, using a second frozen model, embedding layers etc. Yes, they see some gains in ablations, but still feels like doing something simpler also could work well. The experiment on adjuvant chemotherapy response prediction is interesting.
```

##### [24-06-11] [paper397]
- A Whole-Slide Foundation Model for Digital Pathology from Real-World Data
 [[pdf]](https://www.nature.com/articles/s41586-024-07441-w) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Whole-Slide%20Foundation%20Model%20for%20Digital%20Pathology%20from%20Real-World%20Data.pdf)
- `Nature, 2024`
- [Computational Pathology]
```
Well-written and interesting paper, I quite enjoyed reading it. ~1.3 billion 256x256 patches, from ~170k WSIs, from ~30k patients. 45% of the slides are from lung tissue, 30% from bowel, 9% from CNS/brain, 3% from breast (Suppl. Fig 1). I didn't look too too carefully at the results, I was mostly just interested in the data and method, but seems reasonable. The vision-language experiments are interesting, the fact that they to do this contrastive alignment on the slide level, they use ~17k WSI - pathology report pairs.
```

##### [24-06-09] [paper396]
- Multistep Distillation of Diffusion Models via Moment Matching
 [[pdf]](https://arxiv.org/abs/2406.04103) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Multistep%20Distillation%20of%20Diffusion%20Models%20via%20Moment%20Matching.pdf)
- `arxiv, 2024-06`
- [Diffusion Models]
```
Fairly interesting and well-written paper. I should probably have read a more basic paper about diffusion model distillation instead, don't think I was able to fully appreciate the details here. Struggled to properly follow everything in Section 3.2. The proposed method seems somewhat ad hoc, but also seems to work well in practice. Quite interesting to see an example of a SOTA distillation method at least.
```

##### [24-06-08] [paper395]
- Flow Matching for Generative Modeling
 [[pdf]](https://openreview.net/forum?id=PqvMRDCJT9t) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Flow%20Matching%20for%20Generative%20Modeling.pdf)
- `ICLR 2023`
- [Diffusion Models]
```
Well-written and interesting paper, I've been meaning to read this for quite some time now. I struggled a bit to follow Section 3 and 4, but overall the method makes sense I think. Would need to read more, and probably also discuss this with someone, to properly understand and appreciate all the details.
```

##### [24-05-29] [paper394]
- No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance
 [[pdf]](https://arxiv.org/abs/2404.04125) [[annotated pdf]](https://drive.google.com/file/d/13t_Mvf4Jv7ulkzfvB05Kq_ydoBE2HXwN/view?usp=sharing)
- `arxiv, 2024-04`
- [Vision-Language Models]
```
Well-written and quite interesting paper. These findings are quite expected though, right? CLIP works well on a variety of downstream tasks because it's trained on a huge dataset containing all kind of "concepts"? For it to perform well on a certain task it needs to be represented in the pretraining dataset somehow? But still, quite interesting paper that I liked overall.
```

##### [24-05-23] [paper393]
- BioFusionNet: Deep Learning-Based Survival Risk Stratification in ER+ Breast Cancer Through Multifeature and Multimodal Data Fusion
 [[pdf]](https://arxiv.org/abs/2402.10717) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/BioFusionNet%20Deep%20Learning-Based%20Survival%20Risk%20Stratification%20in%20ER%2B%20Breast%20Cancer%20Through%20Multifeature%20and%20Multimodal%20Data%20Fusion.pdf)
- `arxiv, 2024-02`
- [Computational Pathology], [Survival Analysis]
```
Somewhat interesting paper. Well-written overall. Section 1 and Figure 1 are interesting, give good background on breast cancer diagnosis/treatment. Figure 2 and 3 illustrate the method in a nice way. However, the method just seems unnecessarily complicated to me. Why fuse three different feature extractors via VAE? Doesn't seem to help that much according to Table 5? And, they only have 249 cases in total for train/val, which means that the number of events is very quite low. Also, in Table 5, seems like most of the performance gain comes from the weighted Cox loss, which doesn't quite seem reasonable. It definitely makes sense to fuse image data with genetic/clinical data, I just feel like it must be possible to see performance gains using more straightforward approaches.
```

##### [24-05-15] [paper392]
- CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection
 [[pdf]](https://arxiv.org/abs/2301.00785) [[annotated pdf]](https://drive.google.com/file/d/1IEgM9wuMIO3QMzCjHJrjylaQI8iDtrbS/view?usp=sharing)
- `ICCV 2023`
- [Vision-Language Models]
```
Somewhat interesting paper. I found it quite difficult to understand their proposed model in Section 3.2 / Figure 2. Multiple design choices here also seem somewhat arbitrary, they're not really motivated or compared with simpler baselines.
```

##### [24-05-14] [paper391]
- A Foundational Multimodal Vision Language AI Assistant for Human Pathology
 [[pdf]](https://arxiv.org/abs/2312.07814) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Foundational%20Multimodal%20Vision%20Language%20AI%20Assistant%20for%20Human%20Pathology.pdf)
- `arxiv, 2023-12`
- [Computational Pathology], [Vision-Language Models]
```
Well-written and interesting paper, I quite enjoyed reading it. The model is actually quite conceptually simple (take the UNI vision encoder, fine-tune it on the image-caption data from the CONCH paper, then combine it with the Llama 2 LLM to form an MLLM, which is fine-tuned on ~250k instructions). This type of application is of course a bit more of a long shot, it'll probably take a while (and a lot of thorough validation work) before a model like this actually is used in real-world clinical practice. But, it's still pretty cool and an interesting technical problem.
```

##### [24-04-25] [paper390]
- A Visual-Language Foundation Model for Computational Pathology
 [[pdf]](https://www.nature.com/articles/s41591-024-02856-4) [_unfortunately not open access, thus no annotated pdf_]
- `Nature Medicine, 2024`
- [Computational Pathology], [Vision-Language Models]
```
Well-written and interesting paper, I quite enjoyed reading it. Contains a lot of details though, so I had to skim some of the Methods sections. ~1.17 million image-caption pairs, so perhaps a bit less data than I expected. But interesting/neat/cool that it's even possible to create this type of dataset from paper figures + figure captions. Interesting that only ~400k of these images are H&E, the rest are IHC + special (although I don't quite know what "special" means here).
```

##### [24-04-11] [paper389]
- Towards a General-Purpose Foundation Model for Computational Pathology
 [[pdf]](https://www.nature.com/articles/s41591-024-02857-3) [_unfortunately not open access, thus no annotated pdf_]
- `Nature Medicine, 2024`
- [Computational Pathology]
```
Interesting paper, well-written, I enjoyed reading it. Extensive evaluation, everything is very thoroughly done. ViT-L model trained using DINOv2, on ~100 million patches extracted from ~100k WSIs.
```

##### [24-05-12] [paper388]
- The Clinician-AI Interface: Intended Use and Explainability in FDA-Cleared AI Devices for Medical Image Interpretation
 [[pdf]](https://www.nature.com/articles/s41746-024-01080-1) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/The%20Clinician-AI%20Interface%20Intended%20Use%20and%20Explainability%20in%20FDA-Cleared%20AI%20Devices%20for%20Medical%20Image%20Interpretation.pdf)
- `npj Digital Medicine, 2024 (Brief Communication)`
- [ML for Medicine/Healthcare]
```
Short paper, just ~3.5 pages. Somewhat interesting.
```

##### [24-05-12] [paper387]
- Overcoming Limitations in Current Measures of Drug Response May Enable AI-Driven Precision Oncology
 [[pdf]](https://www.nature.com/articles/s41698-024-00583-0) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Overcoming%20Limitations%20in%20Current%20Measures%20of%20Drug%20Response%20May%20Enable%20AI-Driven%20Precision%20Oncology.pdf)
- `npj Precision Oncology, 2024 (Brief Communication)`
- [ML for Medicine/Healthcare]
```
 Quite short paper, ~5.5 pages. Quite well-written, somewhat interesting.
```

##### [24-05-11] [paper386]
- Self-Supervised Attention-Based Deep Learning for Pan-Cancer Mutation Prediction from Histopathology
 [[pdf]](https://www.nature.com/articles/s41698-023-00365-0) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Self-Supervised%20Attention-Based%20Deep%20Learning%20for%20Pan-Cancer%20Mutation%20Prediction%20from%20Histopathology.pdf)
- `npj Precision Oncology, 2023 (Brief Communication)`
- [Computational Pathology]
```
A short paper, just 3.5 pages. Quite interesting and well-written.
```

##### [24-04-25] [paper385]
- Understanding Deep Learning
 [[pdf]](https://udlbook.github.io/udlbook/) [[annotated pdf]](https://drive.google.com/file/d/1RIFN9VJ6cUo3PvJ9fz5z_1lTngERzO8v/view?usp=sharing)
- `The MIT Press, 2023`
- [Miscellaneous]
```
A book that we read over multiple weeks in a reading group, 1-2 chapters per week. Chapters that I liked / found more useful than the rest: 1, 2, 8, 9, 12, 14, 15, 17, 18, 21. 

Overall, the book contains a lot of neat/interesting/illustrative figures that really help to explain various concepts and methods. While I already knew most of what's covered in the book, it still provided a number of quite interesting/neat insights on various topics. Also, I definitely found the chapter about transformers useful (this is something that I sort of knew, but reading the chapter made me realize that I've never really taken the time to properly go through all the basics).

Overall, I quite enjoyed going through the entire book. I took quite a lot of time of course, but I do think that it probably was worth it. I think it was quite useful, gave me a more thorough understanding of quite a few things.
```

##### [24-03-28] [paper384]
- A Good Feature Extractor Is All You Need for Weakly Supervised Pathology Slide Classification
 [[pdf]](https://arxiv.org/abs/2311.11772) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Good%20Feature%20Extractor%20Is%20All%20You%20Need%20for%20Weakly%20Supervised%20Pathology%20Slide%20Classification.pdf)
- `arxiv, 2023-11`
- [Computational Pathology]
```
Interesting paper, but I was somewhat confused in the end. Also, I didn't really have enough time to properly go through the details. I liked Section 1 and 2, but then it became less clear. The conclusion in Section 4.2 seems strange to me, that they found no clear difference with/without stain normalization for ~any~ feature extractor, not even for the ImageNet extractors. This doesn't really make sense to me. Thus, I'm not quite sure what to do with their main conclusions/recommendations.
```

##### [24-03-23] [paper383]
- A Systematic Pan-Cancer Study on Deep Learning-Based Prediction of Multi-Omic Biomarkers From Routine Pathology Images
 [[pdf]](https://www.nature.com/articles/s43856-024-00471-5) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Systematic%20Pan-Cancer%20Study%20on%20Deep%20Learning-Based%20Prediction%20of%20Multi-Omic%20Biomarkers%20From%20Routine%20Pathology%20Images.pdf)
- `Communications Medicine, 2024`
- [ML for Medicine/Healthcare], [Computational Pathology]
```
Well-written overall, fairly interesting. Fully appreciating this paper definitely requires a stronger medical background than I have, but I still found parts of it quite useful/interesting.
```

##### [24-03-23] [paper382]
- The Future of Artificial Intelligence in Digital Pathology - Results of a Survey Across Stakeholder Groups
 [[pdf]](https://onlinelibrary.wiley.com/doi/10.1111/his.14659) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/The%20Future%20of%20Artificial%20Intelligence%20in%20Digital%20Pathology%20Results%20of%20a%20Survey%20Across%20Stakeholder%20Groups.pdf)
- `Histopathology, 2022`
- [ML for Medicine/Healthcare], [Computational Pathology]
```
Short paper of just 5 pages, very quick to read. Somewhat interesting, the main takeaway is at least clear: "The prediction of treatment response directly from routine pathology slides is regarded as the most promising future application", "Prediction of genetic alterations, gene expression and survival directly from routine pathology images scored consistently high throughout subgroups".
```

##### [24-03-13] [paper381]
- Generalizable Biomarker Prediction From Cancer Pathology Slides With Self-Supervised Deep Learning: A Retrospective Multi-Centric Study
 [[pdf]](https://www.sciencedirect.com/science/article/pii/S2666379123000861?via%3Dihub) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Generalizable%20Biomarker%20Prediction%20From%20Cancer%20Pathology%20Slides%20With%20Self-Supervised%20Deep%20Learning%20A%20Retrospective%20Multi-Centric%20Study.pdf)
- `Cell Reports Medicine, 2023`
- [ML for Medicine/Healthcare], [Computational Pathology]
```
Well-written and quite interesting paper. This general SSL-attMIL model approach definitely seems reasonable. Not super impressed by the scale of the experiments though perhaps, just two different datasets/sites with ~2k patients each (this is of course much much better than just having a single site, but still).
```

##### [24-03-13] [paper380]
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
 [[pdf]](https://arxiv.org/abs/2010.11929) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/An%20Image%20is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale.pdf)
- `ICLR 2021`
- [Transformers]
```
Quite interesting paper. I read this after having read the chapter on transformers in the "Understanding Deep Learning" book. The method is conceptually very simple, which is neat. I didn't really know that the ViT model is this straightforward. So, it was a good decision to read this paper at this time.
```

##### [24-03-08] [paper379]
- All Models Are Wrong and Yours Are Useless: Making Clinical Prediction Models Impactful for Patients
 [[pdf]](https://www.nature.com/articles/s41698-024-00553-6) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/All%20Models%20Are%20Wrong%20and%20Yours%20Are%20Useless%20Making%20Clinical%20Prediction%20Models%20Impactful%20for%20Patients.pdf)
- `npj Precision Oncology, 2024`
- [ML for Medicine/Healthcare]
```
Comment, just ~2 pages long. Well-written and quite interesting, definitely worth reading. The five observations and the checklist in box 1 make sense overall ("Do you address a clear clinical decision point? Are you really, really sure?"). Made me think quite a lot about the different roles academic researchers and industry/companies play / should play in this. Not quite sure how much of the nitty gritty engineering work (required to make an actual clinical tool) academic researchers can be expected to take on? But on the other hand, of course, if you actually want your research to have real-world impact, you might simply need to be prepared to take on at least some of this engineering (and less interesting/exciting) work.
```

##### [24-01-14] [paper378]
- Autosurv: Interpretable Deep Learning Framework for Cancer Survival Analysis Incorporating Clinical and Multi-Omics Data
 [[pdf]](https://www.nature.com/articles/s41698-023-00494-6) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Autosurv%20Interpretable%20Deep%20Learning%20Framework%20for%20Cancer%20Survival%20Analysis%20Incorporating%20Clinical%20and%20Multi-Omics%20Data.pdf)
- `npj Precision Oncology, 2024`
- [ML for Medicine/Healthcare], [Survival Analysis]
```
Somewhat interesting paper. I skimmed through quite large parts of it, those which contained very detailed descriptions of various results. One layer deepSurv network for survival analysis, i.e. trained with the Cox partial likelihood.
```

##### [24-01-13] [paper377]
- Learning Individual Survival Models from PanCancer Whole Transcriptome Data
 [[pdf]](https://aacrjournals.org/clincancerres/article/29/19/3924/729105/Learning-Individual-Survival-Models-from-PanCancer) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20Individual%20Survival%20Models%20from%20PanCancer%20Whole%20Transcriptome%20Data.pdf)
- `Clinical Cancer Research, 2023`
- [ML for Medicine/Healthcare], [Survival Analysis]
```
Well written and interesting papers. Neat approach, to compress the ~16k dimensional gene expressions into 100 dimensional features. I just skimmed the "Interpretation of NMF Factors" and "Biological Interpretation of NMF Representation" sections, I didn't understand this anyway.
```

##### [24-01-12] [paper376]
- Accurate Personalized Survival Prediction for Amyotrophic Lateral Sclerosis Patients
 [[pdf]](https://www.nature.com/articles/s41598-023-47935-7) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Accurate%20Personalized%20Survival%20Prediction%20for%20Amyotrophic%20Lateral%20Sclerosis%20Patients.pdf)
- `Scientific Reports, 2023`
- [ML for Medicine/Healthcare], [Survival Analysis]
```
Interesting and very well written paper, I enjoyed reading it. Relatively short paper (basically all technical details seem to be in the supplementary material, which I didn't read), quick and easy to read.
```

##### [24-01-12] [paper375]
- CenTime: Event-Conditional Modelling of Censoring in Survival Analysis
 [[pdf]](https://www.sciencedirect.com/science/article/pii/S1361841523002761) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/CenTime%20Event-Conditional%20Modelling%20of%20Censoring%20in%20Survival%20Analysis.pdf)
- `Medical Image Analysis, 2024`
- [ML for Medicine/Healthcare], [Survival Analysis]
```
Fairly interesting paper, well written overall. Quite quick to read. I don't fully understand Section 2, difficult for me to judge how impactful this contribution is. The discretized Gaussian model seems somewhat odd to me.
```

##### [24-01-12] [paper374]
- A Self-Supervised Vision Transformer to Predict Survival From Histopathology in Renal Cell Carcinoma
 [[pdf]](https://link.springer.com/article/10.1007/s00345-023-04489-7) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Self-Supervised%20Vision%20Transformer%20to%20Predict%20Survival%20From%20Histopathology%20in%20Renal%20Cell%20Carcinoma.pdf)
- `World Journal of Urology, 2023`
- [Computational Pathology], [Survival Analysis]
```
Fairly interesting paper, well written overall. Quick to read. The results don't seem overly impressive (with significant high/low risk stratification for overall survival only on the train set)?
```

##### [24-01-11] [paper373]
- Conformalized Survival Analysis with Adaptive Cutoffs
 [[pdf]](https://arxiv.org/abs/2211.01227) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Conformalized%20Survival%20Analysis%20with%20Adaptive%20Cutoffs.pdf)
- `Biometrika, 2023`
- [Survival Analysis]
```
Well written and interesting paper. Not enitirely sure if this actually would be useufl in practice, but definitely an interesting potential approach for evaluating time-to-event prediction models.
```

##### [24-01-10] [paper372]
- Learning Accurate Personalized Survival Models for Predicting Hospital Discharge and Mortality of COVID-19 Patients
 [[pdf]](https://www.nature.com/articles/s41598-022-08601-6) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20Accurate%20Personalized%20Survival%20Models%20for%20Predicting%20Hospital%20Discharge%20and%20Mortality%20of%20COVID-19%20Patients.pdf)
- `Scientific Reports, 2022`
- [ML for Medicine/Healthcare], [Survival Analysis]
```
Interesting and overall well-written paper. The description of the experimental results was a bit confusing, I don't think that's entirely correct. The discussion is interesting.
```

##### [24-01-10] [paper371]
- Proper Scoring Rules for Survival Analysis
 [[pdf]](https://arxiv.org/abs/2305.00621) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Proper%20Scoring%20Rules%20for%20Survival%20Analysis.pdf)
- `ICML 2023`
- [Survival Analysis]
```
Well-written and somewhat interesting paper. Section 1 - Section 4 gave some quite good background, but then I got a bit confused and sceptical in Section 5 - 6. Not sure what the main practical implication/takeaway for me should be.
```

##### [24-01-10] [paper370]
- An Effective Meaningful Way to Evaluate Survival Models
 [[pdf]](https://arxiv.org/abs/2306.01196) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/An%20Effective%20Meaningful%20Way%20to%20Evaluate%20Survival%20Models.pdf)
- `ICML 2023`
- [Survival Analysis]
```
Quite interesting and well-written paper. The specific proposed methods/metrics are perhaps not overly interesting, but I found the paper really useful to read. Good descriptions of and interesting discussions about how various metrics relate to eachother. Appendix C is definitely useful.
```

##### [24-01-09] [paper369]
- Uncertainty Estimation in Cancer Survival Prediction
 [[pdf]](https://arxiv.org/abs/2003.08573) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20Estimation%20in%20Cancer%20Survival%20Prediction.pdf)
- `ICLR Workshops 2020`
- [Survival Analysis]
```
Workshop paper, just 5 pages long. Quick to read. I didn't find the details for the specific used model etc. overly interetsing, but the overall aim of the paper + the way they created OOD datsets is interesting/neat.
```

##### [24-01-09] [paper368]
- Maximum Likelihood Estimation of Flexible Survival Densities with Importance Sampling
 [[pdf]](https://arxiv.org/abs/2311.01660) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Maximum%20Likelihood%20Estimation%20of%20Flexible%20Survival%20Densities%20with%20Importance%20Sampling.pdf)
- `MLHC 2023`
- [Survival Analysis]
```
Quite well-written and quite interesting paper. The proposed method is conceptually straightforward and it's neat that it works that well to just approximate the integrals etc. But, I think this would be fairly inconvenient to use in practice to actually output predictions?
```

##### [24-01-08] [paper367]
- Survival Mixture Density Networks
 [[pdf]](https://arxiv.org/abs/2208.10759) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Survival%20Mixture%20Density%20Networks.pdf)
- `MLHC 2022`
- [Survival Analysis]
```
Well-written and interesting paper. Good background and descriptions for the survival analysis problem, everything is explained well, I enjoyed reading it (at least up to and including Section 5.2).
```

##### [24-01-08] [paper366]
- Using Bayesian Neural Networks to Select Features and Compute Credible Intervals for Personalized Survival Prediction
 [[pdf]](https://ieeexplore.ieee.org/document/10158019) [_unfortunately not open access, thus no annotated pdf_]
- `Transactions on Biomedical Engineering, 2023`
- [Survival Analysis]
```
Interesting and quite well-written paper. The specific proposed method (variational BNN with different priors) seems unnecessarily complex to me, but the overall approach and aims are definitely interesting. 
```

##### [24-01-07] [paper365]
- Calibration and Uncertainty in Neural Time-to-Event Modeling
 [[pdf]](https://ieeexplore.ieee.org/document/9244076) [_unfortunately not open access, thus no annotated pdf_]
- `Transactions on Neural Networks and Learning Systems, 2020`
- [Survival Analysis]
```
Well-written paper overall, but not particularly interesting for me. The proposed method seems quite complex.
```

##### [24-01-07] [paper364]
- Calibration: The Achilles Heel of Predictive Analytics
 [[pdf]](https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-019-1466-7) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Calibration%20The%20Achilles%20Heel%20of%20Predictive%20Analytics.pdf)
- `BMC Medicine, 2019`
- [ML for Medicine/Healthcare]
```
Opinion paper, a bit shorter than usual. Somewhat interesting. A couple of quite neat arguments for (and examples of) why calibration is important.
```

##### [24-01-06] [paper363]
- Censored Quantile Regression Neural Networks for Distribution-Free Survival Analysis
 [[pdf]](https://arxiv.org/abs/2205.13496) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Censored%20Quantile%20Regression%20Neural%20Networks%20for%20Distribution-Free%20Survival%20Analysis.pdf)
- `NeurIPS 2022`
- [Survival Analysis] 
```
Interesting and very well-written paper, I enjoyed reading it. The experimental evaluation, with multiple datasets of three different types and so on, is neat.
```

##### [24-01-06] [paper362]
- Estimating Calibrated Individualized Survival Curves with Deep Learning
 [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/16098) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Estimating%20Calibrated%20Individualized%20Survival%20Curves%20with%20Deep%20Learning.pdf)
- `AAAI 2021`
- [Survival Analysis] 
```
Well-written and interesting paper. I'm not quite sold on the overall approach (to output probabilities for a discrete set of time points etc.), and not overly impressed by the experimental evaluation, but it was interesting, quite happy that I decided to read it. 
```

##### [24-01-05] [paper361]
- Neural Frailty Machine: Beyond Proportional Hazard Assumption in Neural Survival Regressions
 [[pdf]](https://arxiv.org/abs/2303.10358) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Frailty%20Machine%20Beyond%20Proportional%20Hazard%20Assumption%20in%20Neural%20Survival%20Regressions.pdf)
- `NeurIPS 2023`
- [Survival Analysis] 
```
Well-written and interesting paper. The propsed mehtod is interesting, even though the results not exactly are revolutionary (seems difficult in general to come up with a method that clearly outperforms all baselines across these types of datasets). 
```

##### [24-01-05] [paper360]
- Deep Extended Hazard Models for Survival Analysis
 [[pdf]](https://proceedings.neurips.cc/paper/2021/hash/7f6caf1f0ba788cd7953d817724c2b6e-Abstract.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Extended%20Hazard%20Models%20for%20Survival%20Analysis.pdf)
- `NeurIPS 2021`
- [Survival Analysis] 
```
Interesting and very well-written paper. Section 1 and 2 are great, they provide a very thorough introduction and background for survival analysis, Cox proportional hazard models etc!
```

##### [24-01-05] [paper359]
- X-CAL: Explicit Calibration for Survival Analysis
 [[pdf]](https://arxiv.org/abs/2101.05346) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/X-CAL%20Explicit%20Calibration%20for%20Survival%20Analysis.pdf)
- `NeurIPS 2020`
- [Survival Analysis] 
```
Somewhat interesting paper. The abstract and introduction was interesting, but then I struggled to properly understand Section 2 - 3. Not overly impressed by the results.
```

##### [24-01-04] [paper358]
- DeepSurv: Personalized Treatment Recommender System Using a Cox Proportional Hazards Deep Neural Network
 [[pdf]](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/DeepSurv%20Personalized%20Treatment%20Recommender%20System%20Using%20a%20Cox%20Proportional%20Hazards%20Deep%20Neural%20Network.pdf)
- `BMC Medical Research Methodology, 2018`
- [Survival Analysis] 
```
Well-written and interesting paper overall. I wanted to learn some basics for survival analysis, Cox proportional hazards model etc., and this paper provided exaclty that. All the details about the network architecture and the results I then just went through quite quickly.
```

##### [24-03-02] [paper357]
- Quantifying the Effects of Data Augmentation and Stain Color Normalization in Convolutional Neural Networks for Computational Pathology
 [[pdf]](https://arxiv.org/abs/1902.06543) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Quantifying%20the%20Effects%20of%20Data%20Augmentation%20and%20Stain%20Color%20Normalization%20in%20Convolutional%20Neural%20Networks%20for%20Computational%20Pathology.pdf)
- `Medical Image Analysis, 2019`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
Well-written and quite interesting paper. Very practical with actionable takeaways: "we found HSV and HED color transformations to be the key ingredients to improve performance", "We concluded that using the stain color normalization methods evaluated in this paper without proper stain color augmentation is insufficient to reduce the generalization error caused by stain variation", "Based on our empirical evaluation, we found that any type of stain color augmentation, i.e. HSV or HED transformation, should always be used".
```

##### [24-02-29] [paper356]
- Pan-Cancer Integrative Histology-Genomic Analysis via Multimodal Deep Learning
 [[pdf]](https://www.sciencedirect.com/science/article/pii/S1535610822003178?via%3Dihub) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Pan-Cancer%20Integrative%20Histology-Genomic%20Analysis%20via%20Multimodal%20Deep%20Learning.pdf)
- `Cancer Cell, 2022`
-  [ML for Medicine/Healthcare], [Computational Pathology], [Survival Analysis]
```
Quite interesting paper. I liked the Introduction and the Discussion. "which suggests that molecular features drive most of the risk prediction in MMF (Figure 2C; Table S3). This substantiates the observation that molecular profiles are more prognostic for survival than WSIs in most cancer types" makes sense and is quite interesting. I am somewhat confused by the "Survival loss function" section in the appendix though, seems a bit strange to discretize the survival time into just four bins? And OK, so the network outputs four values, but how do they get a single "risk score" for each patient then, in order to compute the C-index etc?
```

##### [24-02-25] [paper355]
- Designing Deep Learning Studies in Cancer Diagnostics
 [[pdf]](https://www.nature.com/articles/s41568-020-00327-9) [_unfortunately not open access, thus no annotated pdf_]
- `Nature Reviews Cancer, 2021`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
Well-written and quite interesting paper. The conclusion and proposed evaluation approach make sense overall ("...helps distinguish rigorous, retrospective validation studies from studies that repeatedly evaluated the external cohort and might end up reporting severely biased performance estimates").
```

##### [24-02-17] [paper354]
- A Systematic Analysis of Deep Learning in Genomics and Histopathology for Precision Oncology
 [[pdf]](https://bmcmedgenomics.biomedcentral.com/articles/10.1186/s12920-024-01796-9) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Systematic%20Analysis%20of%20Deep%20Learning%20in%20Genomics%20and%20Histopathology%20for%20Precision%20Oncology.pdf)
- `BMC Medical Genomics, 2024`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
Well-written and fairly interesting paper. I liked the Background section. Quite interesting with some of the discussed trends over time, the different types of studied problems/applications etc. Also, I like that they use the term DL instead of AI.
```

##### [24-02-09] [paper353]
- Artificial Intelligence in Histopathology: Enhancing Cancer Research and Clinical Oncology
 [[pdf]](https://www.nature.com/articles/s43018-022-00436-4) [_unfortunately not open access, thus no annotated pdf_]
- `Nature Cancer, 2022`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
Well-written review paper, interesting and a quite enjoyable read overall. The "Applications in cancer research and diagnostics" section was particularly interesting.
```

##### [24-02-07] [paper352]
- Time-Series Forecasting With Deep Learning: A Survey
 [[pdf]](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0209) [_unfortunately not open access, thus no annotated pdf_]
- `Philosophical Transactions of the Royal Society A, 2021`
-  [Miscellaneous]
```
Well-written overall, somewhat interesting. I quite liked Section 1 and 2.
```

##### [24-02-07] [paper351]
- Physics-Informed Machine Learning and its Structural Integrity Applications: State of the Art
 [[pdf]](https://royalsocietypublishing.org/doi/10.1098/rsta.2022.0406) [_unfortunately not open access, thus no annotated pdf_]
- `Philosophical Transactions of the Royal Society A, 2023`
-  [Miscellaneous]
```
Not particularily interesting. Didn't really feel like I learned that much. Would have preferred if Section 2 contained more specific details, I think.
```

##### [24-02-06] [paper350]
- Differentiable Samplers for Deep Latent Variable Models
 [[pdf]](https://royalsocietypublishing.org/doi/10.1098/rsta.2022.0147) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Differentiable%20Samplers%20for%20Deep%20Latent%20Variable%20Models.pdf)
- `Philosophical Transactions of the Royal Society A, 2023`
-  [Miscellaneous]
```
Well-written overall, somewhat interesting. My background knowledge is definitely not good enough for me to properly understand and appreciate this. I quite liked Section 1 and 5 at least.
```

##### [24-02-04] [paper349]
- RR-CP: Reliable-Region-Based Conformal Prediction for Trustworthy Medical Image Classification
 [[pdf]](https://arxiv.org/abs/2309.04760) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/RR-CP%20Reliable-Region-Based%20Conformal%20Prediction%20for%20Trustworthy%20Medical%20Image%20Classification.pdf)
- `MICCAI Workshops 2023`
-  [ML for Medicine/Healthcare], [Uncertainty Estimation]
```
Quite well-written and fairly interesting paper. The proposed method seems reasonable overall, but it seems somewhat odd to evaluate only with 99.5% desired coverage. Not sure how useful/different this method would be in practice.
```

##### [24-02-03] [paper348]
- Key Challenges for Delivering Clinical Impact with Artificial Intelligence
 [[pdf]](https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-019-1426-2) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Key%20Challenges%20for%20Delivering%20Clinical%20Impact%20with%20Artificial%20Intelligence.pdf)
- `BMC Medicine, 2019`
-  [ML for Medicine/Healthcare]
```
Opinion paper, ~6 pages. Well-written and quite interesting. Gives a good overview of various aspects and potential issues.
```

##### [24-02-03] [paper347]
- Second Opinion Needed: Communicating Uncertainty in Medical Machine Learning
 [[pdf]](https://www.nature.com/articles/s41746-020-00367-3) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Second%20Opinion%20Needed%20Communicating%20Uncertainty%20in%20Medical%20Machine%20Learning.pdf)
- `npj Digital Medicine, 2021`
-  [ML for Medicine/Healthcare], [Uncertainty Estimation]
```
Perspective paper, just 4 pages. Fairly interesting. Not entirely clear who the target audience for this paper is perhaps. Some quite neat arguments for why uncertainty estimation is important/useful within the medical domain, but still mostly things I've seen before.
```

##### [24-02-02] [paper346]
- Predictive Uncertainty Estimation for Out-Of-Distribution Detection in Digital Pathology
 [[pdf]](https://www.sciencedirect.com/science/article/pii/S1361841522002833) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Predictive%20Uncertainty%20Estimation%20for%20Out-Of-Distribution%20Detection%20in%20Digital%20Pathology.pdf)
- `Medical Image Analysis, 2023`
-  [ML for Medicine/Healthcare], [Computational Pathology], [Uncertainty Estimation], [Out-of-Distribution Detection]
```
Quite well-written and fairly interesting paper. I tend to not love these types of OOD detection papers, where methods are evaluated just in terms of how well they can separate IID and OOD datasets. This has always seemed a bit arbitrary to me. The results here are also a bit messy, difficult to say what the main takeaways should be.
```

##### [24-02-02] [paper345]
- On the Calibration of Neural Networks for Histological Slide-Level Classification
 [[pdf]](https://arxiv.org/abs/2312.09719) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20the%20Calibration%20of%20Neural%20Networks%20for%20Histological%20Slide-Level%20Classification.pdf)
- `arxiv, 2023-12`
-  [ML for Medicine/Healthcare], [Computational Pathology], [Uncertainty Estimation]
```
Short paper of just 5 pages, somewhat interesting. Not an overly extensive evaluation, quite basic setup. All three models seem fairly overconfident.
```

##### [24-02-01] [paper344]
- Social Network Analysis of Cell Networks Improves Deep Learning for Prediction of Molecular Pathways and Key Mutations in Colorectal Cancer
 [[pdf]](https://www.sciencedirect.com/science/article/pii/S1361841523003316) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Social%20Network%20Analysis%20of%20Cell%20Networks%20Improves%20Deep%20Learning%20for%20Prediction%20of%20Molecular%20Pathways%20and%20Key%20Mutations%20in%20Colorectal%20Cancer.pdf)
- `Medical Image Analysis, 2024`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
Well written and quite interesting paper. The proposed method involves quite a few steps, but I think everything is explained well overall in Section 3. Figure 2 gives a good overview. The main idea, to incorporate explicit cell network-based features with the standard image features, makes some sense I think, and it seems to give at least a small boost in performance. But it's difficult for me to judge how impactful this actually might be.
```

##### [24-01-04] [paper343]
- End-To-End Prognostication in Colorectal Cancer by Deep Learning: A Retrospective, Multicentre Study
 [[pdf]](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(23)00208-X/fulltext#%20) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/End-To-End%20Prognostication%20in%20Colorectal%20Cancer%20by%20Deep%20Learning%20A%20Retrospective%2C%20Multicentre%20Study.pdf)
- `The Lancet Digital Health, 2024`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
Well-written and quite interesting paper. Everything seems solid, good background material for me. The used model, with attention-based aggregation of tile-level feature vectors into a single slide-level feature vector, makes sense. 
```

##### [24-01-04] [paper342]
- Clinical Evaluation of Deep Learning-Based Risk Profiling in Breast Cancer Histopathology and Comparison to an Established Multigene Assay
 [[pdf]](https://www.medrxiv.org/content/10.1101/2023.08.31.23294882v2) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Clinical%20Evaluation%20of%20Deep%20Learning-Based%20Risk%20Profiling%20in%20Breast%20Cancer%20Histopathology%20and%20Comparison%20to%20an%20Established%20Multigene%20Assay.pdf)
- `medrxiv, 2023-09`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
Interesting paper overall, definitely good background material for me.
```

#### Papers Read in 2023:

##### [23-12-30] [paper341]
- Colorectal Cancer Risk Stratification on Histological Slides Based on Survival Curves Predicted by Deep Learning
 [[pdf]](https://www.nature.com/articles/s41698-023-00451-3) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Colorectal%20Cancer%20Risk%20Stratification%20on%20Histological%20Slides%20Based%20on%20Survival%20Curves%20Predicted%20by%20Deep%20Learning.pdf)
- `npj Precision Oncology, 2023`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
Well-written and interesting paper. The used model shown in Figure 1 makes sense. The comparison of various feature extractors (pretrained in different ways) is interesting.
```

##### [23-12-28] [paper340]
- Deep Learning-Based Risk Stratification of Preoperative Breast Biopsies Using Digital Whole Slide Images
 [[pdf]](https://www.medrxiv.org/content/10.1101/2023.08.22.23294409v1) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Learning-Based%20Risk%20Stratification%20of%20Preoperative%20Breast%20Biopsies%20Using%20Digital%20Whole%20Slide%20Images.pdf)
- `medrxiv, 2023-08`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
Interesting paper. Very good background material for me. Everything is quite clearly described. The performance comparisons between biopsy and resected tumour etc. are interesting. 
```

##### [23-12-23] [paper339]
- Quilt-1M: One Million Image-Text Pairs for Histopathology
 [[pdf]](https://openreview.net/forum?id=OL2JQoO0kq) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Quilt-1M%20One%20Million%20Image-Text%20Pairs%20for%20Histopathology.pdf)
- `NeurIPS 2023 Track on Datasets and Benchmark`
-  [Vision-Language Models], [ML for Medicine/Healthcare], [Computational Pathology]
```
Well-written and fairly interesting paper. It mostly contains details about the dataset creation process. While this definitely is impressive and pretty cool, I personally didn't find it overly interesting. The zero-shot / linear probing results seem quite promising, but it's difficult for me to judge how impressive they actually are.
```

##### [23-12-21] [paper338]
- Development and Prognostic Validation of a Three-Level NHG-Like Deep Learning-Based Model for Histological Grading of Breast Cancer
 [[pdf]](https://www.medrxiv.org/content/10.1101/2023.02.15.23285956v1) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Development%20and%20Prognostic%20Validation%20of%20a%20Three-Level%20NHG-Like%20Deep%20Learning-Based%20Model%20for%20Histological%20Grading%20of%20Breast%20Cancer.pdf)
- `medrxiv, 2023-02`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
Interesting paper. Very good background material for me. I still don't fully understand everything in the results (survival analysis etc.), but it feels like I'm slowly learning with each paper at least.
```

##### [23-12-20] [paper337]
- Improved Breast Cancer Histological Grading Using Deep Learning
 [[pdf]](https://www.annalsofoncology.org/article/S0923-7534(21)04486-0/fulltext) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Improved%20Breast%20Cancer%20Histological%20Grading%20Using%20Deep%20Learning.pdf)
- `Annals of Oncology, 2022`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
Well-written and interesting paper. The application and overall approach makes sense, it's pretty cool. I think I'm starting to understand more of the medical content (the way these models are evaluated, how the results are presented etc.), it's also described well in this paper.
```

##### [23-12-20] [paper336]
- Deep Learning in Cancer Pathology: A New Generation of Clinical Biomarkers
 [[pdf]](https://www.nature.com/articles/s41416-020-01122-x) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Learning%20in%20Cancer%20Pathology%20A%20New%20Generation%20of%20Clinical%20Biomarkers.pdf)
- `British Journal of Cancer, 2021`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
A very well-written and interesting review paper, I enjoyed reading it. I also found it really useful, it helped me understand what biomarkers actually are and what they are used for much better than before. I can recommend reading this paper, I think it's a good way to gain quite a lot of medical background knowledge.
```

##### [23-12-19] [paper335]
- Artificial Intelligence as the Next Step Towards Precision Pathology
 [[pdf]](https://onlinelibrary.wiley.com/doi/10.1111/joim.13030) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Artificial%20Intelligence%20as%20the%20Next%20Step%20Towards%20Precision%20Pathology.pdf)
- `Journal of Internal Medicine, 2020`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
Review paper. Quite interesting. I think it was quite useful for me to read, good to just get a sense of what things have been tried before.
```

##### [23-12-19] [paper334]
- Artificial Intelligence for Diagnosis and Grading of Prostate Cancer in Biopsies: A Population-Based, Diagnostic Study
 [[pdf]](https://www.sciencedirect.com/science/article/pii/S1470204519307387?via%3Dihub) [_unfortunately not open access, thus no annotated pdf_]
- `The Lancet Oncology, 2020`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
Well-written and fairly interesting paper. Quite similar to other papers, but another piece of good background knowledge for me.
```

##### [23-12-18] [paper333]
- Clinical-Grade Computational Pathology Using Weakly Supervised Deep Learning on Whole Slide Images
 [[pdf]](https://www.nature.com/articles/s41591-019-0508-1) [_unfortunately not open access, thus no annotated pdf_]
- `Nature Medicine, 2019`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
Well-written and quite interesting paper. Everything is clearly described. I think this is a good introductory paper to read if one wants to get into computational pathology.
```

##### [23-12-18] [paper332]
- Classification and Mutation Prediction From Non–Small Cell Lung Cancer Histopathology Images Using Deep Learning
 [[pdf]](https://www.nature.com/articles/s41591-018-0177-5) [_unfortunately not open access, thus no annotated pdf_]
- `Nature Medicine, 2018`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
Quite well-written and fairly interesting paper. I quite liked the introduction, it gives some good background. The ML-related things are simple and described well. The second half of the Results section contains a lot of medical details which I'm unable to properly understand/appreciate.
```

##### [23-12-15] [paper331]
- Predicting Molecular Phenotypes from Histopathology Images: A Transcriptome-Wide Expression–Morphology Analysis in Breast Cancer
 [[pdf]](https://aacrjournals.org/cancerres/article/81/19/5115/670326/Predicting-Molecular-Phenotypes-from) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Predicting%20Molecular%20Phenotypes%20from%20Histopathology%20Images%20A%20Transcriptome-Wide%20Expression%E2%80%93Morphology%20Analysis%20in%20Breast%20Cancer.pdf)
- `Cancer Research, 2021`
-  [ML for Medicine/Healthcare], [Computational Pathology]
```
A ton of medical background that I'm not even close to properly understanding, but still quite interesting to read. Odd but kind of cool that they trained a CNN for each of the ~17 000 (!) different genes. The actual ML model is quite straightforward and makes sense.
```

##### [23-12-14] [paper330]
- Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning
 [[pdf]](https://arxiv.org/abs/2206.02647) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Scaling%20Vision%20Transformers%20to%20Gigapixel%20Images%20via%20Hierarchical%20Self-Supervised%20Learning.pdf)
- `CVPR 2022`
- [Representation Learning], [Transformers], [ML for Medicine/Healthcare], [Computational Pathology]
```
Quite interesting and well-written paper. The proposed method makes intuitive sense, I think. However, I definitely don't have the necessary background knowledge to fully appreciate this paper, or judge how novel/impressive/interesting the proposed method and the experimental results actually are. I should have started reading some earlier and more basic computational pathology papers.
```

##### [23-11-23] [paper329]
- Algorithmic Fairness In Artificial Intelligence For Medicine And Healthcare
 [[pdf]](https://www.nature.com/articles/s41551-023-01056-8) [_unfortunately not open access, thus no annotated pdf_]
- `Nature Biomedical Engineering, 2023`
- [ML for Medicine/Healthcare], [Social Consequences of ML]
```
Quite interesting paper. Fairly long and dense, so I got a bit tired towards the end. I quite enjoyed reading page 1 - 10, and then I found the "Paths forward" section less interesting. They talk a lot about dataset shifts, which I liked. They gave some good concrete examples. I also found a bunch of seemingly interesting papers in the (long) reference list.
```

##### [23-11-14] [paper328]
- Large Language Models Propagate Race-based Medicine
 [[pdf]](https://www.nature.com/articles/s41746-023-00939-z) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Large%20Language%20Models%20Propagate%20Race-based%20Medicine.pdf)
- `npj Digital Medicine, 2023`
- [ML for Medicine/Healthcare], [Social Consequences of ML]
```
Very short paper (just 2 pages long), but still quite interesting. It raises some interesting questions and thoughts. I also read this news article about the paper: https://apnews.com/article/ai-chatbots-racist-medicine-chatgpt-bard-6f2a330086acd0a1f8955ac995bdde4d, which also was quite interesting.
```

##### [23-11-10] [paper327]
- Sociotechnical Safety Evaluation of Generative AI Systems
 [[pdf]](https://arxiv.org/abs/2310.11986) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Sociotechnical%20Safety%20Evaluation%20of%20Generative%20AI%20Systems.pdf)
- `2023-10`
- [Social Consequences of ML]
```
Interesting paper, I quite enjoyed reading it. Longer than what I usually read (~25-30 pages). I didn't intend to read the entire paper but ended up doing so anyway, because I found it interesting overall. Section 4 was less interesting. If just reading parts of it, I would recommend Section 2, 5, A.1 and A.3. Section 5.4.2 - 5.4.4 is interesting, on who actually should be evaluating ML systems, that we need ~independent~ evaluators etc. I like this quote from Section 5.4.2 "Evaluations must be conducted during the process of AI development, to bake in ethical and social considerations from the inception of an AI system rather than imperfectly patching them on as an afterthought".
```

##### [23-11-09] [paper326]
- Ambient Diffusion: Learning Clean Distributions from Corrupted Data
 [[pdf]](https://openreview.net/forum?id=wBJBLy9kBY) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Ambient%20Diffusion%3A%20Learning%20Clean%20Distributions%20from%20Corrupted%20Data.pdf)
- `NeurIPS 2023`
- [Diffusion Models]
```
Well-written and quite interesting paper. The memorization/privacy discussion is quite interesting. I would have liked to see the "Diffusion No Further Corruption" baseline from Fig 3 in the other experiments as well.
```

##### [23-11-09] [paper325]
- Consistent Diffusion Models: Mitigating Sampling Drift by Learning to be Consistent
 [[pdf]](https://openreview.net/forum?id=GfZGdJHj27) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Consistent%20Diffusion%20Models%3A%20Mitigating%20Sampling%20Drift%20by%20Learning%20to%20be%20Consistent.pdf)
- `NeurIPS 2023`
- [Diffusion Models]
```
Well-written and fairly interesting paper. The overall idea makes sense, but I'm not entirely convinced that the performance gains in practice are worth the 50% increase in training cost.
```

##### [23-11-08] [paper324]
- Considerations For Addressing Bias In Artificial Intelligence For Health Equity
 [[pdf]](https://www.nature.com/articles/s41746-023-00913-9) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Considerations%20For%20Addressing%20Bias%20In%20Artificial%20Intelligence%20For%20Health%20Equity.pdf)
- `npj Digital Medicine, 2023`
- [ML for Medicine/Healthcare], [Social Consequences of ML]
```
Quite well-written and fairly interesting paper. The overall approach (to consider health equity effects across the entire product lifecycle of ML systems) of course makes sense, but the paper is still quite vague. I would have liked to see more actionable, concrete takeaways.
```

##### [23-11-02] [paper323]
- A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle
 [[pdf]](https://arxiv.org/abs/1901.10002) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Framework%20for%20Understanding%20Sources%20of%20Harm%20throughout%20the%20Machine%20Learning%20Life%20Cycle.pdf)
- `EAAMO 2021`
- [Social Consequences of ML]
```
Interesting paper, I enjoyed reading it. The description of different potential sources of harm (across the entire process from data collection to real-world deployment) in Section 3 is concise and clear. I have probably read about all these things at some point over the years, but to have it all summarized and structured in this way is neat. I definitely think this could be useful. If nothing else, is is a good introduction to these issues.
```

##### [23-03-14] [paper322]
- Supervised Contrastive Regression
 [[pdf]](https://arxiv.org/abs/2210.01189) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Supervised%20Contrastive%20Regression.pdf)
- `2022-10`
- [Representation Learning]
```
Well-written and interesting paper. Figure 1 is really interesting. Their proposed method makes intuitive sense, and it seems to consistently improve the regression accuracy. 
```

##### [23-09-13] [paper321]
- Adding Conditional Control to Text-to-Image Diffusion Models
 [[pdf]](https://arxiv.org/abs/2302.05543) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Adding%20Conditional%20Control%20to%20Text-to-Image%20Diffusion%20Models.pdf)
- `2023-02`
- [Diffusion Models], [Vision-Language Models]
```
Well-written and quite interesting paper. The "sudden convergence phenomenon" in Figure 4 seems odd. The results in Figure 11 are actually very cool.
```

##### [23-08-23] [paper320]
- Random Word Data Augmentation with CLIP for Zero-Shot Anomaly Detection
 [[pdf]](https://arxiv.org/abs/2308.11119) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Random%20Word%20Data%20Augmentation%20with%20CLIP%20for%20Zero-Shot%20Anomaly%20Detection.pdf)
- `BMVC 2023`
- [Vision-Language Models]
```
Interesting and well-written paper, I enjoyed reading it (even though I really don't like the BMVC template). The proposed method in Section 3 is clever/neat/interesting.
```

##### [23-08-23] [paper319]
- TextIR: A Simple Framework for Text-based Editable Image Restoration
 [[pdf]](https://arxiv.org/abs/2302.14736) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/TextIR%3A%20A%20Simple%20Framework%20for%20Text-based%20Editable%20Image%20Restoration.pdf)
- `2023-02`
- [Vision-Language Models], [Image Restoration]
```
Quite interesting and well-written paper. The idea in Section 3.1 is interesting/neat. The results in Figure 6 - 8 are quite interesting.
```

##### [23-08-22] [paper318]
- Learning to Prompt for Open-Vocabulary Object Detection with Vision-Language Model
 [[pdf]](https://arxiv.org/abs/2203.14940) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20to%20Prompt%20for%20Open-Vocabulary%20Object%20Detection%20with%20Vision-Language%20Model.pdf)
- `CVPR 2022`
- [Vision-Language Models], [Object Detection]
```
Quite interesting and quite well-written paper. They basically improve the "Open-vocabulary Object Detection via Vision and Language Knowledge Distillation" paper by using learnable prompts. Section 4.1 gives a pretty good background.
```

##### [23-08-21] [paper317]
- Open-vocabulary Object Detection via Vision and Language Knowledge Distillation
 [[pdf]](https://arxiv.org/abs/2104.13921) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Open-vocabulary%20Object%20Detection%20via%20Vision%20and%20Language%20Knowledge%20Distillation.pdf)
- `ICLR 2022`
- [Vision-Language Models], [Object Detection]
```
Quite well-written and fairly interesting paper. The simple (but slow) baseline in Section 3.2 makes sense, but then I struggled to properly understand the proposed method in Section 3.3. I might lack some required background knowledge.
```

##### [23-08-21] [paper316]
- All-In-One Image Restoration for Unknown Corruption
 [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/All-In-One%20Image%20Restoration%20for%20Unknown%20Corruption.pdf)
- `CVPR 2022`
- [Image Restoration]
```
Quite well-written and fairly interesting paper. Did not take very long to read. The general idea of the "Contrastive-Based Degradation Encoder" makes sense.
```

##### [23-08-18] [paper315]
- ProRes: Exploring Degradation-aware Visual Prompt for Universal Image Restoration
 [[pdf]](https://arxiv.org/abs/2306.13653) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/ProRes%3A%20Exploring%20Degradation-aware%20Visual%20Prompt%20for%20Universal%20Image%20Restoration.pdf)
- `2023-06`
- [Image Restoration]
```
Quite interesting and well-written paper. If I understand everything correctly, they need a user to select the correct task-specific visual prompt at test-time. I.e., the user needs to specify if a given input image is an image for denoising, low-light enhancement, deraining or deblurring. This seems like a quite significant limitation to me. Would like to have a model that, after being trained on restoration task 1, 2, ..., N, can restore a given image without any user input, for images from all N tasks.
```

##### [23-08-17] [paper314]
- PromptIR: Prompting for All-in-One Blind Image Restoration
 [[pdf]](https://arxiv.org/abs/2306.13090) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/PromptIR%3A%20Prompting%20for%20All-in-One%20Blind%20Image%20Restoration.pdf)
- `NeurIPS 2023`
- [Image Restoration]
```
Well-written and quite interesting paper. They describe their overall method well in Section 3. I was not familiar with prompt-learning, but I think they did a good jobb explaining it.
```

##### [23-08-17] [paper313]
- InstructPix2Pix: Learning to Follow Image Editing Instructions
 [[pdf]](https://arxiv.org/abs/2211.09800) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/InstructPix2Pix%3A%20Learning%20to%20Follow%20Image%20Editing%20Instructions.pdf)
- `CVPR 2023`
- [Diffusion Models], [Vision-Language Models]
```
Well-written and quite interesting paper. The method is conceptually simple and makes intuitive sense. Definitely impressive visual results (I'm especially impressed by Figure 7 and the right part of Figure 17). Figure 14 is important, interesting to see such a clear example of gender bias in the data being reflected in the model.
```

##### [23-09-23] [paper312]
- Machine learning: Trends, Perspectives, and Prospects
 [[pdf]](https://www.cs.cmu.edu/~tom/pubs/Science-ML-2015.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Machine%20learning%3A%20Trends%2C%20Perspectives%2C%20and%20Prospects.pdf)
- `Science, 2015`
- [Miscellaneous]
```
Well-written paper. I read it for my thesis writing, wanted to see some basic definitions of machine learning, I quite liked it.
```

##### [23-09-19] [paper311]
- Blinded, Randomized Trial of Sonographer versus AI Cardiac Function Assessment
 [[pdf]](https://www.nature.com/articles/s41586-023-05947-3) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Blinded%2C%20Randomized%20Trial%20of%20Sonographer%20versus%20AI%20Cardiac%20Function%20Assessment.pdf)
- `Nature, 2023`
- [ML for Medicine/Healthcare]
```
Well-written and interesting paper. It seems like I quite enjoy reading these types of papers.
```

##### [23-09-19] [paper310]
- Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style
 [[pdf]](https://arxiv.org/abs/2106.04619) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Self-Supervised%20Learning%20with%20Data%20Augmentations%20Provably%20Isolates%20Content%20from%20Style.pdf)
- `NeurIPS 2021`
- [Representation Learning]
```
Quite well-written and somewhat interesting paper. I struggled to properly understand quite large parts of it, probably because I lack some background knowledge. It's not clear to me what the main takeaway / main practical implication of this paper is.
```

##### [23-09-14] [paper309]
- Artificial Intelligence-Supported Screen Reading versus Standard Double Reading in the Mammography Screening with Artificial Intelligence Trial (MASAI): A Clinical Safety Analysis of a Randomised, Controlled, Non-inferiority, Single-Blinded, Screening Accuracy Study
 [[pdf]](https://www.thelancet.com/journals/lanonc/article/PIIS1470-2045(23)00298-X/fulltext) [_unfortunately not open access, thus no annotated pdf_]
- `The Lancet Oncology, 2023`
- [ML for Medicine/Healthcare]
```
Very very similar to the "Artificial Intelligence for Breast Cancer Detection in Screening Mammography in Sweden: A Prospective, Population-Based, Paired-Reader, Non-inferiority Study" paper, also well written and very interesting (and, it probably has the longest title of any paper I have ever read).
```

##### [23-09-14] [paper308]
- Efficient Formal Safety Analysis of Neural Networks
 [[pdf]](https://arxiv.org/abs/1809.08098) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Efficient%20Formal%20Safety%20Analysis%20of%20Neural%20Networks.pdf)
- `NeurIPS 2018`
- [Theoretical Properties of Deep Learning]
```
Well-written and quite interesting paper. I didn't entirely follow all details, and also find it difficult to know exactly how to interpret the results. I lack some background knowledge. I'm still not quite sure what a method like this actually could be used for in practice, how useful it actually would be for someone like me. Reading the paper made me think quite a lot though, which is a good thing.
```

##### [23-09-14] [paper307]
- Artificial Intelligence for Breast Cancer Detection in Screening Mammography in Sweden: A Prospective, Population-Based, Paired-Reader, Non-inferiority Study
 [[pdf]](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(23)00153-X/fulltext) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Artificial%20intelligence%20for%20breast%20cancer%20detection%20in%20screening%20mammography%20in%20Sweden%3A%20a%20prospective%2C%20population-based%2C%20paired-reader%2C%20non-inferiority%20study.pdf)
- `The Lancet Digital Health, 2023`
- [ML for Medicine/Healthcare]
```
Well-written and very interesting paper. A bit different compared to the ML papers I usually read of course, but different in a good way. Definitely an impressive study with ~50 000 participants, and an ML system integrated into the standard mammography screening workflow at a hospital. The entire Discussion section is interesting.
```

##### [23-08-31] [paper306]
- A Law of Data Separation in Deep Learning
 [[pdf]](https://arxiv.org/abs/2210.17020) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Law%20of%20Data%20Separation%20in%20Deep%20Learning.pdf)
- `PNAS, 2023`
- [Theoretical Properties of Deep Learning]
```
Quite well-written and fairly interesting paper. Interesting up until the end of Section 1, but then I got a bit confused and less impressed/convinced. Difficult for me to judge how general these findings actually are, or how useful they would be in practice. 
```

##### [23-08-24] [paper305]
- Loss Landscapes are All You Need: Neural Network Generalization Can Be Explained Without the Implicit Bias of Gradient Descent
 [[pdf]](https://openreview.net/forum?id=QC10RmRbZy9) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Loss%20Landscapes%20are%20All%20You%20Need%3A%20Neural%20Network%20Generalization%20Can%20Be%20Explained%20Without%20the%20Implicit%20Bias%20of%20Gradient%20Descent.pdf)
- `ICLR 2023`
- [Theoretical Properties of Deep Learning]
```
 Interesting and quite well-written paper. I really liked the paper up until and including Section 4.1, but then I got less impressed. I found the experiments a bit confusing overall, and not entirely convincing. The paper structure is also a bit odd after Section 4 (why is Section 5 a separate section? Section 6 seems sort of out-of-place).
```

##### [23-08-15] [paper304]
- Aligned Diffusion Schrödinger Bridges
 [[pdf]](https://arxiv.org/abs/2302.11419) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Aligned%20Diffusion%20Schr%C3%B6dinger%20Bridges.pdf)
- `UAI 2023`
- [Diffusion Models]
```
Interesting and very well-written paper. Cool applications from biology, although I definitely don't understand them fully. Don't quite understand how they get to the loss function in eq. (7) (match the terms in (3) and (6), yes, but why should this then be minimized?). 
```

##### [23-06-14] [paper303]
- Transport with Support: Data-Conditional Diffusion Bridges
 [[pdf]](https://arxiv.org/abs/2301.13636) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Transport%20with%20Support%3A%20Data-Conditional%20Diffusion%20Bridges.pdf)
- `2023-01`
- [Diffusion Models]
```
Well written and quite interesting paper. Not exactly what I had expected, my background knowledge was probably not sufficient to fully understand and appreciate this paper. I still enjoyed reading it though. Neat figures and examples.
```

##### [23-06-09] [paper302]
- An Overlooked Key to Excellence in Research: A Longitudinal Cohort Study on the Association Between the Psycho-Social Work Environment and Research Performance
 [[pdf]](https://www.tandfonline.com/doi/full/10.1080/03075079.2020.1744127) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/An%20Overlooked%20Key%20to%20Excellence%20in%20Research%3A%20A%20Longitudinal%20Cohort%20Study%20on%20the%20Association%20Between%20the%20Psycho-Social%20Work%20Environment%20and%20Research%20Performance.pdf)
- `Studies in Higher Education, 2021`
- [Miscellaneous]
```
Quite well written and interesting paper. I wanted to read something completely different compared to what I usually read. This paper was mentioned in a lecture I attended and seemed quite interesting. I don't regret reading it. I have never heard of "SEM analysis", thus it's difficult for me to judge how significant the results are. I think one can quite safely conclude that a good psycho-social work environment positively impacts research performance/excellence, but it's probably difficult to say ~how~ big this impact actually is. And, how big this impact is compared to various other factors. Either way, I quite enjoyed reading the paper.
```

##### [23-06-03] [paper301]
- Building One-class Detector for Anything: Open-vocabulary Zero-shot OOD Detection Using Text-image Models
 [[pdf]](https://arxiv.org/abs/2305.17207) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Building%20One-class%20Detector%20for%20Anything%3A%20Open-vocabulary%20Zero-shot%20OOD%20Detection%20Using%20Text-image%20Models.pdf)
- `2023-05`
- [Out-of-Distribution Detection], [Vision-Language Models]
```
Well written and interesting paper. Section 2.1 provides a good background, and their proposed OOD scores in Section 2.2 make intuitive sense. The datasets and evaluation setup in Section 3 are described well. The experimental results definitely seem promising.
```

##### [23-06-02] [paper300]
- Benchmarking Common Uncertainty Estimation Methods with Histopathological Images under Domain Shift and Label Noise
 [[pdf]](https://arxiv.org/abs/2301.01054) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Benchmarking%20Common%20Uncertainty%20Estimation%20Methods%20with%20Histopathological%20Images%20under%20Domain%20Shift%20and%20Label%20Noise.pdf)
- `2023-01`
- [Uncertainty Estimation], [ML for Medicine/Healthcare]
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
- [ML for Medicine/Healthcare]
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
- [Miscellaneous]
```
Quite well-written and somewhat interesting paper. I really struggled to understand everything properly, I definitely don't have the required background knowledge. I don't quite understand what data they train the network on, do they train separate networks for each example? Not clear to me how generally applicable this method actually is.
```

##### [23-04-15] [paper290]
- A Roadmap to Fair and Trustworthy Prediction Model Validation in Healthcare
 [[pdf]](https://arxiv.org/abs/2304.03779) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Roadmap%20to%20Fair%20and%20Trustworthy%20Prediction%20Model%20Validation%20in%20Healthcare.pdf)
- `2023-04`
- [ML for Medicine/Healthcare]
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
- [Uncertainty Estimation], [ML for Medicine/Healthcare]
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
- `GECCO 2021`
- [Miscellaneous]
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
- `CVPR 2023`
- [Out-of-Distribution Detection]
```
Quite interesting, but not overly well-written paper. I don't like the "... is all you need" title, and they focus too much on selling how their method beats SOTA (Figure 1 does definitely not illustrate the performance difference in a fair way).
```

##### [23-03-10] [paper279]
- Out-of-Distribution Detection and Selective Generation for Conditional Language Models
 [[pdf]](https://openreview.net/forum?id=kJUS5nD0vPB) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Out-of-Distribution%20Detection%20and%20Selective%20Generation%20for%20Conditional%20Language%20Models.pdf)
- `ICLR 2023`
- [Out-of-Distribution Detection], [Selective Prediction], [NLP]
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
- `CVPR Workshops 2023`
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
- `ICML 2023`
- [Theoretical Properties of Deep Learning]
```
Interesting paper. I would have needed a bit more time to read it though, felt like I didn't quite have enough time to properly understand everything and evaluate the significance of the findings. Might have to go back to this paper again.
```

##### [23-02-09] [paper259]
- The Forward-Forward Algorithm: Some Preliminary Investigations
 [[pdf]](https://arxiv.org/abs/2212.13345) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/The%20Forward-Forward%20Algorithm:%20Some%20Preliminary%20Investigations.pdf)
- `2022-12`
- [Miscellaneous]
```
Somewhat interesting, but quite odd paper. I was quite confused by multiple parts of it. This is probably partly because of my background, but I do also think that the paper could be more clearly structured.
```

##### [23-02-01] [paper258]
- Everything is Connected: Graph Neural Networks
 [[pdf]](https://arxiv.org/abs/2301.08210) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Everything%20is%20Connected:%20Graph%20Neural%20Networks.pdf)
- `Current Opinion in Structural Biology, 2023`
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
- [Social Consequences of ML]
```
Well-written and quite interesting paper. Describes the distributive justice principles of John Rawls' book "A theory of justice" and explores/discusses what these might imply for how "AI systems" should be regulated/deployed/etc. Doesn't really provide any overly concrete takeaways, at least not for me, but still a quite enjoyable read.
```

##### [22-12-08] [paper252]
- Talking About Large Language Models
 [[pdf]](https://arxiv.org/abs/2212.03551) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Talking%20About%20Large%20Language%20Models.pdf)
- `2022-12`
- [Social Consequences of ML], [NLP]
```
Well-written and interesting paper. Sections 1-6 and Section 11 are very interesting. A breath of fresh air to read this in the midst of the ChatGPT hype. It contains a lot of good quotes, for example:"To ensure that we can make informed decisions about the trustworthiness and safety of the AI systems we deploy, it is advisable to keep to the fore the way those systems actually work, and thereby to avoid imputing to them capacities they lack, while making the best use of the remarkable capabilities they genuinely possess".
```

##### [22-12-06] [paper251]
- Artificial Intelligence, Humanistic Ethics
 [[pdf]](https://direct.mit.edu/daed/article-pdf/151/2/232/2009174/daed_a_01912.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Artificial%20Intelligence%2C%20Humanistic%20Ethics.pdf)
- `Daedalus, 2022`
- [Social Consequences of ML]
```
Well-written and interesting paper. Provides some interesting comments/critique on utilitarianism and how engineers/scientists like myself might be inclined to find that approach attractive: "The optimizing mindset prevalent among computer scientists and economists, among other powerful actors, has led to an approach focused on maximizing the fulfilment of human preferences..... But this preference-based utilitarianism is open to serious objections. This essay sketches an alternative, “humanistic” ethics for AI that is sensitive to aspects of human engagement with the ethical often missed by the dominant approach." - - - - "So ethics is reduced to an exercise in prediction and optimization: which act or policy is likely to lead to the optimal fulfilment of human preferences?" - - - - "This incommensurability calls into question the availability of some optimizing function that determines the single option that is, all things considered, most beneficial or morally right, the quest for which has animated a lot of utilitarian thinking in ethics."
```

##### [22-12-06] [paper250]
- Physics-Informed Neural Networks for Cardiac Activation Mapping
 [[pdf]](https://www.frontiersin.org/articles/10.3389/fphy.2020.00042/full) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Physics-Informed%20Neural%20Networks%20for%20Cardiac%20Activation%20Mapping.pdf)
- `Frontiers in Physics, 2020`
- [ML for Medicine/Healthcare]
```
Quite well-written and somewhat interesting paper.
```

##### [22-12-05] [paper249]
- AI Ethics and its Pitfalls: Not Living up to its own Standards?
 [[pdf]](https://link.springer.com/article/10.1007/s43681-022-00173-5) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/AI%20ethics%20and%20its%20pitfalls:%20not%20living%20up%20to%20its%20own%20standards%3F.pdf)
- `AI and Ethics, 2022`
- [Social Consequences of ML]
```
Well-written and somewhat interesting paper. Good reminder that also the practice of ML ethics could have unintended negative consequences. Section 2.6 is quite interesting.
```

##### [22-12-02] [paper248]
- Blind Spots in AI Ethics
 [[pdf]](https://link.springer.com/article/10.1007/s43681-021-00122-8) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Blind%20spots%20in%20AI%20ethics.pdf)
- `AI and Ethics, 2022`
- [Social Consequences of ML]
```
Well-written and very interesting paper. I enjoyed reading it, and it made me think - which is a good thing! Contains quite a few quotes which I really liked, for example: "However, it is wrong to assume that the goal is ethical AI. Rather, the primary aim from which detailed norms can be derived should be a peaceful, sustainable, and just society. Hence, AI ethics must dare to ask the question where in an ethical society one should use AI and its inherent principle of predictive modeling and classification at all".
```

##### [22-12-01] [paper247]
- The Ethics of AI Ethics: An Evaluation of Guidelines
 [[pdf]](https://link.springer.com/article/10.1007/s11023-020-09517-8) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/The%20Ethics%20of%20AI%20Ethics:%20An%20Evaluation%20of%20Guidelines.pdf)
- `Minds and Machines, 2020`
- [Social Consequences of ML]
```
Well-written and interesting paper. I liked that it discussed some actual ethical theories in Section 4.2. Sections 3.2, 3.3. and 4.1 were also interesting.
```

##### [22-12-01] [paper246]
- The Uselessness of AI Ethics
 [[pdf]](https://link.springer.com/article/10.1007/s43681-022-00209-w) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/The%20uselessness%20of%20AI%20ethics.pdf)
- `AI and Ethics, 2022`
- [Social Consequences of ML]
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
- [Social Consequences of ML]
```
Well-written and quite interesting paper. Just 5 pages long, quick to read. Sort of like an opinion piece. I enjoyed reading it. Main takeaway: "Instead of trying to reinvent ethics, or adopt ethical guidelines in isolation, it is incumbent upon us to recognize the need for broadly ethical organizations. These will be the only entrants in a position to build truly ethical AI. You cannot simply have AI ethics. It requires real ethical due diligence at the organizational level—perhaps, in some cases, even industry-wide refection".
```

##### [22-11-25] [paper243]
- Expert Responsibility in AI Development
 [[pdf]](https://link.springer.com/article/10.1007/s00146-022-01498-9) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Expert%20responsibility%20in%20AI%20development.pdf)
- `AI & Society, 2022`
- [Social Consequences of ML]
```
Well-written and interesting paper, quite straightforward to follow and understand everything. Section 6 & 7 are interesting, with the discussion about unintended consequences of recommender algorithms (how they contribute to an impaired democratic debate).
```

##### [22-11-25] [paper242]
- The future of AI in our hands? To what extent are we as individuals morally responsible for guiding the development of AI in a desirable direction?
 [[pdf]](https://link.springer.com/article/10.1007/s43681-021-00125-5) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/The%20future%20of%20AI%20in%20our%20hands%3F%20To%20what%20extent%20are%20we%20as%20individuals%20morally%20responsible%20for%20guiding%20the%20development%20of%20AI%20in%20a%20desirable%20direction%3F.pdf)
- `AI and Ethics, 2022`
- [Social Consequences of ML]
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
- [Miscellaneous]
```
Well-written and quite interesting paper. The proposed method is explained well and makes intuitive sense overall, and seems to perform well in the intended setting.
```

##### [22-11-09] [paper239]
- Learning Deep Representations by Mutual Information Estimation and Maximization
 [[pdf]](https://openreview.net/forum?id=Bklr3j0cKX) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20deep%20representations%20by%20mutual%20information%20estimation%20and%20maximization.pdf)
- `ICLR 2019`
- [Representation Learning]
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
- [Miscellaneous]
```
An opinion peace, not really a technical paper. Just 3-4 pages long. Well-written and quite interesting paper though, I quite enjoyed reading it. What the authors write at the end "Fundamental biology should not choose between small-scale mechanistic understanding and large-scale prediction. It should embrace the complementary strengths of mechanistic modelling and machine learning approaches to provide, for example, the missing link between patient outcome prediction and the mechanistic understanding of disease progression" makes a lot of sense to, this is my main takeaway. I also find the statement "The training of a new generation of researchers versatile in all these fields will be vital in making this breakthrough" quite interesting, this is probably true for really making progress in medical machine learning applications as well?
```

##### [22-09-22] [paper233]
- Adversarial Examples Are Not Bugs, They Are Features
 [[pdf]](https://papers.nips.cc/paper/2019/hash/e2c420d928d4bf8ce0ff2ec19b371514-Abstract.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Adversarial%20Examples%20Are%20Not%20Bugs%2C%20They%20Are%20Features.pdf)
- `NeurIPS 2019`
- [Miscellaneous]
```
Well-written and interesting paper, I quite enjoyed reading it. I found this quite a lot more interesting than previous papers I have read on adversarial examples. 
```

##### [22-09-15] [paper232]
- Learning to Learn by Gradient Descent by Gradient Descent
 [[pdf]](https://proceedings.neurips.cc/paper/2016/file/fb87582825f9d28a8d42c5e5e5e8b23d-Paper.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20to%20learn%20by%20gradient%20descent%20by%20gradient%20descent.pdf)
- `NeurIPS 2016`
- [Miscellaneous]
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
- `IDA 2020`
- [Uncertainty Estimation]
```
Quite well-written and somewhat interesting paper. 
```

##### [22-06-23] [paper229]
- Linear Time Sinkhorn Divergences using Positive Features
 [[pdf]](https://proceedings.neurips.cc/paper/2020/hash/9bde76f262285bb1eaeb7b40c758b53e-Abstract.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Linear%20Time%20Sinkhorn%20Divergences%20using%20Positive%20Features.pdf)
- `NeurIPS 2020`
- [Miscellaneous]
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
- [Representation Learning]
```
Quite well-written and somewhat interesting paper. Definitely not my area of expertise (learning disentangled representations of e.g. images) and I didn't have a lot of time to read the paper, I struggled to understand big parts of the paper.
```

##### [22-06-02] [paper226]
- Shaking the Foundations: Delusions in Sequence Models for Interaction and Control
 [[pdf]](https://arxiv.org/abs/2110.10819) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Shaking%20the%20foundations:%20delusions%20in%20sequence%20models%20for%20interaction%20and%20control.pdf)
- `2021-10`
- [Sequence Modeling]
```
Quite well-written and somewhat interesting paper. Definitely not my area of expertise (causality). I didn't understand everything properly, and it's very difficult for me to judge how interesting this paper actually is.
```

##### [22-05-23] [paper225]
- When are Bayesian Model Probabilities Overconfident?
 [[pdf]](https://arxiv.org/abs/2003.04026) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/When%20are%20Bayesian%20model%20probabilities%20overconfident%3F.pdf)
- `2020-03`
- [Miscellaneous]
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
- [Uncertainty Estimation], [ML for Medicine/Healthcare]
```
Somewhat interesting paper. Image and text classification. The general problem setting (that a model can either predict or defer to an expert) is interesting and the paper is well-written overall, but in the end I can't really state any specific takeaways. I didn't understand section 4 or 5 properly. I don't think I can judge the significance of their results/contributions. 
```

##### [22-04-06] [paper219]
- Uncalibrated Models Can Improve Human-AI Collaboration
 [[pdf]](https://arxiv.org/abs/2202.05983) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncalibrated%20Models%20Can%20Improve%20Human-AI%20Collaboration.pdf)
- `NeurIPS 2022`
- [ML for Medicine/Healthcare]
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
- [ML for Medicine/Healthcare], [Out-of-Distribution Detection], [Transformers]
```
Well-written and interesting paper. I was not familiar with the VQ-GAN/VAE model, so I was confused by Section 2.3 at first, but now I think that I understand most of it. Their VQ-GAN + transformer approach seems quite complex indeed, but also seems to perform well. However, they didn't really compare with any other OOD detection method. I find it somewhat difficult to tell how useful this actually could be in practice.
```

##### [22-03-31] [paper214]
- Delving into Deep Imbalanced Regression
 [[pdf]](https://arxiv.org/abs/2102.09554) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Delving%20into%20Deep%20Imbalanced%20Regression.pdf)
- `ICML 2021`
- [Miscellaneous]
```
Well-written and somewhat interesting paper. The "health condition score" estimation problem seems potentially interesting. They only consider problems with 1D regression targets. Their two proposed methods are clearly explained. I could probably encounter this imbalanced issue at some point, and then I'll keep this paper in mind.
```

##### [22-03-31] [paper213]
- Hidden in Plain Sight: Subgroup Shifts Escape OOD Detection
 [[pdf]](https://openreview.net/forum?id=aZgiUNye2Cz) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Hidden%20in%20Plain%20Sight:%20Subgroup%20Shifts%20Escape%20OOD%20Detection.pdf)
- `MIDL 2022`
- [ML for Medicine/Healthcare], [Out-of-Distribution Detection], [Distribution Shifts]
```
Quite well-written, but somewhat confusing paper. The experiment in Table 1 seems odd to me, why would we expect or even want digit-5 images to be classified as OOD when the training data actually includes a bunch of digit-5 images (the bottom row)? And for what they write in the final paragraph of Section 3 (that the accuracy is a bit lower for the hospital 3 subgroup), this wouldn't actually be a problem in practice if the model then also is more uncertain for these examples? I.e., studying model calibration across the different subgroups would be what's actually interesting? Or am I not understanding this whole subgroup shift properly? I feel quite confused.
```

##### [22-03-30] [paper212]
- Self-Distribution Distillation: Efficient Uncertainty Estimation
 [[pdf]](https://arxiv.org/abs/2203.08295) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Self-Distribution%20Distillation:%20Efficient%20Uncertainty%20Estimation.pdf)
- `UAI 2022`
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
- [Miscellaneous]
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
- `ICLR 2021`
- [Theoretical Properties of Deep Learning]
```
Well-written and quite interesting paper. I didn't take the time to try and really understand all the details, but a quite enjoyable read. The proposed framework seems to make some intuitive sense and lead to some fairly interesting observations/insights, but it's difficult for me to judge how significant it actually is.
```

##### [22-03-08] [paper204]
- Selective Regression Under Fairness Criteria
 [[pdf]](https://arxiv.org/abs/2110.15403) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Selective%20Regression%20Under%20Fairness%20Criteria.pdf)
- `ICML 2022`
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
- `npj Digital Medicine, 2021`
- [Uncertainty Estimation], [ML for Medicine/Healthcare]
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
- `ICML 2019`
- [Uncertainty Estimation], [Selective Prediction]
```
Well-written and quite interesting paper. The proposed method is quite interesting and makes some intuitive sense, but I would assume that the calibration technique in Section 5 has similar issues as temperature scaling (i.e., the calibration might still break under various data shifts)?
```

##### [22-03-04] [paper199]
- NUQ: Nonparametric Uncertainty Quantification for Deterministic Neural Networks
 [[pdf]](https://arxiv.org/abs/2202.03101) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/NUQ:%20Nonparametric%20Uncertainty%20Quantification%20for%20Deterministic%20Neural%20Networks.pdf)
- `NeurIPS 2022`
- [Uncertainty Estimation], [Out-of-Distribution Detection]
```
Interesting paper. I found it difficult to understand Section 2, I wouldn't really be able to implement their proposed NUQ method. Only image classification, but their experimental evaluation is still quite extensive. And, they obtain strong performance.
```

##### [22-03-03] [paper198]
- On the Practicality of Deterministic Epistemic Uncertainty
 [[pdf]](https://arxiv.org/abs/2107.00649) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20the%20Practicality%20of%20Deterministic%20Epistemic%20Uncertainty.pdf)
- `ICML 2022`
- [Uncertainty Estimation]
```
Interesting and well-written paper. Their evaluation with the corrupted datasets makes sense I think. The results are interesting, the fact that ensembling/MC-dropout consistently outperforms the other methods. Another reminder of how strong of a baseline ensembling is when it comes to uncertainty estimation? Also, I think that their proposed rAULC is more or less equivalent to AUSE (area under the sparsification error curve)?
```

##### [22-03-03] [paper197]
- Transformers Can Do Bayesian Inference
 [[pdf]](https://arxiv.org/abs/2112.10510) [[code]](https://github.com/automl/TransformersCanDoBayesianInference) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Transformers%20Can%20Do%20Bayesian%20Inference.pdf)
- *Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, Frank Hutter*
- `ICLR 2022`
- [Transformers]
```
Quite interesting and well-written paper. I did however find it difficult to properly understand everything, it feels like a lot of details are omitted (I wouldn't really know how to actually implement this in practice). It's difficult for me to judge how impressive the results are or how practically useful this approach actually might be, what limitations are there? Overall though, it does indeed seem quite interesting.
```

##### [22-03-02] [paper196]
- A Deep Bayesian Neural Network for Cardiac Arrhythmia Classification with Rejection from ECG Recordings
 [[pdf]](https://arxiv.org/abs/2203.00512) [[code]](https://github.com/hsd1503/ecg_uncertainty) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Deep%20Bayesian%20Neural%20Network%20for%20Cardiac%20Arrhythmia%20Classification%20with%20Rejection%20from%20ECG%20Recordings.pdf)
- *Wenrui Zhang, Xinxin Di, Guodong Wei, Shijia Geng, Zhaoji Fu, Shenda Hong*
- `2022-02`
- [Uncertainty Estimation], [ML for Medicine/Healthcare]
```
Somewhat interesting paper. They use a softmax model with MC-dropout to compute uncertainty estimates. The evaluation is not very extensive, they mostly just check that the classification accuracy improves as they reject more and more samples based on a uncertainty threshold.
```

##### [22-02-26] [paper195]
- Out of Distribution Data Detection Using Dropout Bayesian Neural Networks
 [[pdf]](https://arxiv.org/abs/2202.08985) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Out%20of%20Distribution%20Data%20Detection%20Using%20Dropout%20Bayesian%20Neural%20Networks.pdf)
- *Andre T. Nguyen, Fred Lu, Gary Lopez Munoz, Edward Raff, Charles Nicholas, James Holt*
- `AAAI 2022`
- [Out-of-Distribution Detection]
```
Quite interesting and well-written paper. It seemed quite niche at first, but I think their analysis could potentially be useful.
```

##### [22-02-26] [paper194]
- Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks
 [[pdf]](https://arxiv.org/abs/1706.02690) [[code]](https://github.com/facebookresearch/odin) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Enhancing%20The%20Reliability%20of%20Out-of-distribution%20Image%20Detection%20in%20Neural%20Networks.pdf)
- *Shiyu Liang, Yixuan Li, R. Srikant*
- `ICLR 2018`
- [Out-of-Distribution Detection]
```
Quite interesting and well-written paper. Two simple modifications of the "maximum softmax score" baseline, and the performance is consistently improved. The input perturbation method is quite interesting. Intuitively, it's not entirely clear to me why it actually works.
```

##### [22-02-25] [paper193]
- Confidence-based Out-of-Distribution Detection: A Comparative Study and Analysis
 [[pdf]](https://arxiv.org/abs/2107.02568) [[code]](https://github.com/christophbrgr/ood_detection_framework) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Confidence-based%20Out-of-Distribution%20Detection:%20A%20Comparative%20Study%20and%20Analysis.pdf)
- *Christoph Berger, Magdalini Paschali, Ben Glocker, Konstantinos Kamnitsas*
- `MICCAI Workshops 2021`
- [Out-of-Distribution Detection], [ML for Medicine/Healthcare]
```
Interesting and well-written paper. Interesting that Mahalanobis works very well on the CIFAR10 vs SVHN but not on the medical imaging dataset. I don't quite get how/why the ODIN method works, I'll probably have to read that paper.
```

##### [22-02-25] [paper192]
- Deep Learning Through the Lens of Example Difficulty
 [[pdf]](https://openreview.net/forum?id=WWRBHhH158K) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Learning%20Through%20the%20Lens%20of%20Example%20Difficulty.pdf)
- *Robert John Nicholas Baldock, Hartmut Maennel, Behnam Neyshabur*
- `NeurIPS 2021`
- [Theoretical Properties of Deep Learning]
```
Quite interesting and well-written paper. The definition of "prediction depth" in Section 2.1 makes sense, and it definitely seems reasonable that this could correlate with example difficulty / prediction confidence in some way. Section 3 and 4, and all the figures, contain a lot of info it seems, I'd probably need to read the paper again to properly understand/appreciate everything.
```

##### [22-02-24] [paper191]
- UncertaINR: Uncertainty Quantification of End-to-End Implicit Neural Representations for Computed Tomography
 [[pdf]](https://arxiv.org/abs/2202.10847) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/UncertaINR:%20Uncertainty%20Quantification%20of%20End-to-End%20Implicit%20Neural%20Representations%20for%20Computed%20Tomography.pdf)
- *Francisca Vasconcelos, Bobby He, Nalini Singh, Yee Whye Teh*
- `TMLR, 2023`
- [Implicit Neural Representations], [Uncertainty Estimation], [ML for Medicine/Healthcare]
```
Interesting and well-written paper. I wasn't very familiar with CT image reconstruction, but they do a good job explaining everything. Interesting that MC-dropout seems important for getting well-calibrated predictions.
```

##### [22-02-21] [paper190]
- Can You Trust Predictive Uncertainty Under Real Dataset Shifts in Digital Pathology?
 [[pdf]](https://backend.orbit.dtu.dk/ws/portalfiles/portal/218217360/MICCAI2020.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Can%20you%20trust%20predictive%20uncertainty%20under%20real%20dataset%20shifts%20in%20digital%20pathology%3F.pdf)
- *Jeppe Thagaard, Søren Hauberg, Bert van der Vegt, Thomas Ebstrup, Johan D. Hansen, Anders B. Dahl*
- `MICCAI 2020`
- [Uncertainty Estimation], [Out-of-Distribution Detection], [ML for Medicine/Healthcare]
```
Quite interesting and well-written paper. They compare MC-dropout, ensemlbing and mixup (and with a standard softmax classifer as the baseline). Nothing groundbreaking, but the studied application (classification of pathology slides for cancer) is very interesting. The FPR95 metrics for OOD detection in Table 4 are terrible for ensembling, but the classification accuracy (89.7) is also pretty much the same as for D_test_int in Tabe 3 (90.1)? So, it doesn't really matter that the model isn't capable of distinguishing this "OOD" data from in-distribution? 
```

##### [22-02-21] [paper189]
- Robust Uncertainty Estimates with Out-of-Distribution Pseudo-Inputs Training
 [[pdf]](https://arxiv.org/abs/2201.05890) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Robust%20uncertainty%20estimates%20with%20out-of-distribution%20pseudo-inputs%20training.pdf)
- *Pierre Segonne, Yevgen Zainchkovskyy, Søren Hauberg*
- `2022-01`
- [Uncertainty Estimation]
```
Somewhat interesting paper. I didn't quite understand everything, so it could be more interesting than I think. The fact that their pseudo-input generation process "relies on the availability of a differentiable density estimate of the data" seems like a big limitation? For regression, they only applied their method to very low-dimensional input data (1D toy regression and UCI benchmarks), but would this work for image-based tasks?
```

##### [22-02-19] [paper188]
- Contrastive Training for Improved Out-of-Distribution Detection
 [[pdf]](https://arxiv.org/abs/2007.05566) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Contrastive%20Training%20for%20Improved%20Out-of-Distribution%20Detection.pdf)
- *Jim Winkens, Rudy Bunel, Abhijit Guha Roy, Robert Stanforth, Vivek Natarajan, Joseph R. Ledsam, Patricia MacWilliams, Pushmeet Kohli, Alan Karthikesalingam, Simon Kohl, Taylan Cemgil, S. M. Ali Eslami, Olaf Ronneberger*
- `2020-07`
- [Out-of-Distribution Detection]
```
Quite interesting and very well-written paper. They take the method from the Mahalanobis paper ("A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks") (however, they fit Gaussians only to the features at the second-to-last network layer, and they don't use the input pre-processing either) and consistently improve OOD detection performance by incorporating contrastive training. Specifically, they first train the network using just the SimCLR loss for a large number of epochs, and then also add the standard classification loss. I didn't quite get why the label smoothing is necessary, but according to Table 2 it's responsible for a large portion of the performance gain.
```

##### [22-02-19] [paper187]
- A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks
 [[pdf]](https://arxiv.org/abs/1807.03888) [[code]](https://github.com/pokaxpoka/deep_Mahalanobis_detector) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Simple%20Unified%20Framework%20for%20Detecting%20Out-of-Distribution%20Samples%20and%20Adversarial%20Attacks.pdf)
- *Kimin Lee, Kibok Lee, Honglak Lee, Jinwoo Shin*
- `NeurIPS 2018`
- [Out-of-Distribution Detection]
```
Well-written and interesting paper. The proposed method is simple and really neat: fit class-conditional Gaussians in the feature space of a pre-trained classifier (basically just LDA on the feature vectors), and then use the Mahalanobis distance to these Gaussians as the confidence score for input x. They then also do this for the features at multiple levels of the network and combine these confidence scores into one. I don't quite get why the "input pre-processing" in Section 2.2 (adding noise to test samples) works, in Table 1 it significantly improves the performance.
```

##### [22-02-19] [paper186]
- Noise Contrastive Priors for Functional Uncertainty
 [[pdf]](https://arxiv.org/abs/1807.09289) [[code]](https://github.com/brain-research/ncp) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Noise%20Contrastive%20Priors%20for%20Functional%20Uncertainty.pdf)
- *Danijar Hafner, Dustin Tran, Timothy Lillicrap, Alex Irpan, James Davidson*
- `UAI 2019`
- [Uncertainty Estimation], [Out-of-Distribution Detection]
```
Quite interesting and well-written paper. Only experiments on a toy 1D regression problem, and flight delay prediction in which the input is 8D. The approach of just adding noise to the input x to get OOD samples would probably not work very well e.g. for image-based problems?
```

##### [22-02-18] [paper185]
- Does Your Dermatology Classifier Know What It Doesn't Know? Detecting the Long-Tail of Unseen Conditions
 [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S1361841521003194?via%3Dihub) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Does%20your%20dermatology%20classifier%20know%20what%20it%20doesn't%20know%3F%20Detecting%20the%20long-tail%20of%20unseen%20conditions.pdf)
- *Abhijit Guha Roy, Jie Ren, Shekoofeh Azizi, Aaron Loh, Vivek Natarajan, Basil Mustafa, Nick Pawlowski, Jan Freyberg, Yuan Liu, Zach Beaver, Nam Vo, Peggy Bui, Samantha Winter, Patricia MacWilliams, Greg S. Corrado, Umesh Telang, Yun Liu, Taylan Cemgil, Alan Karthikesalingam, Balaji Lakshminarayanan, Jim Winkens*
- `Medical Image Analysis, 2022`
- [Out-of-Distribution Detection], [ML for Medicine/Healthcare]
```
Well-written and interesting paper. Quite long, so it took a bit longer than usual to read it. Section 1 and 2 gives a great overview of OOD detection in general, and how it can be used specifically in this dermatology setting. I can definitely recommend reading Section 2 (Related work). They assume access to some outlier data during training, so their approach is similar to the "Outlier exposure" method (specifically in this dermatology setting, they say that this is a fair assumption). Their method is an improvement of the "reject bucket" (add an extra class which you assign to all outlier training data points), in their proposed method they also use fine-grained classification of the outlier skin conditions. Then they also use an ensemble of 5 models, and also a more diverse ensemble (in which they combine models trained with different representation learning techniques). This diverse ensemble obtains the best performance.
```

##### [22-02-16] [paper184]
- Being a Bit Frequentist Improves Bayesian Neural Networks
 [[pdf]](https://arxiv.org/abs/2106.10065) [[code]](https://github.com/wiseodd/bayesian_ood_training) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Being%20a%20Bit%20Frequentist%20Improves%20Bayesian%20Neural%20Networks.pdf)
- *Agustinus Kristiadi, Matthias Hein, Philipp Hennig*
- `AISTATS 2022`
- [Uncertainty Estimation], [Out-of-Distribution Detection]
```
Interesting and well-written paper. The proposed method makes intuitive sense, trying to incorporate the "OOD training" method (i.e., to use some kind of OOD data during training, similar to e.g. the "Deep Anomaly Detection with Outlier Exposure" paper) into the Bayesian deep learning approach. The experimental results do seem quite promising.
```

##### [22-02-15] [paper183]
- Mixtures of Laplace Approximations for Improved Post-Hoc Uncertainty in Deep Learning
 [[pdf]](https://arxiv.org/abs/2111.03577) [[code]](https://github.com/AlexImmer/Laplace) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Mixtures%20of%20Laplace%20Approximations%20for%20Improved%20Post-Hoc%20Uncertainty%20in%20Deep%20Learning.pdf)
- *Runa Eschenhagen, Erik Daxberger, Philipp Hennig, Agustinus Kristiadi*
- `NeurIPS Workshops 2021`
- [Uncertainty Estimation], [Out-of-Distribution Detection]
```
Well-written and interesting paper. Short paper of just 3 pages, but with an extensive appendix which I definitely recommend going through. The method, training an ensemble and then applying the Laplace approximation to each network, is very simple and intuitively makes a lot of sense. I didn't realize that this would have basically the same test-time speed as ensembling (since they utilize that probit approximation), that's very neat. It also seems to consistently outperform ensembling a bit across almost all tasks and metrics.
```

##### [22-02-15] [paper182]
- Pessimistic Bootstrapping for Uncertainty-Driven Offline Reinforcement Learning
 [[pdf]](https://openreview.net/forum?id=Y4cs1Z3HnqL) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Pessimistic%20Bootstrapping%20for%20Uncertainty-Driven%20Offline%20Reinforcement%20Learning.pdf)
- *Chenjia Bai, Lingxiao Wang, Zhuoran Yang, Zhi-Hong Deng, Animesh Garg, Peng Liu, Zhaoran Wang*
- `ICLR 2022`
- [Uncertainty Estimation], [Reinforcement Learning]
```
Well-written and somewhat interesting paper. I'm not overly familiar with RL, which makes it a bit difficult for me to properly evaluate the paper's contributions. They use standard ensembles for uncertainty estimation combined with an OOD sampling regularization. I thought that the OOD sampling could be interesting, but it seems very specific to RL. I'm sure this paper is quite interesting for people doing RL, but I don't think it's overly useful for me.
```

##### [22-02-15] [paper181]
- On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks
 [[pdf]](https://openreview.net/forum?id=aPOpXlnV1T) [[code]](https://sites.google.com/view/pitfalls-uncertainty?authuser=0) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20the%20Pitfalls%20of%20Heteroscedastic%20Uncertainty%20Estimation%20with%20Probabilistic%20Neural%20Networks.pdf)
- *Maximilian Seitzer, Arash Tavakoli, Dimitrije Antic, Georg Martius*
- `ICLR 2022`
- [Uncertainty Estimation]
```
Quite interesting and very well-written paper, I enjoyed reading it. Their analysis of fitting Gaussian regression models via the NLL is quite interesting, I didn't really expect to learn something new about this. I've seen Gaussian models outperform standard regression (L2 loss) w.r.t. accuracy in some applications/datasets, and it being the other way around in others. In the first case, I've then attributed the success of the Gaussian model to the "learned loss attenuation". The analysis in this paper could perhaps explain why you get this performance boost only in certain applications. Their beta-NLL loss could probably be quite useful, seems like a convenient tool to have.
```

##### [22-02-15] [paper180]
- Sample Efficient Deep Reinforcement Learning via Uncertainty Estimation
 [[pdf]](https://openreview.net/forum?id=vrW3tvDfOJQ) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Sample%20Efficient%20Deep%20Reinforcement%20Learning%20via%20Uncertainty%20Estimation.pdf)
- *Vincent Mai, Kaustubh Mani, Liam Paull*
- `ICLR 2022`
- [Uncertainty Estimation], [Reinforcement Learning]
```
Well-written and somewhat interesting paper. I'm not overly familiar with reinforcement learning, which makes it a bit difficult for me to properly evaluate the paper's contributions, but to me it seems like fairly straightforward method modifications? To use ensembles of Gaussian models (instead of ensembles of models trained using the L2 loss) makes sense. The BIV method I didn't quite get, it seems rather ad hoc? I also don't quite get exactly how it's used in equation (10), is the ensemble of Gaussian models trained _jointly_ using this loss? I don't really know if this could be useful outside of RL.
```

##### [22-02-14] [paper179]
- Laplace Redux -- Effortless Bayesian Deep Learning
 [[pdf]](https://arxiv.org/abs/2106.14806) [[code]](https://github.com/AlexImmer/Laplace) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Laplace%20Redux%20--%20Effortless%20Bayesian%20Deep%20Learning.pdf)
- *Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, Philipp Hennig*
- `NeurIPS 2021`
- [Uncertainty Estimation]
```
Interesting and very well-written paper, I enjoyed reading it. I still think that ensembling probably is quite difficult to beat purely in terms of uncertainty estimation quality, but this definitely seems like a useful tool in many situations. It's not clear to me if the analytical expression for regression in "4. Approximate Predictive Distribution" is applicable also if the variance is input-dependent?
```

##### [22-02-12] [paper178]
- Benchmarking Uncertainty Quantification on Biosignal Classification Tasks under Dataset Shift
 [[pdf]](https://arxiv.org/abs/2112.09196?context=cs) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Benchmarking%20Uncertainty%20Quantification%20on%20Biosignal%20Classification%20Tasks%20under%20Dataset%20Shift.pdf)
- *Tong Xia, Jing Han, Cecilia Mascolo*
- `AAAI Workshops 2022`
- [Uncertainty Estimation], [Out-of-Distribution Detection], [ML for Medicine/Healthcare]
```
Well-written and interesting paper. They synthetically create dataset shifts (e.g. by adding Gaussian noise to the data) of increasing intensity and study whether or not the uncertainty increases as the accuracy degrades. They compare regular softmax, temperature scaling, MC-dropout, ensembling and a simple variational inference method. Their conclusion is basically that ensembling slightly outperforms the other methods, but that no method performs overly well. I think these type of studies are really useful.
```

##### [22-02-12] [paper177]
- Deep Evidential Regression
 [[pdf]](https://arxiv.org/abs/1910.02600) [[code]](https://github.com/aamini/evidential-deep-learning) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Evidential%20Regression.pdf)
- *Alexander Amini, Wilko Schwarting, Ava Soleimany, Daniela Rus*
- `NeurIPS 2020`
- [Uncertainty Estimation], [Out-of-Distribution Detection]
```
Well-written and interesting paper. This is a good paper to read before "Natural Posterior Network: Deep Bayesian Predictive Uncertainty for Exponential Family Distributions". Their proposed method seems to have similar / slightly worse performance than a small ensemble, so the only real advantage is that it's faster at time-time? This is of course very important in many applications, but not in all. The performance also seems quite sensitive to the choice of lambda in the combined loss function (Equation (10)), according to Figure S2 in the appendix?
```

##### [22-02-11] [paper176]
- On Out-of-distribution Detection with Energy-based Models
 [[pdf]](https://arxiv.org/abs/2107.08785) [[code]](https://github.com/selflein/EBM-OOD-Detection) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20Out-of-distribution%20Detection%20with%20Energy-based%20Models.pdf)
- *Sven Elflein, Bertrand Charpentier, Daniel Zügner, Stephan Günnemann*
- `ICML Workshops 2021`
- [Out-of-Distribution Detection], [Energy-Based Models]
```
Well-written and quite interesting paper. A short paper, just 4 pages. They don't study the method from the "Energy-based Out-of-distribution Detection" paper as I had expected, but it was still a quite interesting read. The results in Section 4.2 seem interesting, especially for experiment 3, but I'm not sure that I properly understand everything.
```

##### [22-02-10] [paper175]
- Natural Posterior Network: Deep Bayesian Predictive Uncertainty for Exponential Family Distributions
 [[pdf]](https://openreview.net/forum?id=tV3N0DWMxCg) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Natural%20Posterior%20Network:%20Deep%20Bayesian%20Predictive%20Uncertainty%20for%20Exponential%20Family%20Distributions.pdf)
- *Bertrand Charpentier, Oliver Borchert, Daniel Zügner, Simon Geisler, Stephan Günnemann*
- `ICLR 2022`
- [Uncertainty Estimation], [Out-of-Distribution Detection]
```
Interesting and well-written paper. I didn't quite understand all the details, I'll have to read a couple of related/background papers to be able to properly appreciate and evaluate the proposed method. I definitely feel like I would like to read up on this family of methods. Extensive experimental evaluation, and the results seem promising overall.
```

##### [22-02-09] [paper174]
- Energy-based Out-of-distribution Detection
 [[pdf]](https://arxiv.org/abs/2010.03759) [[code]](https://github.com/wetliu/energy_ood) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Energy-based%20Out-of-distribution%20Detection.pdf)
- *Weitang Liu, Xiaoyun Wang, John D. Owens, Yixuan Li*
- `NeurIPS 2020`
- [Out-of-Distribution Detection], [Energy-Based Models]
```
Interesting and well-written paper. The proposed method is quite clearly explained and makes intuitive sense (at least if you're familiar with EBMs). Compared to using the softmax score, the performance does seem to improve consistently. Seems like fine-tuning on an "auxiliary outlier dataset" is required to get really good performance though, which you can't really assume to have access to in real-world problems, I suppose?
```

##### [22-02-09] [paper173]
- VOS: Learning What You Don't Know by Virtual Outlier Synthesis
 [[pdf]](https://arxiv.org/abs/2202.01197) [[code]](https://github.com/deeplearning-wisc/vos) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/VOS:%20Learning%20What%20You%20Don't%20Know%20by%20Virtual%20Outlier%20Synthesis.pdf)
- *Xuefeng Du, Zhaoning Wang, Mu Cai, Yixuan Li*
- `ICLR 2022`
- [Out-of-Distribution Detection]
```
Interesting and quite well-written paper. I did find it somewhat difficult to understand certain parts though, they could perhaps be explained more clearly. The results seem quite impressive (they do consistently outperform all baselines), but I find it interesting that the "Gaussian noise" baseline in Table 2 performs that well? I should probably have read "Energy-based Out-of-distribution Detection" before reading this paper.
```

#### Papers Read in 2021:

##### [21-12-16] [paper172]
- Efficiently Modeling Long Sequences with Structured State Spaces
 [[pdf]](https://arxiv.org/abs/2111.00396) [[code]](https://github.com/HazyResearch/state-spaces) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Efficiently%20Modeling%20Long%20Sequences%20with%20Structured%20State%20Spaces.pdf)
- *Albert Gu, Karan Goel, Christopher Ré*
- `ICLR 2022`
- [Sequence Modeling]
```
Very interesting and quite well-written paper. Kind of neat/fun to see state-space models being used. The experimental results seem very impressive!? I didn't fully understand everything in Section 3. I had to read Section 3.4 a couple of times to understand how the parameterization actually works in practice (you have H state-space models, one for each feature dimension, so that you can map a sequence of feature vectors to another sequence of feature vectors) (and you can then also have multiple such layers of state-space models, mapping sequence --> sequence --> sequence --> ....).
```

##### [21-12-09] [paper171]
- Periodic Activation Functions Induce Stationarity
 [[pdf]](https://arxiv.org/abs/2110.13572) [[code]](https://github.com/AaltoML/PeriodicBNN) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Periodic%20Activation%20Functions%20Induce%20Stationarity.pdf)
- *Lassi Meronen, Martin Trapp, Arno Solin*
- `NeurIPS 2021`
- [Uncertainty Estimation], [Out-of-Distribution Detection]
```
Quite interesting and well-written paper. Quite a heavy read, probably need to be rather familiar with GPs to properly understand/appreciate everything. Definitely check Appendix D, it gives a better understanding of how the proposed method is applied in practice. I'm not quite sure how strong/impressive the experimental results actually are. Also seems like the method could be a bit inconvenient to implement/use?
```

##### [21-12-03] [paper170]
- Reliable and Trustworthy Machine Learning for Health Using Dataset Shift Detection
 [[pdf]](https://arxiv.org/abs/2110.14019) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Reliable%20and%20Trustworthy%20Machine%20Learning%20for%20Health%20Using%20Dataset%20Shift%20Detection.pdf)
- *Chunjong Park, Anas Awadalla, Tadayoshi Kohno, Shwetak Patel*
- `NeurIPS 2021`
- [Out-of-Distribution Detection], [ML for Medicine/Healthcare]
```
Interesting and very well-written paper. Gives a good overview of the field and contains a lot of seemingly useful references. The evaluation is very comprehensive. The user study is quite neat.
```

##### [21-12-02] [paper169]
- An Information-theoretic Approach to Distribution Shifts
 [[pdf]](https://arxiv.org/abs/2106.03783) [[code]](https://github.com/mfederici/dsit) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/An%20Information-theoretic%20Approach%20to%20Distribution%20Shifts.pdf)
- *Marco Federici, Ryota Tomioka, Patrick Forré*
- `NeurIPS 2021`
- [Theoretical Properties of Deep Learning]
```
Quite well-written paper overall that seemed interesting, but I found it very difficult to properly understand everything. Thus, I can't really tell how interesting/significant their analysis actually is.
```

##### [21-11-25] [paper168]
- On the Importance of Gradients for Detecting Distributional Shifts in the Wild
 [[pdf]](https://arxiv.org/abs/2110.00218) [[code]](https://github.com/deeplearning-wisc/gradnorm_ood) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20the%20Importance%20of%20Gradients%20for%20Detecting%20Distributional%20Shifts%20in%20the%20Wild.pdf)
- *Rui Huang, Andrew Geng, Yixuan Li*
- `NeurIPS 2021`
- [Out-of-Distribution Detection]
```
Quite interesting and well-written paper. The experimental results do seem promising. However, I don't quite get why the proposed method intuitively makes sense, why is it better to only use the parameters of the final network layer?
```

##### [21-11-18] [paper167]
- Masked Autoencoders Are Scalable Vision Learners
 [[pdf]](https://arxiv.org/abs/2111.06377) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners.pdf)
- *Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick*
- `CVPR 2022`
- [Representation Learning]
```
Interesting and well-written paper. The proposed method is simple and makes a lot of intuitive sense, which is rather satisfying. After page 4, there's mostly just detailed ablations and results.
```

##### [21-11-11] [paper166]
- Transferring Inductive Biases through Knowledge Distillation
 [[pdf]](https://arxiv.org/abs/2006.00555) [[code]](https://github.com/samiraabnar/Reflect) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Transferring%20Inductive%20Biases%20through%20Knowledge%20Distillation.pdf)
- *Samira Abnar, Mostafa Dehghani, Willem Zuidema*
- `2020-05`
- [Theoretical Properties of Deep Learning]
```
Quite well-written and somewhat interesting paper. I'm not very familiar with this area. I didn't spend too much time trying to properly evaluate the significance of the findings.
```

##### [21-10-28] [paper165]
- Deep Classifiers with Label Noise Modeling and Distance Awareness
 [[pdf]](https://arxiv.org/abs/2110.02609#) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Classifiers%20with%20Label%20Noise%20Modeling%20and%20Distance%20Awareness.pdf)
- *Vincent Fortuin, Mark Collier, Florian Wenzel, James Allingham, Jeremiah Liu, Dustin Tran, Balaji Lakshminarayanan, Jesse Berent, Rodolphe Jenatton, Effrosyni Kokiopoulou*
- `TMLR, 2022`
- [Uncertainty Estimation]
```
Quite interesting and well-written paper. I find the distance-awareness property more interesting than modelling of input/class-dependent label noise, so the proposed method (HetSNGP) is perhaps not overly interesting compared to the SNGP baseline.
```

##### [21-10-21] [paper164]
- Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets
 [[pdf]](https://mathai-iclr.github.io/papers/papers/MATHAI_29_paper.pdf) [[code]](https://github.com/openai/grok) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Grokking:%20Generalization%20Beyond%20Overfitting%20On%20Small%20Algorithmic%20Datasets.pdf)
- *Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra*
- `ICLR Workshops 2021`
- [Theoretical Properties of Deep Learning]
```
Somewhat interesting paper. The phenomena observed in Figure 1, that validation accuracy suddenly increases long after almost perfect fitting of the training data has been achieved is quite interesting. I didn't quite understand the datasets they use (binary operation tables).
```

##### [21-10-14] [paper163]
- Learning to Simulate Complex Physics with Graph Networks
 [[pdf]](https://arxiv.org/abs/2002.09405) [[code]](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20to%20Simulate%20Complex%20Physics%20with%20Graph%20Networks.pdf)
- *Alvaro Sanchez-Gonzalez, Jonathan Godwin, Tobias Pfaff, Rex Ying, Jure Leskovec, Peter W. Battaglia*
- `ICML 2020`
- [Graph Neural Networks]
```
Quite well-written and somewhat interesting paper. Cool application and a bunch of neat videos. This is not really my area, so I didn't spend too much time/energy trying to fully understand everything.
```

##### [21-10-12] [paper162]
- Neural Unsigned Distance Fields for Implicit Function Learning
 [[pdf]](https://arxiv.org/abs/2010.13938) [[code]](https://github.com/jchibane/ndf/) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Unsigned%20Distance%20Fields%20for%20Implicit%20Function%20Learning.pdf)
- *Julian Chibane, Aymen Mir, Gerard Pons-Moll*
- `NeurIPS 2020`
- [Implicit Neural Representations]
```
Interesting and very well-written paper, I really enjoyed reading it! The paper also gives a good understanding of neural implicit representations in general.
```

##### [21-10-08] [paper161]
- Probabilistic 3D Human Shape and Pose Estimation from Multiple Unconstrained Images in the Wild
 [[pdf]](https://arxiv.org/abs/2103.10978) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Probabilistic%203D%20Human%20Shape%20and%20Pose%20Estimation%20from%20Multiple%20Unconstrained%20Images%20in%20the%20Wild.pdf)
- *Akash Sengupta, Ignas Budvytis, Roberto Cipolla*
- `CVPR 2021`
- [3D Human Pose Estimation]
```
Well-written and quite interesting paper. I read it mainly as background for "Hierarchical Kinematic Probability Distributions for 3D Human Shape and Pose Estimation from Images in the Wild" which is written by exactly the same authors. In this paper, they predict a single Gaussian distribution for the pose (instead of hierarchical matrix-Fisher distributions). Also, they mainly focus on the body shape. They also use silhouettes + 2D keypoint heatmaps as input (instead of edge-filters + 2D keypoint heatmaps).
```

##### [21-10-08] [paper160]
- Synthetic Training for Accurate 3D Human Pose and Shape Estimation in the Wild
 [[pdf]](https://arxiv.org/abs/2009.10013) [[code]](https://github.com/akashsengupta1997/STRAPS-3DHumanShapePose) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Synthetic%20Training%20for%20Accurate%203D%20Human%20Pose%20and%20Shape%20Estimation%20in%20the%20Wild.pdf)
- *Akash Sengupta, Ignas Budvytis, Roberto Cipolla*
- `BMVC 2020`
- [3D Human Pose Estimation]
```
Well-written and farily interesting paper. I read it mainly as background for "Hierarchical Kinematic Probability Distributions for 3D Human Shape and Pose Estimation from Images in the Wild" which is written by exactly the same authors. In this paper, they just use direct regression. They also use silhouettes + 2D keypoint heatmaps as input (instead of edge-filters + 2D keypoint heatmaps).
```

##### [21-10-07] [paper159]
- Learning Motion Priors for 4D Human Body Capture in 3D Scenes
 [[pdf]](https://arxiv.org/abs/2108.10399) [[code]](https://github.com/sanweiliti/LEMO) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20Motion%20Priors%20for%204D%20Human%20Body%20Capture%20in%203D%20Scenes.pdf)
- *Siwei Zhang, Yan Zhang, Federica Bogo, Marc Pollefeys, Siyu Tang*
- `ICCV 2021`
- [3D Human Pose Estimation]
```
Well-written and quite interesting paper. I didn't fully understand everything though, and it feels like I probably don't know this specific setting/problem well enough to fully appreciate the paper. 
```

##### [21-10-07] [paper158]
- Hierarchical Kinematic Probability Distributions for 3D Human Shape and Pose Estimation from Images in the Wild
 [[pdf]](https://arxiv.org/abs/2110.00990) [[code]](https://github.com/akashsengupta1997/HierarchicalProbabilistic3DHuman) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Hierarchical%20Kinematic%20Probability%20Distributions%20for%203D%20Human%20Shape%20and%20Pose%20Estimation%20from%20Images%20in%20the%20Wild.pdf)
- *Akash Sengupta, Ignas Budvytis, Roberto Cipolla*
- `ICCV 2021`
- [3D Human Pose Estimation]
```
Well-written and very interesting paper, I enjoyed reading it. The hierarchical distribution prediction approach makes sense and consistently outperforms the independent baseline. Using matrix-Fisher distributions makes sense. The synthetic training framework and the input representation of edge-filters + 2D keypoint heatmaps are both interesting.
```

##### [21-10-06] [paper157]
- SMD-Nets: Stereo Mixture Density Networks
 [[pdf]](https://arxiv.org/abs/2104.03866) [[code]](https://github.com/fabiotosi92/SMD-Nets) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/SMD-Nets:%20Stereo%20Mixture%20Density%20Networks.pdf)
- *Fabio Tosi, Yiyi Liao, Carolin Schmitt, Andreas Geiger*
- `CVPR 2021`
- [Uncertainty Estimation]
```
Well-written and interesting paper. Quite easy to read and follow, the method is clearly explained and makes intuitive sense.
```

##### [21-10-04] [paper156]
- We are More than Our Joints: Predicting how 3D Bodies Move
 [[pdf]](https://arxiv.org/abs/2012.00619) [[code]](https://github.com/yz-cnsdqz/MOJO-release) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/We%20are%20More%20than%20Our%20Joints:%20Predicting%20how%203D%20Bodies%20Move.pdf)
- *Yan Zhang, Michael J. Black, Siyu Tang*
- `CVPR 2021`
- [3D Human Pose Estimation]
```
Well-written and fairly interesting paper. The marker-based representation, instead of using skeleton joints, makes sense. The recursive projection scheme also makes sense, but seems very slow (2.27 sec/frame)? I didn't quite get all the details for their DCT representation of the latent space.
```

##### [21-10-03] [paper155]
- imGHUM: Implicit Generative Models of 3D Human Shape and Articulated Pose
 [[pdf]](https://arxiv.org/abs/2108.10842) [[code]](https://github.com/google-research/google-research/tree/master/imghum) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/imGHUM:%20Implicit%20Generative%20Models%20of%203D%20Human%20Shape%20and%20Articulated%20Pose.pdf)
- *Thiemo Alldieck, Hongyi Xu, Cristian Sminchisescu*
- `ICCV 2021`
- [3D Human Pose Estimation], [Implicit Neural Representations]
```
Interesting and very well-written paper, I really enjoyed reading it. Interesting combination of implicit representations and 3D human modelling. The "inclusive human modelling" application is neat and important.
```

##### [21-10-03] [paper154]
- DI-Fusion: Online Implicit 3D Reconstruction with Deep Priors
 [[pdf]](https://arxiv.org/abs/2012.05551) [[code]](https://github.com/huangjh-pub/di-fusion) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/DI-Fusion:%20Online%20Implicit%203D%20Reconstruction%20with%20Deep%20Priors.pdf)
- *Jiahui Huang, Shi-Sheng Huang, Haoxuan Song, Shi-Min Hu*
- `CVPR 2021`
- [Implicit Neural Representations]
```
Well-written and interesting paper, I enjoyed reading it. Neat application of implicit representations. The paper also gives a quite good overview of online 3D reconstruction in general.
```

##### [21-10-02] [paper153]
- Contextually Plausible and Diverse 3D Human Motion Prediction
 [[pdf]](https://arxiv.org/abs/1912.08521) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Contextually%20Plausible%20and%20Diverse%203D%20Human%20Motion%20Prediction.pdf)
- *Sadegh Aliakbarian, Fatemeh Sadat Saleh, Lars Petersson, Stephen Gould, Mathieu Salzmann*
- `ICCV 2021`
- [3D Human Pose Estimation]
```
Well-written and quite interesting paper. The main idea, using a learned conditional prior p(z|c) instead of just p(z), makes sense and was shown beneficial also in "HuMoR: 3D Human Motion Model for Robust Pose Estimation". I'm however somewhat confused by their specific implementation in Section 4, doesn't seem like a standard cVAE implementation?
```

##### [21-10-01] [paper152]
- Local Implicit Grid Representations for 3D Scenes
 [[pdf]](https://arxiv.org/abs/2003.08981) [[code]](https://github.com/tensorflow/graphics/tree/master/tensorflow_graphics/projects/local_implicit_grid) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Local%20Implicit%20Grid%20Representations%20for%203D%20Scenes.pdf)
- *Chiyu Max Jiang, Avneesh Sud, Ameesh Makadia, Jingwei Huang, Matthias Nießner, Thomas Funkhouser*
- `CVPR 2020`
- [Implicit Neural Representations]
```
Well-written and quite interesting paper. Interesting application, being able to reconstruct full 3D scenes from sparse point clouds. I didn't fully understand everything, as I don't have a particularly strong graphics background.
```

##### [21-09-29] [paper151]
- Information Dropout: Learning Optimal Representations Through Noisy Computation
 [[pdf]](https://arxiv.org/abs/1611.01353) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Information%20Dropout:%20Learning%20Optimal%20Representations%20Through%20Noisy%20Computation.pdf)
- *Alessandro Achille, Stefano Soatto*
- `2016-11`
- [Representation Learning]
```
Well-written and somewhat interesting paper overall. I'm not overly familiar with the topics of the paper, and didn't fully understand everything. Some results and insights seem quite interesting/neat, but I'm not sure exactly what the main takeaways should be, or how significant they actually are.
```

##### [21-09-24] [paper150]
- Encoder-decoder with Multi-level Attention for 3D Human Shape and Pose Estimation
 [[pdf]](https://arxiv.org/abs/2109.02303) [[code]](https://github.com/ziniuwan/maed) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Encoder-decoder%20with%20Multi-level%20Attention%20for%203D%20Human%20Shape%20and%20Pose%20Estimation.pdf)
- *Ziniu Wan, Zhengjia Li, Maoqing Tian, Jianbo Liu, Shuai Yi, Hongsheng Li*
- `ICCV 2021`
- [3D Human Pose Estimation]
```
Well-written and fairly interesting paper. Quite a lot of details on the attention architecture, which I personally don't find overly interesting. The experimental results are quite impressive, but I would like to see a comparison in terms of computational cost at test-time. It sounds like their method is rather slow.
```

##### [21-09-23] [paper149]
- Physics-based Human Motion Estimation and Synthesis from Videos
 [[pdf]](https://arxiv.org/abs/2109.09913) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Physics-based%20Human%20Motion%20Estimation%20and%20Synthesis%20from%20Videos.pdf)
- *Kevin Xie, Tingwu Wang, Umar Iqbal, Yunrong Guo, Sanja Fidler, Florian Shkurti*
- `ICCV 2021`
- [3D Human Pose Estimation]
```
Well-written and quite interesting paper. The general idea, refining frame-by-frame pose estimates via physical constraints, intuitively makes a lot of sense. I did however find it quite difficult to understand all the details in Section 3.
```

##### [21-09-21] [paper148]
- Hierarchical VAEs Know What They Don't Know
 [[pdf]](https://arxiv.org/abs/2102.08248) [[code]](https://github.com/JakobHavtorn/hvae-oodd) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Hierarchical%20VAEs%20Know%20What%20They%20Don't%20Know.pdf)
- *Jakob D. Havtorn, Jes Frellsen, Søren Hauberg, Lars Maaløe*
- `ICML 2021`
- [Uncertainty Estimation], [VAEs]
```
Very well-written and quite interesting paper, I enjoyed reading it. Everything is quite well-explained, it's relatively easy to follow. The paper provides a good overview of the out-of-distribution detection problem and current methods.
```

##### [21-09-17] [paper147]
- Human Pose Regression with Residual Log-likelihood Estimation
 [[pdf]](https://arxiv.org/abs/2107.11291) [[code]](https://github.com/Jeff-sjtu/res-loglikelihood-regression) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Human%20Pose%20Regression%20with%20Residual%20Log-likelihood%20Estimation.pdf)
- *Jiefeng Li, Siyuan Bian, Ailing Zeng, Can Wang, Bo Pang, Wentao Liu, Cewu Lu*
- `ICCV 2021`
- [3D Human Pose Estimation]
```
Quite interesting paper, but also quite strange/confusing. I don't think the proposed method is explained particularly well, at least I found it quite difficult to properly understand what they actually are doing. In the end it seems like they are learning a global loss function that is very similar to doing probabilistic regression with a Gauss/Laplace model of p(y|x) (with learned mean and variance)? See Figure 4 in the Appendix. And while it's true that their performance is much better than for direct regression with an L2/L1 loss (see e.g. Table 1), they only compare with Gauss/Laplace probabilistic regression once (Table 7) and in that case the Laplace model is actually quite competitive?
```

##### [21-09-15] [paper146]
- NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
 [[pdf]](https://arxiv.org/abs/2003.08934) [[code]](https://github.com/bmild/nerf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/NeRF:%20Representing%20Scenes%20as%20Neural%20Radiance%20Fields%20for%20View%20Synthesis.pdf)
- *Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng*
- `ECCV 2020`
- [Implicit Neural Representations]
```
Extremely well-written and interesting paper. I really enjoyed reading it, and I would recommend anyone interested in computer vision to read it as well. All parts of the proposed method are clearly explained and relatively easy to understand, including the volume rendering techniques which I was unfamiliar with.
```

##### [21-09-08] [paper145]
- Revisiting the Calibration of Modern Neural Networks
 [[pdf]](https://arxiv.org/abs/2106.07998) [[code]](https://github.com/google-research/robustness_metrics/tree/master/robustness_metrics/projects/revisiting_calibration) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Revisiting%20the%20Calibration%20of%20Modern%20Neural%20Networks.pdf)
- *Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby, Dustin Tran, Mario Lucic*
- `NeurIPS 2021`
- [Uncertainty Estimation]
```
Well-written paper. Everything is quite clearly explained and easy to understand. Quite enjoyable to read overall. Thorough experimental evaluation. Quite interesting findings.
```

##### [21-09-02] [paper144]
- Differentiable Particle Filtering via Entropy-Regularized Optimal Transport
 [[pdf]](https://arxiv.org/abs/2102.07850) [[code]](https://github.com/JTT94/filterflow) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Differentiable%20Particle%20Filtering%20via%20Entropy-Regularized%20Optimal%20Transport.pdf)
- *Adrien Corenflos, James Thornton, George Deligiannidis, Arnaud Doucet*
- `ICML 2021`
- [Sequence Modeling]

##### [21-09-02] [paper143]
- Character Controllers Using Motion VAEs
 [[pdf]](https://arxiv.org/abs/2103.14274) [[code]](https://github.com/electronicarts/character-motion-vaes) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Character%20Controllers%20Using%20Motion%20VAEs.pdf)
- *Hung Yu Ling, Fabio Zinno, George Cheng, Michiel van de Panne*
- `SIGGRAPH 2020`
- [3D Human Pose Estimation]

##### [21-08-27] [paper142]
- DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation
 [[pdf]](https://arxiv.org/abs/1901.05103) [[code]](https://github.com/facebookresearch/DeepSDF) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/DeepSDF:%20Learning%20Continuous%20Signed%20Distance%20Functions%20for%20Shape%20Representation.pdf)
- *Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, Steven Lovegrove*
- `CVPR 2019`
- [Implicit Neural Representations]

##### [21-06-19] [paper141]
- Generating Multiple Hypotheses for 3D Human Pose Estimation with Mixture Density Network
 [[pdf]](https://arxiv.org/abs/1904.05547) [[code]](https://github.com/chaneyddtt/Generating-Multiple-Hypotheses-for-3D-Human-Pose-Estimation-with-Mixture-Density-Network) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Generating%20Multiple%20Hypotheses%20for%203D%20Human%20Pose%20Estimation%20with%20Mixture%20Density%20Network.pdf)
- *Chen Li, Gim Hee Lee*
- `CVPR 2019`
- [3D Human Pose Estimation]

##### [21-06-19] [paper140]
- Expressive Body Capture: 3D Hands, Face, and Body from a Single Image
 [[pdf]](https://arxiv.org/abs/1904.05866) [[code]](https://github.com/vchoutas/smplify-x) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Expressive%20Body%20Capture:%203D%20Hands%2C%20Face%2C%20and%20Body%20from%20a%20Single%20Image.pdf)
- *Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, Michael J. Black*
- `CVPR 2019`
- [3D Human Pose Estimation]
```
Very well-written and quite interesting paper. Gives a good understanding of the SMPL model and the SMPLify method.
```

##### [21-06-18] [paper139]
- Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image
 [[pdf]](https://arxiv.org/abs/1607.08128) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Keep%20it%20SMPL:%20Automatic%20Estimation%20of%203D%20Human%20Pose%20and%20Shape%20from%20a%20Single%20Image.pdf)
- *Federica Bogo, Angjoo Kanazawa, Christoph Lassner, Peter Gehler, Javier Romero, Michael J. Black*
- `ECCV 2016`
- [3D Human Pose Estimation]

##### [21-06-18] [paper138]
- Beyond Static Features for Temporally Consistent 3D Human Pose and Shape from a Video
 [[pdf]](https://arxiv.org/abs/2011.08627) [[code]](https://github.com/hongsukchoi/TCMR_RELEASE) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Beyond%20Static%20Features%20for%20Temporally%20Consistent%203D%20Human%20Pose%20and%20Shape%20from%20a%20Video.pdf)
- *Hongsuk Choi, Gyeongsik Moon, Ju Yong Chang, Kyoung Mu Lee*
- `CVPR 2021`
- [3D Human Pose Estimation]

##### [21-06-17] [paper137]
- Exemplar Fine-Tuning for 3D Human Model Fitting Towards In-the-Wild 3D Human Pose Estimation
 [[pdf]](https://arxiv.org/abs/2004.03686) [[code]](https://github.com/facebookresearch/eft) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Exemplar%20Fine-Tuning%20for%203D%20Human%20Model%20Fitting%20Towards%20In-the-Wild%203D%20Human%20Pose%20Estimation.pdf)
- *Hanbyul Joo, Natalia Neverova, Andrea Vedaldi*
- `3DV 2021`
- [3D Human Pose Estimation]

##### [21-06-17] [paper136]
- Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop
 [[pdf]](https://arxiv.org/abs/1909.12828) [[code]](https://github.com/nkolot/SPIN) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20to%20Reconstruct%203D%20Human%20Pose%20and%20Shape%20via%20Model-fitting%20in%20the%20Loop.pdf)
- *Nikos Kolotouros, Georgios Pavlakos, Michael J. Black, Kostas Daniilidis*
- `ICCV 2019`
- [3D Human Pose Estimation]

##### [21-06-16] [paper135]
- A simple yet effective baseline for 3d human pose estimation
 [[pdf]](https://arxiv.org/abs/1705.03098) [[code]](https://github.com/una-dinosauria/3d-pose-baseline) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20simple%20yet%20effective%20baseline%20for%203d%20human%20pose%20estimation.pdf)
- *Julieta Martinez, Rayat Hossain, Javier Romero, James J. Little*
- `ICCV 2017`
- [3D Human Pose Estimation]

##### [21-06-16] [paper134]
- Estimating Egocentric 3D Human Pose in Global Space
 [[pdf]](https://arxiv.org/abs/2104.13454) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Estimating%20Egocentric%203D%20Human%20Pose%20in%20Global%20Space.pdf)
- *Jian Wang, Lingjie Liu, Weipeng Xu, Kripasindhu Sarkar, Christian Theobalt*
- `ICCV 2021`
- [3D Human Pose Estimation]

##### [21-06-15] [paper133]
- End-to-end Recovery of Human Shape and Pose
 [[pdf]](https://arxiv.org/abs/1712.06584) [[code]](https://github.com/akanazawa/hmr) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/End-to-end%20Recovery%20of%20Human%20Shape%20and%20Pose.pdf)
- *Angjoo Kanazawa, Michael J. Black, David W. Jacobs, Jitendra Malik*
- `CVPR 2018`
- [3D Human Pose Estimation]

##### [21-06-14] [paper132]
- 3D Multi-bodies: Fitting Sets of Plausible 3D Human Models to Ambiguous Image Data
 [[pdf]](https://arxiv.org/abs/2011.00980) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/3D%20Multi-bodies:%20Fitting%20Sets%20of%20Plausible%203D%20Human%20Models%20to%20Ambiguous%20Image%20Data.pdf)
- *Benjamin Biggs, Sébastien Ehrhadt, Hanbyul Joo, Benjamin Graham, Andrea Vedaldi, David Novotny*
- `NeurIPS 2020`
- [3D Human Pose Estimation]

##### [21-06-04] [paper131]
- HuMoR: 3D Human Motion Model for Robust Pose Estimation
 [[pdf]](https://arxiv.org/abs/2105.04668) [[code]](https://geometry.stanford.edu/projects/humor/) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/HuMoR:%203D%20Human%20Motion%20Model%20for%20Robust%20Pose%20Estimation.pdf)
- *Davis Rempe, Tolga Birdal, Aaron Hertzmann, Jimei Yang, Srinath Sridhar, Leonidas J. Guibas*
- `ICCV 2021`
- [3D Human Pose Estimation]

##### [21-05-07] [paper130]
- PixelTransformer: Sample Conditioned Signal Generation
 [[pdf]](https://arxiv.org/abs/2103.15813) [[code]](https://github.com/shubhtuls/PixelTransformer) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/PixelTransformer:%20Sample%20Conditioned%20Signal%20Generation.pdf)
- *Shubham Tulsiani, Abhinav Gupta*
- `ICML 2021`
- [Neural Processes], [Transformers]

##### [21-04-29] [paper129]
- Stiff Neural Ordinary Differential Equations
 [[pdf]](https://arxiv.org/abs/2103.15341) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Stiff%20Neural%20Ordinary%20Differential%20Equations.pdf)
- *Suyong Kim, Weiqi Ji, Sili Deng, Yingbo Ma, Christopher Rackauckas*
- `2021-03`
- [Neural ODEs]

##### [21-04-16] [paper128]
- Learning Mesh-Based Simulation with Graph Networks
 [[pdf]](https://arxiv.org/abs/2010.03409) [[code]](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20Mesh-Based%20Simulation%20with%20Graph%20Networks.pdf)
- *Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, Peter W. Battaglia*
- `ICLR 2021`
- [Graph Neural Networks]

##### [21-04-09] [paper127]
- Q-Learning in enormous action spaces via amortized approximate maximization
 [[pdf]](https://arxiv.org/abs/2001.08116) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Q-Learning%20in%20enormous%20action%20spaces%20via%20amortized%20approximate%20maximization.pdf)
- *Tom Van de Wiele, David Warde-Farley, Andriy Mnih, Volodymyr Mnih*
- `2020-01`
- [Reinforcement Learning]

##### [21-04-01] [paper126]
- Loss Surface Simplexes for Mode Connecting Volumes and Fast Ensembling
 [[pdf]](https://arxiv.org/abs/2102.13042) [[code]](https://github.com/g-benton/loss-surface-simplexes) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Loss%20Surface%20Simplexes%20for%20Mode%20Connecting%20Volumes%20and%20Fast%20Ensembling.pdf)
- *Gregory W. Benton, Wesley J. Maddox, Sanae Lotfi, Andrew Gordon Wilson*
- `ICML 2021`
- [Uncertainty Estimation], [Ensembling]

##### [21-03-26] [paper125]
- Your GAN is Secretly an Energy-based Model and You Should use Discriminator Driven Latent Sampling
 [[pdf]](https://arxiv.org/abs/2003.06060) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Your%20GAN%20is%20Secretly%20an%20Energy-based%20Model%20and%20You%20Should%20Use%20Discriminator%20Driven%20Latent%20Sampling.pdf)
- *Tong Che, Ruixiang Zhang, Jascha Sohl-Dickstein, Hugo Larochelle, Liam Paull, Yuan Cao, Yoshua Bengio*
- `NeurIPS 2020`
- [Energy-Based Models]

##### [21-03-19] [paper124]
- Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability
 [[pdf]](https://arxiv.org/abs/2103.00065) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Gradient%20Descent%20on%20Neural%20Networks%20Typically%20Occurs%20at%20the%20Edge%20of%20Stability.pdf)
- *Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, Ameet Talwalkar*
- `ICLR 2021`
- [Theoretical Properties of Deep Learning]

##### [21-03-12] [paper123]
- Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
 [[pdf]](https://arxiv.org/abs/2006.09882) [[code]](https://github.com/facebookresearch/swav) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Unsupervised%20Learning%20of%20Visual%20Features%20by%20Contrasting%20Cluster%20Assignments.pdf)
- *Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, Armand Joulin*
- `NeurIPS 2020`
- [Representation Learning]

##### [21-03-04] [paper122]
- Infinitely Deep Bayesian Neural Networks with Stochastic Differential Equations
 [[pdf]](https://arxiv.org/abs/2102.06559) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Infinitely%20Deep%20Bayesian%20Neural%20Networks%20with%20Stochastic%20Differential%20Equations.pdf)
- *Winnie Xu, Ricky T.Q. Chen, Xuechen Li, David Duvenaud*
- `AISTATS 2022`
- [Neural ODEs], [Uncertainty Estimation]

##### [21-02-26] [paper121]
- Neural Relational Inference for Interacting Systems
 [[pdf]](https://arxiv.org/abs/1802.04687) [[code]](https://github.com/ethanfetaya/NRI) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Relational%20Inference%20for%20Interacting%20Systems.pdf)
- *Thomas Kipf, Ethan Fetaya, Kuan-Chieh Wang, Max Welling, Richard Zemel*
- `ICML 2018`
- [Miscellaneous]

##### [21-02-19] [paper120]
- Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision
 [[pdf]](https://arxiv.org/abs/2102.05918) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Scaling%20Up%20Visual%20and%20Vision-Language%20Representation%20Learning%20With%20Noisy%20Text%20Supervision.pdf)
- *Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig*
- `ICML 2021`
- [Representation Learning], [Vision-Language Models]

##### [21-02-12] [paper119]
- On the Origin of Implicit Regularization in Stochastic Gradient Descent
 [[pdf]](https://arxiv.org/abs/2101.12176) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20the%20Origin%20of%20Implicit%20Regularization%20in%20Stochastic%20Gradient%20Descent.pdf)
- *Samuel L. Smith, Benoit Dherin, David G. T. Barrett, Soham De*
- `ICLR 2021`
- [Theoretical Properties of Deep Learning]

##### [21-02-05] [paper118]
- Meta Pseudo Labels
 [[pdf]](https://arxiv.org/abs/2003.10580) [[code]](https://github.com/google-research/google-research/tree/master/meta_pseudo_labels) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Meta%20Pseudo%20Labels.pdf)
- *Hieu Pham, Zihang Dai, Qizhe Xie, Minh-Thang Luong, Quoc V. Le*
- `CVPR 2021`
- [Miscellaneous]

##### [21-01-29] [paper117]
- No MCMC for Me: Amortized Sampling for Fast and Stable Training of Energy-Based Models
 [[pdf]](https://arxiv.org/abs/2010.04230) [[code]](https://github.com/wgrathwohl/VERA) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/No%20MCMC%20for%20me:%20Amortized%20sampling%20for%20fast%20and%20stable%20training%20of%20energy-based%20models.pdf)
- *Will Grathwohl, Jacob Kelly, Milad Hashemi, Mohammad Norouzi, Kevin Swersky, David Duvenaud*
- `ICLR 2021`
- [Energy-Based Models]

##### [21-01-22] [paper116]
- Getting a CLUE: A Method for Explaining Uncertainty Estimates
 [[pdf]](https://arxiv.org/abs/2006.06848) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Getting%20a%20CLUE:%20A%20Method%20for%20Explaining%20Uncertainty%20Estimates.pdf)
- *Javier Antorán, Umang Bhatt, Tameem Adel, Adrian Weller, José Miguel Hernández-Lobato*
- `ICLR 2021`
- [Uncertainty Estimation]

##### [21-01-15] [paper115]
- Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
 [[pdf]](https://arxiv.org/abs/2006.16236) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Transformers%20are%20RNNs:%20Fast%20Autoregressive%20Transformers%20with%20Linear%20Attention.pdf)
- *Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret*
- `ICML 2020`
- [Transformers]

#### Papers Read in 2020:

##### [20-12-18] [paper114]
- Score-Based Generative Modeling through Stochastic Differential Equations
 [[pdf]](https://arxiv.org/abs/2011.13456) [[code]](https://github.com/yang-song/score_sde) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Score-Based%20Generative%20Modeling%20through%20Stochastic%20Differential%20Equations.pdf)
- *Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole*
- `ICLR 2021`
- [Diffusion Models]

##### [20-12-14] [paper113]
- Dissecting Neural ODEs
 [[pdf]](https://arxiv.org/abs/2002.08071) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Dissecting%20Neural%20ODEs.pdf)
- *Stefano Massaroli, Michael Poli, Jinkyoo Park, Atsushi Yamashita, Hajime Asama*
- `NeurIPS 2020`
- [Neural ODEs]

##### [20-11-27] [paper112]
- Rethinking Attention with Performers
 [[pdf]](https://arxiv.org/abs/2009.14794) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Rethinking%20Attention%20with%20Performers.pdf)
- *Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller*
- `ICLR 2021`
- [Transformers]

##### [20-11-23] [paper111]
- Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images
 [[pdf]](https://arxiv.org/abs/2011.10650) [[code]](https://github.com/openai/vdvae) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Very%20Deep%20VAEs%20Generalize%20Autoregressive%20Models%20and%20Can%20Outperform%20Them%20on%20Images.pdf)
- *Rewon Child*
- `ICLR 2021`
- [VAEs]

##### [20-11-13] [paper110]
- VAEBM: A Symbiosis between Variational Autoencoders and Energy-based Models
 [[pdf]](https://arxiv.org/abs/2010.00654) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/VAEBM:%20A%20Symbiosis%20between%20Variational%20Autoencoders%20and%20Energy-based%20Models.pdf)
- *Zhisheng Xiao, Karsten Kreis, Jan Kautz, Arash Vahdat*
- `ICLR 2021`
- [Energy-Based Models], [VAEs]

##### [20-11-06] [paper109]
- Approximate Inference Turns Deep Networks into Gaussian Processes
 [[pdf]](https://arxiv.org/abs/1906.01930) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Approximate%20Inference%20Turns%20Deep%20Networks%20into%20Gaussian%20Processes.pdf)
- *Mohammad Emtiyaz Khan, Alexander Immer, Ehsan Abedi, Maciej Korzepa*
- `NeurIPS 2019`
- [Theoretical Properties of Deep Learning]

##### [20-10-16] [paper108]
- Implicit Gradient Regularization [[pdf]](https://arxiv.org/abs/2009.11162) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Implicit%20Gradient%20Regularization.pdf)
- *David G.T. Barrett, Benoit Dherin*
- `ICLR 2021`
- [Theoretical Properties of Deep Learning]
```
Well-written and somewhat interesting paper. Quite interesting concept, makes some intuitive sense. Not sure if the experimental results were super convincing though.
```

##### [20-10-09] [paper107]
- Satellite Conjunction Analysis and the False Confidence Theorem [[pdf]](https://royalsocietypublishing.org/doi/pdf/10.1098/rspa.2018.0565) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Satellite%20conjunction%20analysis%20and%20the%20false%20confidence%20theorem.pdf)
- *Michael Scott Balch, Ryan Martin, Scott Ferson*
- `2018-03`
- [Miscellaneous]
```
Quite well-written and somewhat interesting paper. Section 6 (Future and on-going work) is IMO the most interesting part of the paper ("We recognize the natural desire to balance the goal of preventing collisions against the goal of keeping manoeuvres at a reasonable level, and we further recognize that it may not be possible to achieve an acceptable balance between these two goals using present tracking resources"). To me, it seems like the difference between their proposed approach and the standard approach is mainly just a change in how to interpret very uncertain satellite trajectories. In the standard approach, two very uncertain trajectories are deemed NOT likely to collide (the two satellites could be basically anywhere, so what are the chances they will collide?) . In their approach, they seem to instead say: "the two satellites could be basically anywhere, so they COULD collide!". They argue their approach prioritize safety (which I guess they do, they will check more trajectories since they COULD collide), but it must also actually be useful in practice. I mean, the safest way to drive a car is to just remain stationary at all times, otherwise you risk colliding with something.
```

##### [20-09-24] [paper106]
- Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness [[pdf]](https://arxiv.org/abs/2006.10108) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Simple%20and%20Principled%20Uncertainty%20Estimation%20with%20Deterministic%20Deep%20Learning%20via%20Distance%20Awareness.pdf)
- *Jeremiah Zhe Liu, Zi Lin, Shreyas Padhy, Dustin Tran, Tania Bedrax-Weiss, Balaji Lakshminarayanan*
- `NeurIPS 2020`
- [Uncertainty Estimation]
```
Interesting paper. Quite a heavy read (section 2 and 3). I didn't really spend enough time reading the paper to fully understand everything. The "distance awareness" concept intuitively makes a lot of sense, the example in Figure 1 is impressive, and the results on CIFAR10/100 are also encouraging. I did find section 3.1 quite confusing, Appendix A was definitely useful.
```

##### [20-09-21] [paper105]
- Uncertainty Estimation Using a Single Deep Deterministic Neural Network [[pdf]](https://arxiv.org/abs/2003.02037) [[code]](https://github.com/y0ast/deterministic-uncertainty-quantification) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20Estimation%20Using%20a%20Single%20Deep%20Deterministic%20Neural%20Network.pdf)
- *Joost van Amersfoort, Lewis Smith, Yee Whye Teh, Yarin Gal*
- `ICML 2020`
- [Uncertainty Estimation]
```
Well-written and quite interesting paper. Interesting and neat idea, it definitely makes some intuitive sense. In the end though, I was not overly impressed. Once they used the more realistic setup on the CIFAR10 experiment (not using a third dataset to tune lambda), the proposed method was outperformed by ensembling (also using quite few networks). Yes, their method is more computationally efficient at test time (which is indeed very important in many applications), but it also seems quite a lot less convenient to train, involves setting a couple of important hyperparameters and so on. Interesting method and a step in the right direction though.
```

##### [20-09-11] [paper104]
- Gated Linear Networks [[pdf]](https://arxiv.org/abs/1910.01526) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Gated%20Linear%20Networks.pdf)
- *Joel Veness, Tor Lattimore, David Budden, Avishkar Bhoopchand, Christopher Mattern, Agnieszka Grabska-Barwinska, Eren Sezener, Jianan Wang, Peter Toth, Simon Schmitt, Marcus Hutter*
- `AAAI 2021`
- [Miscellaneous]
```
Quite well-written and somewhat interesting paper. Interesting paper in the sense that it was quite different compared to basically all other papers I've read. The proposed method seemed odd in the beginning, but eventually I think I understood it reasonably well. Still not quite sure how useful GLNs actually would be in practice though. It seems promising for online/continual learning applications, but only toy examples were considered in the paper? I don't think I understand the method well enough to properly assess its potential impact.
```

##### [20-09-04] [paper103]
- Denoising Diffusion Probabilistic Models [[pdf]](https://arxiv.org/abs/2006.11239) [[code]](https://github.com/hojonathanho/diffusion) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Denoising%20Diffusion%20Probabilistic%20Models.pdf)
- *Jonathan Ho, Ajay Jain, Pieter Abbeel*
- `NeurIPS 2020`
- [Energy-Based Models]
```
Quite well-written and interesting paper. I do find the connection between "diffusion probabilistic models" and denoising score matching relatively interesting. Since I was not familiar with diffusion probabilistic models, the paper was however a quite heavy read, and the established connection didn't really improve my intuition (reading Generative Modeling by Estimating Gradients of the Data Distribution gave a better understanding of score matching, I think).
```

##### [20-06-18] [paper102]
- Joint Training of Variational Auto-Encoder and Latent Energy-Based Model [[pdf]](https://arxiv.org/abs/2006.06059) [[code]](https://hthth0801.github.io/jointLearning/) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Joint%20Training%20of%20Variational%20Auto-Encoder%20and%20Latent%20Energy-Based%20Model.pdf)
- *Tian Han, Erik Nijkamp, Linqi Zhou, Bo Pang, Song-Chun Zhu, Ying Nian Wu*
- `CVPR 2020`
- [VAEs], [Energy-Based Models]
```
Interesting and very well-written paper. Neat and interesting idea. The paper is well-written and provides a clear and quite intuitive description of EBMs, VAEs and other related work. The comment "Learning well-formed energy landscape remains a challenging problem, and our experience suggests that the learned energy function can be sensitive to the setting of hyper-parameters and within the training algorithm." is somewhat concerning.
```

##### [20-06-12] [paper101]
- End-to-End Object Detection with Transformers [[pdf]](https://arxiv.org/abs/2005.12872) [[code]](https://github.com/facebookresearch/detr) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/End-to-End%20Object%20Detection%20with%20Transformers.pdf)
- *Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko*
- `ECCV 2020`
- [Object Detection]
```
Interesting and well-written paper. Interesting and quite neat idea. Impressive results on object detection, and panoptic segmentation. It seems like the model requires longer training (500 vs 109 epochs?), and might be somewhat more difficult to train? Would be interesting to play around with the code. The "decoder output slot analysis" in Figure 7 is quite interesting. Would be interesting to further study what information has been captured in the object queries (which are just N vectors?) during training.
```

##### [20-06-05] [paper100]
- Efficient and Scalable Bayesian Neural Nets with Rank-1 Factors [[pdf]](https://arxiv.org/abs/2005.07186) [[code]](https://github.com/google/edward2) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Efficient%20and%20Scalable%20Bayesian%20Neural%20Nets%20with%20Rank-1%20Factors.pdf)
- *Michael W. Dusenberry, Ghassen Jerfel, Yeming Wen, Yi-an Ma, Jasper Snoek, Katherine Heller, Balaji Lakshminarayanan, Dustin Tran*
- `ICML 2020`
- [Uncertainty Estimation], [Variational Inference]
```
Quite well-written and interesting paper. Extenstion of the BatchEnsemble paper. Still a quite neat and simple idea, and performance seems to be consistently improved compared to BatchEnsemble. Not quite clear to me if the model is much more difficult to implement or train. Seems quite promising overall.
```

##### [20-05-27] [paper99]
- BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning [[pdf]](https://arxiv.org/abs/2002.06715) [[code]](https://github.com/google/edward2) [[video]](https://iclr.cc/virtual_2020/poster_Sklf1yrYDr.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/BatchEnsemble:%20An%20Alternative%20Approach%20to%20Efficient%20Ensemble%20and%20Lifelong%20Learning.pdf)
- *Yeming Wen, Dustin Tran, Jimmy Ba*
- `ICLR 2020`
- [Uncertainty Estimation], [Ensembling]
```
Quite interesting and well-written paper. Neat and quite simple idea. I am however not entirely sure how easy it is to implement, it must complicate things somewhat at least? Not overly impressed by the calibration/uncertainty experiments, the proposed method is actually quite significantly outperformed by standard ensembling. The decrease in test-time computational cost is however impressive.
```

##### [20-05-10] [paper98]
- Stable Neural Flows [[pdf]](https://arxiv.org/abs/2003.08063) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Stable%20Neural%20Flows%20.pdf)
- *Stefano Massaroli, Michael Poli, Michelangelo Bin, Jinkyoo Park, Atsushi Yamashita, Hajime Asama*
- `2020-03`
- [Neural ODEs]
```
Somewhat well-written and interesting paper. Somewhat odd paper, I did not properly understand everything. It is not clear to me how the energy functional used here is connected to energy-based models.
```

##### [20-04-17] [paper97]
- How Good is the Bayes Posterior in Deep Neural Networks Really? [[pdf]](https://arxiv.org/abs/2002.02405) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/How%20Good%20is%20the%20Bayes%20Posterior%20in%20Deep%20Neural%20Networks%20Really%3F.pdf)
- *Florian Wenzel, Kevin Roth, Bastiaan S. Veeling, Jakub Świątkowski, Linh Tran, Stephan Mandt, Jasper Snoek, Tim Salimans, Rodolphe Jenatton, Sebastian Nowozin*
- `ICML 2020`
- [Uncertainty Estimation], [Stochastic Gradient MCMC]
```
Somewhat well-written and interesting paper. Quite odd paper. They refer to the appendix a whole lot, this work is not really suited for an 8 page paper IMO. They present a bunch of hypotheses, but I do not quite know what to do with the results in the end. The paper is rather inconclusive. I found it somewhat odd that they only evaluate the methods in terms of predictive performance, that is usually not the reason why people turn to Bayesian deep learning models.
```

##### [20-04-09] [paper96]
- Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration [[pdf]](https://arxiv.org/abs/1910.12656) [[code]](https://github.com/dirichletcal/experiments_neurips) [[poster]](https://dirichletcal.github.io/documents/neurips2019/poster.pdf) [[slides]](https://dirichletcal.github.io/documents/neurips2019/slides.pdf) [[video]](https://dirichletcal.github.io/documents/neurips2019/video/Meelis_Ettekanne.mp4) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Beyond%20temperature%20scaling:%20Obtaining%20well-calibrated%20multiclass%20probabilities%20with%20Dirichlet%20calibration.pdf)
- *Meelis Kull, Miquel Perello-Nieto, Markus Kängsepp, Telmo Silva Filho, Hao Song, Peter Flach*
- `NeurIPS 2019`
- [Uncertainty Estimation]
```
Well-written and quite interesting paper. Does a good job describing different notions of calibration (Definition 1 - 3). Classwise-ECE intuitively makes sense as a reasonable metric. I did not quite follow the paragraph on interpretability (or figure 2). The experiments seem extensive and rigorously conducted. So, matrix scaling (with ODIR regularization) outperforms Dirichlet calibration?
```

##### [20-04-03] [paper95]
- Normalizing Flows: An Introduction and Review of Current Methods [[pdf]](https://arxiv.org/abs/1908.09257) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Normalizing%20Flows:%20An%20Introduction%20and%20Review%20of%20Current%20Methods.pdf)
- *Ivan Kobyzev, Simon Prince, Marcus A. Brubaker*
- `TPAMI, 2021`
- [Normalizing Flows]
```
Quite well-written and somewhat interesting paper. The paper is probably too short for it to actually fulfill the goal of "provide context and explanation to enable a reader to become familiar with the basics". It seems to me like one would have to have a pretty good understanding of normalizing flows, and various common variants, already beforehand to actually benefit much from this paper.
```

##### [20-03-27] [paper94]
- Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning [[pdf]](https://arxiv.org/abs/2002.06470) [[code]](https://github.com/bayesgroup/pytorch-ensembles) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Pitfalls%20of%20In-Domain%20Uncertainty%20Estimation%20and%20Ensembling%20in%20Deep%20Learning.pdf)
- *Arsenii Ashukha, Alexander Lyzhov, Dmitry Molchanov, Dmitry Vetrov*
- `ICLR 2020`
- [Uncertainty Estimation], [Ensembling], [Stochastic Gradient MCMC]
```
Quite well-written and interesting paper. The number of compared methods is quite impressive. The paper provides further evidence for what intuitively makes A LOT of sense: "Deep ensembles dominate other methods given a fixed test-time budget. The results indicate, in particular, that exploration of different modes in the loss landscape is crucial for good predictive performance". While deep ensembles might require a larger amount of total training time, they are extremely simple to train and separate ensemble members can be trained completely in parallel. Overall then, deep ensembles is a baseline that's extremely hard to beat IMO. Not convinced that "calibrated log-likelihood" is an ideal metric that addresses the described flaws of commonly used metrics. For example, "...especially calibrated log-likelihood is highly correlated with accuracy" does not seem ideal. Also, how would you generalize it to regression?
```

##### [20-03-26] [paper93]
- Conservative Uncertainty Estimation By Fitting Prior Networks [[pdf]](https://openreview.net/forum?id=BJlahxHYDS) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Conservative%20Uncertainty%20Estimation%20By%20Fitting%20Prior%20Networks.pdf)
- `ICLR 2020`
- [Uncertainty Estimation]
```
Interesting and somewhat well-written paper. I found it quite difficult to actually understand the method at first, I think the authors could have done a better job describing it. I guess that "f" should be replaced with "f_i" in equation (2)? "...the obtained uncertainties are larger than ones arrived at by Bayesian inference.", I did not quite get this though. The estimated uncertainty is conservative w.r.t. the posterior process associated with the prior process (the prior process defined by randomly initializing neural networks), but only if this prior process can be assumed to be Gaussian? So, do we actually have any guarantees? I am not sure if the proposed method actually is any less "hand-wavy" than e.g. ensembling. The experimental results seem quite promising, but I do not agree that this is "an extensive empirical comparison" (only experiments on CIFAR-10).
```

##### [20-03-09] [paper92]
- Batch Normalization Biases Deep Residual Networks Towards Shallow Paths [[pdf]](https://arxiv.org/abs/2002.10444) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Batch%20Normalization%20Biases%20Deep%20Residual%20Networks%20Towards%20Shallow%20Path.pdf)
- *Soham De, Samuel L. Smith*
- `NeurIPS 2020`
- [Theoretical Properties of Deep Learning]
```
Quite well-written and somewhat interesting paper. The fact that SkipInit enabled training of very deep networks without batchNorm is quite interesting. I don't think I fully understood absolutely everything.
```

##### [20-02-28] [paper91]
- Bayesian Deep Learning and a Probabilistic Perspective of Generalization [[pdf]](https://arxiv.org/abs/2002.08791) [[code]](https://github.com/izmailovpavel/understandingbdl) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Bayesian%20Deep%20Learning%20and%20a%20Probabilistic%20Perspective%20of%20Generalization.pdf)
- *Andrew Gordon Wilson, Pavel Izmailov*
- `NeurIPS 2020`
- [Uncertainty Estimation], [Ensembling]
```
Quite interesting and somewhat well-written paper. While I did find the paper quite interesting, I also found it somewhat confusing overall. The authors touch upon many different concepts, and the connection between them is not always very clear. It it not quite clear what the main selling point of the paper is. Comparing ensembling with MultiSWAG does not really seem fair to me, as MultiSWAG would be 20x slower at test-time. The fact that MultiSWA (note: MultiSWA, not MultiSWAG) seems to outperform ensembling quite consistently in their experiment is however quite interesting, it is not obvious to me why that should be the case.
```

##### [20-02-21] [paper90]
- Convolutional Conditional Neural Processes [[pdf]](https://arxiv.org/abs/1910.13556) [[code]](https://github.com/cambridge-mlg/convcnp) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Convolutional%20Conditional%20Neural%20Processes.pdf)
- *Jonathan Gordon, Wessel P. Bruinsma, Andrew Y. K. Foong, James Requeima, Yann Dubois, Richard E. Turner*
- `ICLR 2020`
- [Neural Processes]
```
Quite interesting and well-written paper. Took me a pretty long time to read this paper, it is a quite heavy/dense read. I still do not quite get when this type of model could/should be used in practice, all experiments in the paper seem at least somewhat synthetic to me.
```

##### [20-02-18] [paper89]
- Probabilistic 3D Multi-Object Tracking for Autonomous Driving [[pdf]](https://arxiv.org/abs/2001.05673) [[code]](https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Probabilistic%203D%20Multi-Object%20Tracking%20for%20Autonomous%20Driving.pdf)
- *Hsu-kuang Chiu, Antonio Prioletti, Jie Li, Jeannette Bohg*
- `ICRA 2021`
- [3D Multi-Object Tracking]
```
Interesting and well-written paper. They provide more details for the Kalman filter, which I appreciate. The design choices that differs compared to AB3DMOT all make sense I think (e.g., Mahalanobis distance instead of 3D-IoU as the affinity measure in the data association), but the gain in performance in Table 1 does not seem overly significant, at least not compared to the huge gain seen when switching to the MEGVII 3D detector in AB3DMOT.
```

##### [20-02-15] [paper88]
- A Baseline for 3D Multi-Object Tracking [[pdf]](https://arxiv.org/abs/1907.03961) [[code]](https://github.com/xinshuoweng/AB3DMOT) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Baseline%20for%203D%20Multi-Object%20Tracking.pdf)
- *Xinshuo Weng, Kris Kitani*
- `IROS 2020`
- [3D Multi-Object Tracking]
```
Well-written and interesting paper. Provides a neat introduction to 3D multi-object tracking in general, especially since the proposed method is intentionally straightforward and simple. It seems like a very good starting point. It is not clear to me exactly how the update step i in the Kalman filter is implemented? How did they set the covariance matrices? (I guess you could find this in the provided code though)
```

##### [20-02-14] [paper87]
- A Contrastive Divergence for Combining Variational Inference and MCMC [[pdf]](https://arxiv.org/abs/1905.04062) [[code]](https://github.com/franrruiz/vcd_divergence) [[slides]](https://franrruiz.github.io/contents/group_talks/EMS-Jul2019.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Contrastive%20Divergence%20for%20Combining%20Variational%20Inference%20and%20MCMC.pdf)
- *Francisco J. R. Ruiz, Michalis K. Titsias*
- `ICML 2019`
- [VAEs]
```
Interesting and very well-written paper. I feel like I never quite know how significant improvements such as those in Table 2 actually are. What would you get if you instead used a more complex variational family (e.g. flow-based) and fitted that via standard KL, would the proposed method outperform also this baseline? And how would those compare in terms of computational cost?
```

##### [20-02-13] [paper86]
- Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning [[pdf]](https://arxiv.org/abs/1710.07283) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Decomposition%20of%20Uncertainty%20in%20Bayesian%20Deep%20Learning%20for%20Efficient%20and%20Risk-sensitive%20Learning.pdf)
- *Stefan Depeweg, José Miguel Hernández-Lobato, Finale Doshi-Velez, Steffen Udluft*
- `ICML 2018`
- [Uncertainty Estimation], [Reinforcement Learning]
```
Well-written and quite interesting paper. Obviously similar to "Uncertainty Decomposition in Bayesian Neural Networks with Latent Variables", but contains more details, further experiments, and does a better job explaining some of the core concepts.
```

##### [20-02-08] [paper85]
- Uncertainty Decomposition in Bayesian Neural Networks with Latent Variables [[pdf]](https://arxiv.org/abs/1706.08495) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20Decomposition%20in%20Bayesian%20Neural%20Networks%20with%20Latent%20Variables.pdf)
- *Stefan Depeweg, José Miguel Hernández-Lobato, Finale Doshi-Velez, Steffen Udluft*
- `2017-06`
- [Uncertainty Estimation], [Reinforcement Learning]
```
Quite well-written and interesting paper. The toy problems illustrated in figure 2 and figure 3 are quite neat. I did however find it quite odd that they did not actually perform any active learning experiments here? Figure 4b is quite confusing with the "insert" for beta=0. I think it would have been better to show this entire figure somehow.
```

##### [20-01-31] [paper84]
- Modelling heterogeneous distributions with an Uncountable Mixture of Asymmetric Laplacians [[pdf]](https://arxiv.org/abs/1910.12288) [[code]](https://github.com/BBVA/UMAL) [[video]](https://vimeo.com/369179175) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Modelling%20heterogeneous%20distributions%20with%20an%20Uncountable%20Mixture%20of%20Asymmetric%20Laplacians.pdf)
- *Axel Brando, Jose A. Rodríguez-Serrano, Jordi Vitrià, Alberto Rubio*
- `NeurIPS 2019`
- [Uncertainty Estimation]
```
Quite well-written and interesting paper. The connection to quantile regression is quite neat, but in the end, their loss in equation 6 just corresponds to a latent variable model (with a uniform distribution for the latent variable tau) trained using straightforward Monte Carlo sampling. I am definitely not impressed with the experiments. They only consider very simple problems, y is always 1D, and they only compare with self-implemented baselines. The results are IMO not overly conclusive either, the single Laplacian model is e.g. better calibrated than their proposed method in Figure 3.
```

##### [20-01-24] [paper83]
- A Primal-Dual link between GANs and Autoencoders [[pdf]](http://papers.nips.cc/paper/8333-a-primal-dual-link-between-gans-and-autoencoders) [[poster]](https://drive.google.com/file/d/1ifPldBOeuSa2Iwh3ESVGmRJQKzs9XgPv/view) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Primal-Dual%20link%20between%20GANs%20and%20Autoencoders.pdf)
- *Hisham Husain, Richard Nock, Robert C. Williamson*
- `NeurIPS 2019`
- [Theoretical Properties of Deep Learning]
```
Somewhat interesting and well-written paper. Very theoretical paper compared to what I usually read. I must admit that I did not really understand that much.
```

##### [20-01-20] [paper82]
- A Connection Between Score Matching and Denoising Autoencoders [[pdf]](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport_1358.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Connection%20Between%20Score%20Matching%20and%20Denoising%20Autoencoders.pdf)
- *Pascal Vincent*
- `Neural Computation, 2011`
- [Energy-Based Models]
```
Quite well-written and interesting paper. The original paper for "denoising score matching", which it does a good job explaining. It also provides some improved understanding of score matching in general, and provides some quite interesting references for further reading.
```

##### [20-01-17] [paper81]
- Multiplicative Interactions and Where to Find Them [[pdf]](https://openreview.net/forum?id=rylnK6VtDH) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Multiplicative%20Interactions%20and%20Where%20to%20Find%20Them.pdf)
- *Siddhant M. Jayakumar, Jacob Menick, Wojciech M. Czarnecki, Jonathan Schwarz, Jack Rae, Simon Osindero, Yee Whye Teh, Tim Harley, Razvan Pascanu*
- `ICLR 2020`
- [Theoretical Properties of Deep Learning], [Sequence Modeling]
```
Well-written and somewhat interesting paper. I had some trouble properly understanding everything. I am however not overly impressed by the choice of experiments. The experiment in figure 2 seems somewhat biased in their favor? I think it would be a more fair comparison if the number of layers in the MLP was allowed to be increased (since this would increase its expressivity)?
```

##### [20-01-16] [paper80]
- Estimation of Non-Normalized Statistical Models by Score Matching [[pdf]](http://www.jmlr.org/papers/v6/hyvarinen05a.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Estimation%20of%20Non-Normalized%20Statistical%20Models%20by%20Score%20Matching.pdf)
- *Aapo Hyvärinen*
- `JMLR, 2005`
- [Energy-Based Models]
```
Interesting and very well-written paper. The original paper for score matching. Somewhat dated of course, but still interesting and very well-written. It provides a really neat introduction to score matching! I did not read section 3 super carefully, as the examples seemed quite dated.
```

##### [20-01-15] [paper79]
- Generative Modeling by Estimating Gradients of the Data Distribution [[pdf]](https://arxiv.org/abs/1907.05600) [[code]](https://github.com/ermongroup/ncsn) [[poster]](https://yang-song.github.io/papers/NeurIPS2019/ncsn-poster.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Generative%20Modeling%20by%20Estimating%20Gradients%20of%20the%20Data%20Distribution.pdf)
- *Yang Song, Stefano Ermon*
- `NeurIPS 2019`
- [Energy-Based Models], [Diffusion Models]
```
Well-written and quite interesting paper. The examples in section 3 are neat and quite pedagogical. I would probably need to read a couple of papers covering the basics of score matching, and then come back and read this paper again to fully appreciate it. Like they write, their training method could be used to train an EBM (by replacing their score network with the gradient of the energy in the EBM). This would then be just like "denoising score matching", but combining multiple noise levels in a combined objective? I suppose that their annealed Langevin approach could also be used to sample from an EBM. This does however seem very computationally expensive, as they run T=100 steps of Langevin dynamics for each of the L=10 noise levels?
```

##### [20-01-14] [paper78]
- Noise-Contrastive Estimation: A New Estimation Principle for Unnormalized Statistical Models [[pdf]](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Noise-contrastive%20estimation:%20A%20new%20estimation%20principle%20for%20unnormalized%20statistical%20models.pdf)
- *Michael Gutmann, Aapo Hyvärinen*
- `AISTATS 2010`
- [Energy-Based Models]
```
Well-written and interesting paper. The original paper for Noise Contrastive Estimation (NCE). Somewhat dated of course, but still interesting and well-written. Provides a quite neat introduction to NCE. They use a VERY simple problem to compare the performance of NCE to MLE with importance sampling, contrastive divergence (CD) and score-matching (and MLE, which gives the reference performance. MLE requires an analytical expression for the normalizing constant). CD has the best performance, but NCE is apparently more computationally efficient. I do not think such a simple problem say too much though. They then also apply NCE on a (by today's standards) very simple unsupervised image modeling problem. It seems to perform as expected.
```

##### [20-01-10] [paper77]
- Z-Forcing: Training Stochastic Recurrent Networks [[pdf]](https://arxiv.org/abs/1711.05411) [[code]](https://github.com/sordonia/zforcing) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Z-Forcing:%20Training%20Stochastic%20Recurrent%20Networks.pdf)
- *Anirudh Goyal, Alessandro Sordoni, Marc-Alexandre Côté, Nan Rosemary Ke, Yoshua Bengio*
- `NeurIPS 2017`
- [VAEs], [Sequence Modeling]
```
Quite interesting and well-written paper. Seems like Marco Fraccaro's thesis covers most of this paper, overall the proposed architecture is still quite similar to VRNN/SRNN both in design and performance. The auxiliary cost seems to improve performance quite consistently, but nothing revolutionary. It is not quite clear to me if the proposed architecture is more or less difficult / computationally expensive to train than SRNN.
```

##### [20-01-08] [paper76]
- Practical Deep Learning with Bayesian Principles [[pdf]](https://arxiv.org/abs/1906.02506) [[code]](https://github.com/team-approx-bayes/dl-with-bayes) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Practical%20Deep%20Learning%20with%20Bayesian%20Principles.pdf)
- *Kazuki Osawa, Siddharth Swaroop, Anirudh Jain, Runa Eschenhagen, Richard E. Turner, Rio Yokota, Mohammad Emtiyaz Khan*
- `NeurIPS 2019`
- [Uncertainty Estimation], [Variational Inference]
```
Interesting and quite well-written paper. To me, this mainly seems like a more practically useful alternative to Bayes by Backprop, scaling up variational inference to e.g. ResNet on ImageNet. The variational posterior approximation q is still just a diagonal Gaussian. I still do not fully understand natural-gradient variational inference. Only image classification is considered. It seems to perform ish as well as Adam in terms of accuracy (although it is 2-5 times slower to train), while quite consistently performing better in terms of calibration (ECE). The authors also compare with MC-dropout in terms of quality of the predictive probabilities, but these results are IMO not very conclusive.
```

##### [20-01-06] [paper75]
- Maximum Entropy Generators for Energy-Based Models [[pdf]](https://arxiv.org/abs/1901.08508) [[code]](https://github.com/ritheshkumar95/energy_based_generative_models) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Maximum%20Entropy%20Generators%20for%20Energy-Based%20Models.pdf)
- *Rithesh Kumar, Sherjil Ozair, Anirudh Goyal, Aaron Courville, Yoshua Bengio*
- `2019-01`
- [Energy-Based Models]
```
Quite well-written and interesting paper. The general idea, learning an energy-based model p_theta by drawing samples from an approximating distribution (that minimizes the KL divergence w.r.t p_theta) instead of generating approximate samples from p_theta using MCMC, is interesting and intuitively makes quite a lot of sense IMO. Since the paper was written prior to the recent work on MCMC-based learning (Learning Non-Convergent Non-Persistent Short-Run MCMC Toward Energy-Based Model, Implicit Generation and Generalization in Energy-Based Models, On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models), it is however difficult to know how well this method actually would stack up in practice.
```

#### Papers Read in 2019:

##### [19-12-22] [paper74]
- Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One [[pdf]](https://arxiv.org/abs/1912.03263) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Your%20Classifier%20is%20Secretly%20an%20Energy%20Based%20Model%20and%20You%20Should%20Treat%20it%20Like%20One.pdf)
- *Will Grathwohl, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, Kevin Swersky*
- `ICLR 2020`
- [Energy-Based Models]
```
Interesting and very well-written paper. I recommend actually going through the appendix as well, as it contains some interesting details. The idea to create an energy-based model for p(x) by marginalizing out y is really neat and makes a lot of sense in this classification setting (in which this corresponds to just summing the logits for all K classes). This EBM for p(x) is then trained using the MCMC-based ML learning method employed in other recent work. Simultaneously, a model for p(y|x) is also trained using the standard approach (softmax / cross entropy), thus training p(x, y) = p(y | x)*p(x). I am however not overly impressed/convinced by their experimental results. All experiments are conducted on relatively small and "toy-ish" datasets (CIFAR10, CIFAR100, SVHN etc), but they still seemed to have experienced A LOT of problems with training instability. Would be interesting to see results e.g. for semantic segmentation on Cityscapes (a more "real-world" task and dataset). Moreover, like the authors also point out themselves, training p(x) using SGLD-based sampling with L steps (they mainly use L=20 steps, but sometimes also have to restart training with L=40 to mitigate instability issues) basically makes training L times slower. I am just not sure if the empirically observed improvements are strong/significant enough to justify this computational overhead.
```

##### [19-12-20] [paper73]
- Noise Contrastive Estimation and Negative Sampling for Conditional Models: Consistency and Statistical Efficiency [[pdf]](https://arxiv.org/abs/1809.01812) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Noise%20Contrastive%20Estimation%20and%20Negative%20Sampling%20for%20Conditional%20Models:%20Consistency%20and%20Statistical%20Efficiency.pdf)
- *Zhuang Ma, Michael Collins*
- `EMNLP 2018`
- [Energy-Based Models], [NLP]
```
Interesting and quite well-written paper. Quite theoretical paper with a bunch of proofs. Interesting to see NCE applied specifically to supervised problems (modelling p(y | x)).
```

##### [19-12-20] [paper72]
- Flow Contrastive Estimation of Energy-Based Models [[pdf]](https://arxiv.org/abs/1912.00589) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Flow%20Contrastive%20Estimation%20of%20Energy-Based%20Models.pdf)
- *Ruiqi Gao, Erik Nijkamp, Diederik P. Kingma, Zhen Xu, Andrew M. Dai, Ying Nian Wu*
- `CVPR 2020`
- [Energy-Based Models], [Normalizing Flows]
```
Well-written and interesting paper. Provides a quite interesting comparison of EBMs and flow-based models in the introduction ("By choosing a flow model, one is making the assumption that the true data distribution is one that is in principle simple to sample from, and is computationally efficient to normalize."). Provides a pretty good introduction to Noise Contrastive Estimation (NCE). The proposed method is interesting and intuitively makes sense. The experimental results are not overly strong/decisive IMO, but that seems to be true for most papers in this area.
```

##### [19-12-19] [paper71]
- On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models [[pdf]](https://arxiv.org/abs/1903.12370) [[code]](https://github.com/point0bar1/ebm-anatomy)  [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20the%20Anatomy%20of%20MCMC-Based%20Maximum%20Likelihood%20Learning%20of%20Energy-Based%20Models.pdf)
- *Erik Nijkamp, Mitch Hill, Tian Han, Song-Chun Zhu, Ying Nian Wu*
- `AAAI 2020`
- [Energy-Based Models]
```
Well-written and very interesting paper, a recommended read! Provides a good review and categorization of previous papers, how they differ from each other etc. Provides a solid theoretical understanding of MCMC-based ML learning of EBMs, with quite a few really interesting (and seemingly useful) insights.
```

##### [19-12-15] [paper70]
- Implicit Generation and Generalization in Energy-Based Models [[pdf]](https://arxiv.org/abs/1903.08689) [[code]](https://github.com/openai/ebm_code_release) [[blog]](https://openai.com/blog/energy-based-models/) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Implicit%20Generation%20and%20Generalization%20in%20Energy-Based%20Models.pdf)
- *Yilun Du, Igor Mordatch*
- `NeurIPS 2019`
- [Energy-Based Models]
```
Interesting, but not overly well-written paper. Very similar to "Learning Non-Convergent Non-Persistent Short-Run MCMC Toward Energy-Based Model", but not as clearly written IMO. I personally find the experiments section somewhat unclear, but I'm also not too familiar with how generative image models usually are evaluated. It sounds like the training was quite unstable without the regularization described in section 3.3?
```

##### [19-12-14] [paper69]
- Learning Non-Convergent Non-Persistent Short-Run MCMC Toward Energy-Based Model [[pdf]](https://arxiv.org/abs/1904.09770) [[poster]](https://neurips.cc/Conferences/2019/Schedule?showEvent=13661) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20Non-Convergent%20Non-Persistent%20Short-Run%20MCMC%20Toward%20Energy-Based%20Model.pdf)
- *Erik Nijkamp, Mitch Hill, Song-Chun Zhu, Ying Nian Wu*
- `NeurIPS 2019`
- [Energy-Based Models]
```
Well-written and interesting paper. Seeing the non-convergent, short-run MCMC as a learned generator/flow model is a really neat and interesting idea. I find figure 9 in the appendix interesting. It is somewhat difficult for me to judge how impressive the experimental results are, I do not really know how strong the baselines are or how significant the improvements are. I found section 4 difficult to follow.
```

##### [19-12-13] [paper68]
- A Tutorial on Energy-Based Learning [[pdf]](http://yann.lecun.com/exdb/publis/orig/lecun-06.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Tutorial%20on%20Energy-Based%20Learning.pdf)
- *Yann LeCun, Sumit Chopra, Raia Hadsell, Marc Aurelio Ranzato, Fu Jie Huang*
- `2006-08`
- [Energy-Based Models]
```
Somewhat dated, but well-written and still quite interesting paper. A good introduction to enegy-based models (EBMs).
```

##### [19-11-29] [paper67]
- Dream to Control: Learning Behaviors by Latent Imagination [[pdf]](https://openreview.net/forum?id=S1lOTC4tDS) [[webpage]](https://dreamrl.github.io/) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Dream%20to%20Control%20Learning%20Behaviors%20by%20Latent%20Imagination.pdf)
- `ICLR 2020`
- [Reinforcement Learning]
```
Interesting and very well-written paper. A recommended read, even if you just want to gain an improved understanding of state-of-the-art RL in general and the PlaNet paper ("Learning Latent Dynamics for Planning from Pixels") in particular. Very similar to PlaNet, the difference is that they here learn an actor-critic model on-top of the learned dynamics, instead of doing planning using MPC. The improvement over PlaNet, in terms of experimental results, seems significant. Since they didn't actually use the latent overshooting in the PlaNet paper, I assume they don't use it here either?
```

##### [19-11-26] [paper66]
- Deep Latent Variable Models for Sequential Data [[pdf]](https://backend.orbit.dtu.dk/ws/portalfiles/portal/160548008/phd475_Fraccaro_M.pdf)  [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Latent%20Variable%20Models%20for%20Sequential%20Data.pdf)
- *Marco Fraccaro*
- `PhD Thesis, 2018`
- [Sequence Modeling], [VAEs]
```
Very well-written, VERY useful. VERY good general introduction to latent variable models, amortized variational inference, VAEs etc. VERY good introduction to various deep latent variable models for sequential data: deep state-space models, VAE-RNNs, VRNNs, SRNNs etc.
```

##### [19-11-22] [paper65]
- Learning Latent Dynamics for Planning from Pixels [[pdf]](https://arxiv.org/abs/1811.04551) [[code]](https://github.com/google-research/planet) [[blog]](https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20Latent%20Dynamics%20for%20Planning%20from%20Pixels.pdf)
- *Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson*
- `ICML 2019`
- [Reinforcement Learning]
```
Well-written and interesting paper! Very good introduction to the entire field of model-based RL I feel like. Seems quite odd to me that they spend an entire page on "Latent overshooting", but then don't actually use it for their RSSM model? It's not entirely clear to me how this approach differs from "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models" (PETS), apart from the fact that PETS actually has access to the state (so, they don't need to apply VAE stuff to construct a latent state representation). The provided code seems like it could be very useful. Is it easy to use? The model seems to train on just 1 GPU in just 1 day anyway, which is good.
```

##### [19-10-28] [paper64]
- Learning nonlinear state-space models using deep autoencoders [[pdf]](http://cse.lab.imtlucca.it/~bemporad/publications/papers/cdc18-autoencoders.pdf)  [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20nonlinear%20state-space%20models%20using%20deep%20autoencoders.pdf)
- *Daniele Masti, Alberto Bemporad*
- `CDC 2018`
- [Sequence Modeling]
```
Well-written and interesting paper. Really interesting approach actually, although somewhat confusing at first read since the method seems to involve quite a few different components. I would like to try to implement this myself and apply it to some simple synthetic example, I think that would significantly improve my understanding of the method and help me better judge its potential.
```

##### [19-10-18] [paper63]
- Improving Variational Inference with Inverse Autoregressive Flow [[pdf]](https://arxiv.org/abs/1606.04934) [[code]](https://github.com/openai/iaf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Improving%20Variational%20Inference%20with%20Inverse%20Autoregressive%20Flow.pdf)
- *Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling*
- `NeurIPS 2016`
- [Normalizing Flows]
```
Interesting and very well-written paper. Does a very good job introducing the general problem setup, normalizing flows, autoregressive models etc. Definitely a good introductory paper, it straightened out a few things I found confusing in Variational Inference with Normalizing Flows. The experimental results are however not very strong nor particularly extensive, IMO.
```

##### [19-10-11] [paper62]
- Variational Inference with Normalizing Flows [[pdf]](https://arxiv.org/abs/1505.05770) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Variational%20Inference%20with%20Normalizing%20Flows.pdf)
- *Danilo Jimenez Rezende, Shakir Mohamed*
- `ICML 2015`
- [Normalizing Flows]
```
Well-written and quite interesting paper. I was initially somewhat confused by this paper, as I was expecting it to deal with variational inference for approximate Bayesian inference. Seems like a good starting point for flow-based methods, I will continue reading-up on more recent/advanced techniques.
```

##### [19-10-04] [paper61]
- Trellis Networks for Sequence Modeling [[pdf]](https://arxiv.org/abs/1810.06682) [[code]](https://github.com/locuslab/trellisnet) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Trellis%20Networks%20for%20Sequence%20Modeling.pdf)
- *Shaojie Bai, J. Zico Kolter, Vladlen Koltun*
- `ICLR 2019`
- [Sequence Modeling]
```
Well-written and quite interesting paper. Interesting model, quite neat indeed how it can be seen as a bridge between RNNs and TCNs. The fact that they share weights across all network layers intuitively seems quite odd to me, but I guess it stems from the construction based on M-truncated RNNs? It is not obvious to me why they chose to use a gated activation function based on the LSTM cell, would using a "normal" activation function (e.g. ReLu) result in a significant drop in performance?
```

##### [19-07-11] [paper60]
- Part-A^2 Net: 3D Part-Aware and Aggregation Neural Network for Object Detection from Point Cloud [[pdf]](https://arxiv.org/abs/1907.03670) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Part-A%5E2%20Net:%203D%20Part-Aware%20and%20Aggregation%20Neural%20Network%20for%20Object%20Detection%20from%20Point%20Cloud.pdf)
- *Shaoshuai Shi, Zhe Wang, Xiaogang Wang, Hongsheng Li*
- `TPAMI, 2020`
- [3D Object Detection]
```
Interesting and quite well-written paper. Same main authors as for the PointRCNN paper. The idea to use the intra-object point locations provided by the ground truth 3D bboxes as extra supervision makes a lot of sense, clever! In this paper, the bin-based losses from PointRCNN are NOT used.
```

##### [19-07-10] [paper59]
- PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud [[pdf]](https://arxiv.org/abs/1812.04244) [[code]](https://github.com/sshaoshuai/PointRCNN) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/PointRCNN:%203D%20Object%20Proposal%20Generation%20and%20Detection%20from%20Point%20Cloud.pdf)
- *Shaoshuai Shi, Xiaogang Wang, Hongsheng Li*
- `CVPR 2019`
- [3D Object Detection]
```
Interesting and quite well-written paper. I think I like this approach to 3DOD. Directly processing the point cloud and generating proposals by classifying each point as foreground/background makes sense, is quite simple and seems to perform well. Their bin-based regression losses seem somewhat strange to me though.
```

##### [19-07-03] [paper58]
- Objects as Points [[pdf]](https://arxiv.org/abs/1904.07850) [[code]](https://github.com/xingyizhou/CenterNet) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Objects%20as%20Points.pdf)
- *Xingyi Zhou, Dequan Wang, Philipp Krähenbühl*
- `2019-04`
- [Object Detection]
```
Quite well-written and interesting paper. Multiple objects (of the same class) having the same (low-resolution) center point is apparently not very common in MS-COCO, but is that true also in real life in automotive applications? And in these cases, would only detecting one of these objects be a major issue? I do not really know, I find it somewhat difficult to even visualize cases where multiple objects would share center points. It is an interesting point that this method essentially corresponds to anchor-based one-stage detectors, but with just one shape-agnostic anchor. Perhaps having multiple anchors per location is not super important then?
```

##### [19-06-12] [paper57]
- ATOM: Accurate Tracking by Overlap Maximization [[pdf]](https://arxiv.org/abs/1811.07628) [[code]](https://github.com/visionml/pytracking) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/ATOM:%20Accurate%20Tracking%20by%20Overlap%20Maximization.pdf)
- *Martin Danelljan, Goutam Bhat, Fahad Shahbaz Khan, Michael Felsberg*
- `CVPR 2019`
- [Visual Tracking]
```
Well-written and interesting paper. They employ the idea of IoU-Net in order to perform target estimation and thus improve tracking accuracy. Interesting that this idea seems to work well also in this case. The paper also gives a quite comprehensive introduction to visual object tracking in general, making the proposed method relatively easy to understand also for someone new to the field.
```

##### [19-06-12] [paper56]
- Acquisition of Localization Confidence for Accurate Object Detection [[pdf]](https://arxiv.org/abs/1807.11590) [[code]](https://github.com/vacancy/PreciseRoIPooling) [[oral presentation]](https://youtu.be/SNCsXOFr_Ug) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Acquisition%20of%20Localization%20Confidence%20for%20Accurate%20Object%20Detection.pdf)
- *Borui Jiang, Ruixuan Luo, Jiayuan Mao, Tete Xiao, Yuning Jiang*
- `ECCV 2018`
- [Object Detection]
```
Interesting idea that intuitively makes a lot of sense, neat to see that it actually seems to work quite well. While the predicted IoU is a measure of "localization confidence", it is not an ideal measure of localization uncertainty. Having an estimated variance each for (x, y, w, h) would provide more information.
```

##### [19-06-05] [paper55]
- LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving [[pdf]](https://arxiv.org/abs/1903.08701) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/LaserNet:%20An%20Efficient%20Probabilistic%203D%20Object%20Detector%20for%20Autonomous%20Driving.pdf)
- *Gregory P. Meyer, Ankit Laddha, Eric Kee, Carlos Vallespi-Gonzalez, Carl K. Wellington*
- `CVPR 2019`
- [Uncertainty Estimation], [3D Object Detection]
```
Quite well-written and interesting paper. It was however quite difficult to fully grasp their proposed method. I struggled to understand some steps of their method, it is e.g. not completely clear to me why both mean shift clustering and adaptive NMS has to be performed. I find the used probabilistic model somewhat strange. They say that "our proposed method is the first to capture the uncertainty of a detection by modeling the distribution of bounding box corners", but actually they just predict a single variance value per bounding box (at least when K=1, which is the case for pedestrians and bikes)? Overall, the method seems rather complicated. It is probably not the streamlined and intuitive 3DOD architecture I have been looking for.
```

##### [19-05-29] [paper54]
- Attention Is All You Need [[pdf]](https://arxiv.org/abs/1706.03762) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Attention%20Is%20All%20You%20Need.pdf)
- *Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin*
- `NeurIPS 2017`
- [Transformers]
```
Quite well-written paper. The proposed architecture was explained in a quite clear way, even for someone who is not super familiar with the field. Not too related to my particular research, but still a quite interesting paper. I also think that the proposed architecture, the Transformer, has been extensively used in subsequent state-of-the-art models (I remember seeing it mentioned in a few different papers)? This paper is thus probably a good background read for those interested in language modeling, translation etc.
```

##### [19-04-05] [paper53]
- Stochastic Gradient Descent as Approximate Bayesian Inference [[pdf]](https://arxiv.org/abs/1704.04289) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Stochastic%20Gradient%20Descent%20as%20Approximate%20Bayesian%20Inference.pdf)
- *Stephan Mandt, Matthew D. Hoffman, David M. Blei*
- `JMLR, 2017`
- [Uncertainty Estimation], [Stochastic Gradient MCMC]
```
Very well-written and quite interesting paper. Good background material on SGD, SG-MCMC and so on. It is however a relatively long paper (26 pages). It makes intuitive sense that running SGD with a constant learning rate will result in a sequence of iterates which first move toward a local minimum and then "bounces around" its vicinity. And, that this "bouncing around" thus should correspond to samples from some kind of stationary distribution, which depends on the learning rate, batch size and other hyper parameters. Trying to find the hyper parameters which minimize the KL divergence between this stationary distribution and the true posterior then seems like a neat idea. I am however not quite sure how reasonable the made assumptions are in more complex real-world problems. I am thus not quite sure how useful the specific proposed methods/formulas actually are.
```

##### [19-03-29] [paper52]
- Generating High Fidelity Images with Subscale Pixel Networks and Multidimensional Upscaling [[pdf]](https://openreview.net/forum?id=HylzTiC5Km) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/GENERATING%20HIGH%20FIDELITY%20IMAGES%20WITH%20SUBSCALE%20PIXEL%20NETWORKS%20AND%20MULTIDIMENSIONAL%20UPSCALING.pdf)
- *Jacob Menick, Nal Kalchbrenner*
- `ICLR 2019`
- [Miscellaneous]
```
Quite interesting paper. I do however think that the proposed method could be more clearly explained, the paper actually left me somewhat confused (I am however not particularly familiar with this specific sub-field). For e.g. the images in Figure 5, it is not clear to me how these are actually generated? Do you take a random image from ImageNet, choose a random slice of this image and then generate the image by size- and depth-upscaling? For training, I guess that they (for each image in the dataset) choose a random image slice, condition on the previous true image slices (according to their ordering), predict/generate the next image slice and compare this with the ground truth to compute an unbiased estimator of the NLL loss. But what do they do during evaluation? I.e., how are the NLL scores in Table 1-3 computed? The experimental results do not seem overly impressive/convincing to me.
```

##### [19-03-15] [paper51]
- A recurrent neural network without chaos [[pdf]](https://arxiv.org/abs/1612.06212) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20recurrent%20neural%20network%20without%20chaos.pdf)
- *Thomas Laurent, James von Brecht*
- `ICLR 2017`
- [Sequence Modeling]
```
Quite well-written and somewhat interesting paper. I note that their LSTM implementation consistently outperformed their proposed CFN, albeit with a small margin. Would be interesting to know if this architecture has been studied further since the release of this paper, can it match LSTM performance also on more complicated tasks?
```

##### [19-03-11] [paper50]
- Auto-Encoding Variational Bayes [[pdf]](https://arxiv.org/abs/1312.6114)
- *Diederik P Kingma, Max Welling*
- `ICLR 2014`
- [VAEs]
```
Quite interesting paper.
```

##### [19-03-04] [paper49]
- Coupled Variational Bayes via Optimization Embedding [[pdf]](https://papers.nips.cc/paper/8177-coupled-variational-bayes-via-optimization-embedding.pdf) [[poster]](http://wyliu.com/papers/LiuNIPS18_CVB_poster.pdf) [[code]](https://github.com/Hanjun-Dai/cvb) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Coupled%20Variational%20Bayes%20via%20Optimization%20Embedding.pdf)
- *Bo Dai, Hanjun Dai, Niao He, Weiyang Liu, Zhen Liu, Jianshu Chen, Lin Xiao, Le Song*
- `NeurIPS 2018`
- [VAEs]
```
Somewhat well-written and interesting paper. It was however a quite heavy read. Also, I should definitely have done some more background reading on VAEs etc. (e.g., "Auto-encoding variational bayes", "Variational inference with normalizing flows", "Improved variational inference with inverse autoregressive flow") before trying to read this paper. I did not properly understand their proposed method, I found section 3 quite difficult to follow. Definitely not clear to me how one actually would implement this in practice. I am not sure how strong the experimental results actually are, it is not completely obvious to me that their proposed method actually outperforms the baselines in a significant way.
```

##### [19-03-01] [paper48]
- Language Models are Unsupervised Multitask Learners [[pdf]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) [[blog post]](https://blog.openai.com/better-language-models/) [[code]](https://github.com/openai/gpt-2) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Language%20Models%20are%20Unsupervised%20Multitask%20Learners.pdf)
- *Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever*
- `2019-02`
- [NLP]
```
Interesting and quite well-written paper. There are not that many technical details, one would probably have to read previous work for that. One probably needs to be somewhat familiar with NLP. Very impressive work from an infrastructure perspective. Just as context to their model with 1.5 billion parameters: a ResNet101 has 45 million parameters, which takes up 180 Mb when saved to disk. DeepLabV3 for semantic segmentation has roughly 75 million parameters. This has become a pretty hyped paper, and I agree that the work is impressive, but it still seems to me like their model is performing roughly as one would expect. It performs really well on general language modeling tasks, which is exactly what it was trained for (although it was not fine-tuned on the specific benchmark datasets), but performs rather poorly on translation and question-answering. The fact that the model has been able to learn some basic translation in this fully unsupervised setting is still quite impressive and interesting though.
```

##### [19-02-27] [paper47]
- Predictive Uncertainty Estimation via Prior Networks [[pdf]](https://arxiv.org/abs/1802.10501) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Predictive%20Uncertainty%20Estimation%20via%20Prior%20Networks.pdf)
- *Andrey Malinin, Mark Gales*
- `NeurIPS 2018`
- [Uncertainty Estimation]
```
Interesting and very well-written paper. It would be interesting to combine this approach with approximate Bayesian modeling (e.g. ensembling). They state in the very last sentence of the paper that their approach needs to be extended also to regression. How would you actually do that? It is not immediately obvious to me. Seems like a quite interesting problem. I would have liked to see a comparison with ensembling as well and not just MC-Dropout (ensembling usually performs better in my experience). Obtaining out-of-distribution samples to train on is probably not at all trivial actually. Yes, this could in theory be any unlabeled data, but how do you know what region of the input image space is covered by your training data? Also, I guess the model could still become over-confident if fed inputs which are far from both the in-distribution and out-of-distribution samples the model has seen during training? So, you really ought to estimate epistemic uncertainty using Bayesian modeling as well?
```

##### [19-02-25] [paper46]
- Evaluating model calibration in classification [[pdf]](https://arxiv.org/abs/1902.06977) [[code]](https://github.com/uu-sml/calibration) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Evaluating%20model%20calibration%20in%20classification.pdf)
- *Juozas Vaicenavicius, David Widmann, Carl Andersson, Fredrik Lindsten, Jacob Roll, Thomas B. Schön*
- `AISTATS 2019`
- [Uncertainty Estimation]
```
Well-written and interesting paper. It is however a quite theoretical paper, and I personally found it difficult to follow certain sections. It also uses notation that I am not fully familiar with. This work seems important, and I will try to keep it in mind in the future. It is however still not quite clear to me what one should do in practice to evaluate and compare calibration of large models on large-scale datasets in a more rigorous way. I will probably need to read the paper again.
```

##### [19-02-22] [paper45]
- Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks [[pdf]](https://arxiv.org/abs/1901.08584) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Fine-Grained%20Analysis%20of%20Optimization%20and%20Generalization%20for%20Overparameterized%20Two-Layer%20Neural%20Networks.pdf)
- *Sanjeev Arora, Simon S. Du, Wei Hu, Zhiyuan Li, Ruosong Wang*
- `ICML 2019`
- [Theoretical Properties of Deep Learning]
```
Somewhat interesting paper that is quite theoretical. I found it to be a rather heavy read, and I did not fully understand absolutely everything. I did not quite get why they fix the weights a_i of the second layer? They use gradient descent (GD) instead of SGD, could you obtain similar results also for SGD? I think that I probably did not understand the paper well enough to really be able to judge how significant/interesting the presented results actually are. How restrictive are their assumptions? In what way / to what extent could these results be of practical use in real-world applications? The reference section seems like a pretty neat resource for previous work on characterization of NN loss landscapes etc.
```

##### [19-02-17] [paper44]
- Visualizing the Loss Landscape of Neural Nets [[pdf]](https://arxiv.org/abs/1712.09913) [[code]](https://github.com/tomgoldstein/loss-landscape) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Visualizing%20the%20Loss%20Landscape%20of%20Neural%20Nets.pdf)
- *Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein*
- `NeurIPS 2018`
- [Miscellaneous]
```
Interesting and quite well-written paper. I think that the paper is a good introduction to methods for NN loss function visualization and previous work aiming to understand the corresponding optimization problem. They cite a number of papers which seem interesting, I will probably try and read a couple of those in the future. It would be interesting to apply their visualization method to some of my own problems, I will probably look more carefully at their code at some point. It is however not immediately obvious to me how to apply their "filter normalization" to e.g. an MLP network.
```

##### [19-02-14] [paper43]
-  A Simple Baseline for Bayesian Uncertainty in Deep Learning [[pdf]](https://arxiv.org/abs/1902.02476) [[code]](https://github.com/wjmaddox/swa_gaussian) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Simple%20Baseline%20for%20Bayesian%20Uncertainty%20in%20Deep%20Learning.pdf)
- *Wesley Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, Andrew Gordon Wilson*
- `NeurIPS 2019`
- [Uncertainty Estimation]
```
Quite well-written and interesting paper. I am not quite sure how I feel about the proposed method though. It seems somewhat odd to me to first fit a Gaussian approximation to samples from the SGD trajectory and then draw new samples from this Gaussian to use for Bayesian model averaging. Why not just directly use some of those SGD samples for model averaging instead? Am I missing something here? Also, in SG-MCMC we have to (essentially) add Gaussian noise to the SGD update and decay the learning rate to obtain samples from the true posterior in the infinite limit. I am thus somewhat confused by the theoretical analysis in this paper. I would have liked to see a comparison with basic ensembling. In section C.5 they write that SWAG usually performs somewhat worse than deep ensembles, but that this is OK since SWAG is much faster to train. "Thus SWAG will be particularly valuable when training time is limited, but inference time may not be.", when is this actually true? It makes intuitive sense that this method will generate parameter samples with some variance (instead of just a single point estimate) and thus also provide some kind of estimate of the model uncertainty. However, it is not really theoretically grounded in any significant way, at least not more than e.g. ensembling. The most interesting experiment for which they provide reliability diagrams is IMO CIFAR-10 --> STL-10. I note that even the best model still is quite significantly over-confident in this case. I really liked their version of reliability diagrams. Makes it easy to compare multiple methods in a single plot.
```

##### [19-02-13] [paper42]
-  Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning [[pdf]](https://arxiv.org/abs/1902.03932) [[code]](https://github.com/ruqizhang/csgmcmc) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Cyclical%20Stochastic%20Gradient%20MCMC%20for%20Bayesian%20Deep%20Learning.pdf)
- *Ruqi Zhang, Chunyuan Li, Jianyi Zhang, Changyou Chen, Andrew Gordon Wilson*
- `ICLR 2020`
- [Uncertainty Estimation], [Stochastic Gradient MCMC]
```
Well-written and VERY interesting paper (I did find a few typos though). Very interesting method. I have however done some experiments using their code, and I find that samples from the same cycle produce very similar predictions. Thus I am somewhat skeptical that the method actually is significantly better than snapshot-ensembling, or just regular ensembling for that matter. The results in table 3 do seem to suggest that there is something to gain from collecting more than just one sample per cycle though, right? I need to do more experiments and investigate this further. Must admit that I struggled to understand much of section 4, I am thus not really sure how impressive their theoretical results actually are.
```

##### [19-02-12] [paper41]
-  Bayesian Dark Knowledge [[pdf]](https://arxiv.org/abs/1506.04416) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Bayesian%20Dark%20Knowledge.pdf)
- *Anoop Korattikara, Vivek Rathod, Kevin Murphy, Max Welling*
- `NeurIPS 2015`
- [Uncertainty Estimation]
```
Well-written and quite interesting paper. The presented idea is something that has crossed my mind a couple of times, and it is indeed an attractive concept, but I have always ended up sort of rejecting the idea, since it sort of seems like it should not work. Take figure 2 for the toy 1d regression problem. It seems pretty obvious to me that one should be able to distill the SGLD predictive posterior into a Gaussian with input-dependent variance, but what about x values that lie outside of the shown interval? Will the model not become over-confident in that region anyway? To me it seems like this method basically only can be used to somewhat extend the region in which the model is appropriately confident. As we move away from the training data, I still think that the model will start to become over-confident at some point? However, perhaps this is still actually useful? Since the "ground truth labels" are generated by just running our SGLD model on any input, I guess we might be able to extend this region of appropriate confidence quite significantly?
```

##### [19-02-07] [paper40]
- Noisy Natural Gradient as Variational Inference [[pdf]](https://arxiv.org/abs/1712.02390) [[video]](https://youtu.be/bWItvHYqKl8) [[code]](https://github.com/pomonam/NoisyNaturalGradient) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Noisy%20Natural%20Gradient%20as%20Variational%20Inference.pdf)
- *Guodong Zhang, Shengyang Sun, David Duvenaud, Roger Grosse*
- `ICML 2018`
- [Uncertainty Estimation], [Variational Inference]
```
Well-written and somewhat interesting paper. Quite a heavy read for me as I am not particularly familiar with natural gradient optimization methods or K-FAC. I get that not being restricted to just fully-factorized Gaussian variational posteriors is something that could improve performance, but is it actually practical for properly large networks? They mention previous work on extending variational methods to non-fully-factorized posteriors, but I found them quite difficult to compare. It is not clear to me whether or not the presented method actually is a clear improvement (either in terms of performance or practicality).
```

##### [19-02-06] [paper39]
- Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks [[pdf]](https://arxiv.org/abs/1502.05336) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Probabilistic%20Backpropagation%20for%20Scalable%20Learning%20of%20Bayesian%20Neural%20Networks.pdf)
- *José Miguel Hernández-Lobato, Ryan P. Adams*
- `ICML 2015`
- [Uncertainty Estimation]
```
Quite well-written and interesting paper. I did however find it somewhat difficult to fully understand the presented method. I find it difficult to compare this method (PBP, which is an Assumed Density Filtering (ADF) method) with Variational Inference (VI) using a diagonal Gaussian as q. The authors seem to argue that their method is superior because it only employs one stochastic approximation (sub-sampling the data), whereas VI employs two (in VI one also approximates an expectation using Monte Carlo samples). In that case I guess that PBP should be very similar to Deterministic Variational Inference for Robust Bayesian Neural Networks? I guess it would be quite difficult to extend this method to CNNs?
```

##### [19-02-05] [paper38]
- Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models [[pdf]](https://arxiv.org/abs/1805.12114) [[poster]](https://kchua.github.io/misc/poster.pdf) [[video]](https://youtu.be/3d8ixUMSiL8) [[code]](https://github.com/kchua/handful-of-trials) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Deep%20Reinforcement%20Learning%20in%20a%20Handful%20of%20Trials%20using%20Probabilistic%20Dynamics%20Models.pdf)
- *Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine*
- `NeurIPS 2018`
- [Uncertainty Estimation], [Ensembling], [Reinforcement Learning]
```
General comments on paper quality:
Well-written and very interesting paper. It applies relatively common methods for uncertainty estimation (ensemble of probabilistic NNs) to an interesting problem in RL and shows promising results.


Paper overview:
The authors present a model-based RL algorithm called Probabilistic Ensembles with Trajectory Sampling (PETS), that (at least roughly) matches the asymptotic performance of SOTA model-free algorithms on four control tasks, while requiring significantly fewer samples (model-based algorithms generally have much better sample efficiency, but worse asymptotic performance than the best model-free algorithms).

They use an ensemble of probabilistic NNs (Probabilistic Ensemble, PE) to learn a probabilistic dynamics model, p_theta(s_t+1 | s_t, a_t), where s_t is the state and a_t is the taken action at time t.

A probabilistic NN outputs the parameters of a probability distribution, in this case by outputting the mean, mu(s_t, a_t), and diagonal covariance matrix, SIGMA(s_t, a_t), of a Gaussian, enabling estimation of aleatoric (data) uncertainty. To also estimate epistemic (model) uncertainty, they train an ensemble of B probabilistic NNs.

The B ensemble models are trained on separate (but overlapping) datasets: for each ensemble model, a dataset is created by drawing N examples with replacement from the original dataset D (which also contains N examples).

The B ensemble models are then used in the trajectory sampling step, where P state particles s_t_p are propagated forward in time by iteratively sampling s_t+1_p ~ p_theta_b(s_t+1_p | s_t_p, a_t)). I.e., each ensemble model outputs a distribution, and they sample particles from these B distributions. This results in P trajectory samples, s_t:t+T_p (which we hope approximate the true distribution over trajectories s_t:t+T). The authors used P=20, B=5 in all their experiments.

Based on these P state trajectory samples, MPC is finally used to compute the next action a_t.


Comments:
Interesting method. Should be possible to benchmark various uncertainty estimation techniques using their setup, just like they compare probabilistic/deterministic ensembles and probabilistic networks. I found it quite interesting that a single probabilistic network (at least somewhat) outperformed a deterministic ensemble (perhaps this would change with a larger ensemble size though?).

Do you actually need to perform the bootstrap procedure when training the ensemble, or would you get the same performance by simply training all B models on the same dataset D?

I struggle somewhat to understand their method for lower/upper bounding the outputted variance during testing (appendix A.1). Do you actually need this? Also, I do not quite get the lines of code. Are max_logvar and min_logvar variables?
```

##### [19-01-28] [paper37]
- Practical Variational Inference for Neural Networks [[pdf]](https://www.cs.toronto.edu/~graves/nips_2011.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Practical%20Variational%20Inference%20for%20Neural%20Networks.pdf)
- *Alex Graves*
- `NeurIPS 2011`
- [Uncertainty Estimation], [Variational Inference]
```
Reasonably well-written and somewhat interesting paper. The paper seems quite dated compared to "Weight Uncertainty in Neural Networks". I also found it significantly more difficult to read and understand than "Weight Uncertainty in Neural Networks". One can probably just skip reading this paper, for an introduction to variational methods applied to neural networks it is better to read "Weight Uncertainty in Neural Networks" instead.
```

##### [19-01-27] [paper36]
- Weight Uncertainty in Neural Networks [[pdf]](https://arxiv.org/abs/1505.05424) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Weight%20Uncertainty%20in%20Neural%20Networks.pdf)
- *Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra*
- `ICML 2015`
- [Uncertainty Estimation], [Variational Inference]
```
General comments on paper quality:
Well-written and interesting paper. I am not particularly familiar with variational methods, but still found the paper quite easy to read and understand.


Comments:
Seems like a good starting point for learning about variational methods applied to neural networks. The theory is presented in a clear way. The presented method also seems fairly straightforward to implement.

They mainly reference "Keeping Neural Networks Simple by Minimizing the Description Length of the Weights" and "Practical Variational Inference for Neural Networks" as relevant previous work.

In equation (2), one would have to run the model on the data for multiple weight samples? Seems quite computationally expensive?

Using a diagonal Gaussian for the variational posterior, I wonder how much of an approximation that actually is? Is the true posterior e.g. very likely to be multi-modal?

The MNIST models are only evaluated in terms of accuracy. The regression experiment is quite neat (good to see that the uncertainty increases away from the training data), but they provide very little details. I find it difficult to draw any real conclusions from the Bandits experiment.
```

##### [19-01-26] [paper35]
- Learning Weight Uncertainty with Stochastic Gradient MCMC for Shape Classification [[pdf]](http://people.duke.edu/~cl319/doc/papers/dbnn_shape_cvpr.pdf)  [[poster]](https://zhegan27.github.io/Papers/dbnn_shape_poster.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Learning%20Weight%20Uncertainty%20with%20Stochastic%20Gradient%20MCMC%20for%20Shape%20Classification.pdf)
- *Chunyuan Li, Andrew Stevens, Changyou Chen, Yunchen Pu, Zhe Gan, Lawrence Carin*
- `CVPR 2016`
- [Uncertainty Estimation], [Stochastic Gradient MCMC]
```
Quite interesting and well-written paper. Quite an easy read compared to many other SG-MCMC papers. I find it weird that they only evaluate their models in terms of accuracy. It is of course a good thing that SG-MCMC methods seem to compare favorably with optimization approaches, but I would have been more interested in an evaluation of some kind of uncertainty estimate (e.g. the sample variance). The studied applications are not overly interesting, the paper seems somewhat dated in that regard.
```

##### [19-01-25] [paper34]
- Meta-Learning For Stochastic Gradient MCMC [[pdf]](https://openreview.net/forum?id=HkeoOo09YX) [[code]](https://github.com/WenboGong/MetaSGMCMC) [[slides]](http://yingzhenli.net/home/pdf/uai_udl_meta_sampler.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Meta-Learning%20For%20Stochastic%20Gradient%20MCMC.pdf)
- *Wenbo Gong, Yingzhen Li, José Miguel Hernández-Lobato*
- `ICLR 2019`
- [Uncertainty Estimation], [Stochastic Gradient MCMC]
```
Fairly interesting paper.
```

##### [19-01-25] [paper33]
-  A Complete Recipe for Stochastic Gradient MCMC [[pdf]](https://arxiv.org/abs/1506.04696) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Complete%20Recipe%20for%20Stochastic%20Gradient%20MCMC.pdf)
- *Yi-An Ma, Tianqi Chen, Emily B. Fox*
- `NeurIPS 2015`
- [Uncertainty Estimation], [Stochastic Gradient MCMC]
```
General comments on paper quality:
Well-written and very interesting paper. After reading the papers on SGLD and SGHMC, this paper ties the theory together and provides a general framework for SG-MCMC.


Paper overview:
The authors present a general framework and recipe for constructing MCMC and SG-MCMC samplers based on continuous Markov processes. The framework entails specifying a stochastic differential equation (SDE) by two matrices, D(z) (positive semi-definite) and Q(z) (skew-symmetric). Here, z = (theta, r), where theta are the model parameters and r are auxiliary variables (r corresponds to the momentum variables in Hamiltonian MC).

Importantly, the presented framework is complete, meaning that all continuous Markov processes with the target distribution as its stationary distribution (i.e., all continuous Markov processes which provide samples from the target distribution) correspond to a specific choice of the matrices D(z), Q(z). Every choice of D(z), Q(z) also specifies a continuous Markov process with the target distribution as its stationary distribution.

The authors show how previous SG-MCMC methods (including SGLD, SGRLD and SGHMC) can be casted to their framework, i.e., what their corresponding D(z), Q(z) are.

They also introduce a new SG-MCMC method, named SGRHMC, by wisely choosing D(z), Q(z).

Finally, they conduct two simple experiments which seem to suggest (at least somewhat) improved performance of SGRHMC compared to previous methods (SGLD, SGRLD, SGHMC).


Comments:
How does one construct \hat{B_t}, the estimate of V(theta_t) (the noise of the stochastic gradient)?

If one (by computational reasons) only can afford evaluating, say, 10 samples to estimate various expectations, what 10 samples should one pick? The final 10 samples, or will those be heavily correlated? Pick the final sample (at time t = T) and then also the samples at time t=T-k*100 (k = 1, 2, ..., 9)? (when should one start collecting samples and with what frequency should they then be collected?)

If one were to train an ensemble of models using SG-MCMC and pick the final sample of each model, how would these samples be distributed?

If the posterior distribution is a simple bowl, like in the right part of figure 2, what will the path of samples actually look like compared to the steps taken by SGD? In figure 2, I guess that gSHRHMC will eventually converge to roughly the bottom of the bowl? So if one were to only collect samples from this later stage of traversing, the samples would actually NOT be (at least approximately) distributed according to the posterior?
```

##### [19-01-24] [paper32]
- Tutorial: Introduction to Stochastic Gradient Markov Chain Monte Carlo Methods [[pdf]](https://cse.buffalo.edu/~changyou/PDF/sgmcmc_intro_without_video.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Tutorial:%20Introduction%20to%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo%20Methods.pdf)
- *Changyou Chen*
- `2016-08`
- [Uncertainty Estimation], [Stochastic Gradient MCMC]
```
Quite interesting.
```

##### [19-01-24] [paper31]
- An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling [[pdf]](https://arxiv.org/abs/1803.01271) [[code]](https://github.com/locuslab/TCN) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/An%20Empirical%20Evaluation%20of%20Generic%20Convolutional%20and%20Recurrent%20Networks%20for%20Sequence%20Modeling.pdf)
- *Shaojie Bai, J. Zico Kolter, Vladlen Koltun*
- `2018-04`
- [Sequence Modeling]
```
General comments on paper quality:
Well-written and interesting paper.


Paper overview:
"We conduct a systematic evaluation of generic convolutional and recurrent architectures for sequence modeling. The models are evaluated across a broad range of standard tasks that are commonly used to benchmark recurrent networks. Our results indicate that a simple convolutional architecture outperforms canonical recurrent networks such as LSTMs across a diverse range of tasks and datasets, while demonstrating longer effective memory. We conclude that the common association between sequence modeling and recurrent networks should be reconsidered, and convolutional networks should be regarded as a natural starting point for sequence modeling tasks."

The authors introduce a quite straightforward CNN designed for sequence modeling, named Temporal Convolutional Network (TCN). They only consider the setting where the output at time t, y_t, is predicted using only the previously observed inputs, x_0, ..., x_t. TCN thus employs causal convolution (zero pad with kernel_size-1 at the start of the input sequence).

To achieve a long effective history size (i.e., that the prediction for y_t should be able to utilize inputs observed much earlier in the input sequence), they use residual blocks (to be able to train deep networks, the effective history scales linearly with increased depth) and dilated convolutions.

They compare TCN with basic LSTM, GRU and vanilla-RNN models on a variety of sequence modeling tasks (which include polyphonic music modeling, word- and character-level language modeling as well as synthetic "stress test" tasks), and find that TCN generally outperforms the other models. The authors do however note that TCN is outperformed by more specialized RNN architectures on a couple of the tasks.

They specifically study the effective history/memory size of the models using the Copy Memory task (Input sequences are digits of length 10 + T + 10, the first 10 are random digits in {1, ..., 8}, the last 11 are 9:s and all the rest are 0:s. The goal is to generate an output of the same length that is 0 everywhere, except the last 10 digits which should be a copy of the first 10 digits in the input sequence), and find that TCN significantly outperforms the LSTM and GRU models (which is a quite interesting result, IMO).


Comments:
Interesting paper that challenges the viewpoint of RNN models being the default starting point for sequence modeling tasks. The presented TCN architecture is quite straightforward, and I do think it makes sense that CNNs might be a very competitive alternative for sequence modeling.
```

##### [19-01-23] [paper30]
- Stochastic Gradient Hamiltonian Monte Carlo [[pdf]](https://arxiv.org/abs/1402.4102) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Stochastic%20Gradient%20Hamiltonian%20Monte%20Carlo.pdf)
- *Tianqi Chen, Emily B. Fox, Carlos Guestrin*
- `ICML 2014`
- [Uncertainty Estimation], [Stochastic Gradient MCMC]
```
Interesting paper.
```

##### [19-01-23] [paper29]
- Bayesian Learning via Stochastic Gradient Langevin Dynamics [[pdf]](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Bayesian%20Learning%20via%20Stochastic%20Gradient%20Langevin%20Dynamics.pdf)
- *Max Welling, Yee Whye Teh*
- `ICML 2011`
- [Uncertainty Estimation], [Stochastic Gradient MCMC]
```
Interesting paper.
```

##### [19-01-17] [paper28]
- How Does Batch Normalization Help Optimization? [[pdf]](https://arxiv.org/abs/1805.11604) [[poster]](http://people.csail.mit.edu/tsipras/batchnorm_poster.pdf) [[video]](https://youtu.be/ZOabsYbmBRM) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/How%20Does%20Batch%20Normalization%20Help%20Optimization%3F.pdf)
- *Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, Aleksander Madry*
- `NeurIPS 2018`
- [Theoretical Properties of Deep Learning]
```
General comments on paper quality:
Quite well-written and interesting paper. A recommended read if you have ever been given the explanation that batch normalization works because it reduces the internal covariate shift.


Paper overview:
The abstract summarizes the paper very well:

"Batch Normalization (BatchNorm) is a widely adopted technique that enables faster and more stable training of deep neural networks (DNNs). Despite its pervasiveness, the exact reasons for BatchNorm's effectiveness are still poorly understood. The popular belief is that this effectiveness stems from controlling the change of the layers' input distributions during training to reduce the so-called "internal covariate shift". In this work, we demonstrate that such distributional stability of layer inputs has little to do with the success of BatchNorm. Instead, we uncover a more fundamental impact of BatchNorm on the training process: it makes the optimization landscape significantly smoother. This smoothness induces a more predictive and stable behavior of the gradients, allowing for faster training."
"In this work, we have investigated the roots of BatchNorm’s effectiveness as a technique for training deep neural networks. We find that the widely believed connection between the performance of BatchNorm and the internal covariate shift is tenuous, at best. In particular, we demonstrate that existence of internal covariate shift, at least when viewed from the - generally adopted – distributional stability perspective, is not a good predictor of training performance. Also, we show that, from an optimization viewpoint, BatchNorm might not be even reducing that shift."

"Instead, we identify a key effect that BatchNorm has on the training process: it reparametrizes the underlying optimization problem to make it more stable (in the sense of loss Lipschitzness) and smooth (in the sense of “effective” β-smoothness of the loss). This implies that the gradients used in training are more predictive and well-behaved, which enables faster and more effective optimization."

"We also show that this smoothing effect is not unique to BatchNorm. In fact, several other natural normalization strategies have similar impact and result in a comparable performance gain."


Comments:
It has never been clear to me how/why batch normalization works, I even had to remove all BatchNorm layers in an architecture once to get the model to train properly. Thus, I definitely appreciate this type of investigation.

It is somewhat unclear to me how general the presented theoretical results actually are.
```

##### [19-01-09] [paper27]
- Relaxed Softmax: Efficient Confidence Auto-Calibration for Safe Pedestrian Detection [[pdf]](https://openreview.net/forum?id=S1lG7aTnqQ) [[poster]](http://www.robots.ox.ac.uk/~vgg/publications/2018/Neumann18c/poster.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Relaxed%20Softmax:%20Efficient%20Confidence%20Auto-Calibration%20for%20Safe%20Pedestrian%20Detection.pdf)
- *Lukas Neumann, Andrew Zisserman, Andrea Vedaldi*
- `NeurIPS Workshops 2018`
- [Uncertainty Estimation]
```
General comments on paper quality:
Reasonably well-written paper. I'm not entirely convinced of the usefulness of the proposed method.


Paper overview:
The authors study pedestrian object detectors and evaluate the quality of their confidence score estimates using reliability diagrams (and related metrics, e.g. Expected Calibration Error). They find that a SOTA detector produces significantly over-confident predictions, i.e., that the obtained accuracy for predictions in any given confidence score interval is lower than the associated confidence score.

To mitigate this problem, they propose a simple modification of the standard softmax layer, called relaxed softmax. Instead of having the network output logits z in R^{K} and computing the probability vector softmax(z) (also in R^{K}), the network instead outputs (z, alpha), where alpha > 0, and the probability vector is computed as softmax(alpha*z). Relaxed softmax is inspired by temperature scaling.

For quantitative evaluation, they use Expected Calibration Error, Average Calibration Error (like ECE, but each bin is assigned an equal weight) and Maximum Calibration Error. They compare softmax, softmax + temperature scaling, softmax + linear scaling (similar to temperature scaling), relaxed softmax and relaxed softmax + linear scaling. They utilize two datasets: Caltech and NightOwls (models are trained on the train sets, linear scaling and temperature scaling are tuned on the val sets, and all metrics are computed on the test sets).

On Caltech, relaxed softmax + linear scaling gives the best calibration metrics, ahead of softmax + temperature scaling. On NightOwl, relaxed softmax is the winner, just ahead of relaxed softmax + linear scaling. The relaxed softmax methods also achieve somewhat worse miss rate metrics (13.26% versus 10.17% on Caltech, I'm not sure how significant of a decrease that actually is).


Comments:
Quite interesting paper, but I am not fully convinced. For example, I find it odd that relaxed softmax beats softmax + temperature scaling on NightOwl but not on Caltech.

It might be that I am missing something, but I also struggle to understand some of their explanations and arguments, e.g. in the final paragraph of section 3.2. I am not quite sure if the network-outputted scale alpha actually does what they say it does.
```

#### Papers Read in 2018:

##### [18-12-12] [paper26]
- Neural Ordinary Differential Equations [[pdf]](https://arxiv.org/abs/1806.07366) [[code]](https://github.com/rtqichen/torchdiffeq) [[slides]](https://www.cs.toronto.edu/~duvenaud/talks/ode-talk-google.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Ordinary%20Differential%20Equations.pdf)
- *Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud*
- `NeurIPS 2018`
- [Neural ODEs]
```
General comments on paper quality:
Reasonably well-written but very interesting paper. The examples could have been more thoroughly explained, it feels like the authors probably struggled to stay within the page limit.


Paper overview:
The authors introduce a new family of deep neural networks by using black-box ODE solvers as a model component.

Instead of specifying a discrete sequence of hidden layers by: h_{t+1} = h_t + f(h_t, theta_t), where f is some neural network architecture, they interpret these iterative updates as an Euler discretization/approximation of the corresponding continuous dynamics and directly specify this ODE: dh(t)/dt = f(h(t), theta). To compute gradients, they use the adjoint method, which essentially entails solving a second, augmented ODE backwards in time.

For example, if you remove the final ReLU layer, do not perform any down-sampling and have the same number of input- and output channels, a residual block in ResNet specifies a transformation precisely of the kind h_{t+1} = h_t + f(h_t, theta_t). Instead of stacking a number of these residual blocks, one could thus directly parameterize the corresponding ODE, dh(t)/dt = f(h(t), theta), and use an ODE solver to obtain h(T) as your output.

When empirically evaluating this approach on MNIST, they find that the model using one ODEBlock instead of 6 ResBlocks achieves almost identical test accuracy, while using fewer parameters (parameters for just one block instead of six).

The authors also apply their approach to density estimation and time-series modeling, but I chose to focus mainly on the ResNet example.


Comments:
Very interesting idea. Would be interesting to attempt to implement e.g. a ResNet101 using this approach. I guess one could try and keep the down-sampling layers, but otherwise replace layer1 - layer4 with one ODEBlock each?

It is however not at all clear to me how well this approach would scale to large-scale problems. Would it become too slow? Or would you lose accuracy/performance? Or would the training perhaps even become unstable? Definitely an interesting idea, but much more empirical evaluation is needed.
```

##### [18-12-06] [paper25]
- Evaluating Bayesian Deep Learning Methods for Semantic Segmentation [[pdf]](https://arxiv.org/abs/1811.12709) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Evaluating%20Bayesian%20Deep%20Learning%20Methods%20for%20Semantic%20Segmentation.pdf)
- *Jishnu Mukhoti, Yarin Gal*
- `2018-11`
- [Uncertainty Estimation]
```
General comments on paper quality:
Quite well-written and interesting paper. Not particularly heavy to read.


Paper overview:
The authors present three metrics designed to evaluate and compare different Bayesian DL methods for the task of semantic segmentation (i.e., models which also output pixel-wise uncertainty estimates).

They train DeepLabv3+ using both MC-dropout (apply dropout also during inference, run multiple forward passes to obtain M samples, compute the sample mean and variance) and Concrete dropout ("a modification of the MC-dropout method where the network tunes the dropout rates as part of the optimization process"), and then compare these two methods on Cityscapes using their suggested metrics. They thus hope to provide quantitative benchmarks which can be used for future comparisons.

Their three presented metrics are (higher values are better):

p(accurate | certain) = n_a_c/(n_a_c + n_i_c).
p(uncertain | inaccurate) = n_i_u/(n_i_c + n_i_u).
PAvPU = (n_a_c + n_i_u)/(n_a_c + n_a_u + n_i_c + n_i_u)
Where:
n_a_c: number of accurate and certain patches.
n_a_u: number of accurate and uncertain patches.
n_i_c: number of inaccurate and certain patches.
n_i_u: number of inaccurate and uncertain patches.
They compute these metrics on patches of size 4x4 pixels (I didn't quite get their reasoning for why this makes more sense than studying this pixel-wise), where a patch is defined as accurate if more than 50% of the pixels in the patch are correctly classified. Similarly, a patch is defined as uncertain if its average pixel-wise uncertainty is above a given threshold. They set this uncertainty threshold to the average uncertainty value on Cityscapes val (which I found somewhat strange, since they then also do all of their evaluations on Cityscapes val).

They found that MC-dropout outperformed concrete dropout with respect to all three metrics.


Comments:
The intended contribution is great, we definitely need to define metrics which can be used to benchmark different uncertainty estimating models. I am not 100% happy with the presentation of their suggested metrics though:
What would be the ideal values for these metrics?
Can the metrics be ranked in terms of importance?
What is the "best" value for the uncertainty threshold, and how should it be chosen?
```

##### [18-12-05] [paper24]
- On Calibration of Modern Neural Networks [[pdf]](https://arxiv.org/abs/1706.04599) [[code]](https://github.com/gpleiss/temperature_scaling) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/On%20Calibration%20of%20Modern%20Neural%20Networks.pdf)
- *Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger*
- `ICML 2017`
- [Uncertainty Estimation]
```
General comments on paper quality:
Well-written and quite interesting paper. Not a particularly heavy read.


Paper overview:
The authors study the concept of confidence calibration in classification models. A model is said to be calibrated (or well-calibrated) if the confidence score corresponding to its prediction actually is representative of the true correctness likelihood, i.e., if the model outputs a confidence score of 0.75 for 1000 examples, roughly 750 of those should be correctly classified by the model.

They empirically find that modern neural networks (e.g. ResNets) usually are poorly calibrated, outputting overconfident predictions (whereas old networks, e.g. LeNet, usually were well-calibrated). They e.g. find that while increasing network depth or width often improves the classification accuracy, it also has a negative effect on model calibration.

The authors then describe a few post-processing methods designed to improve model calibration, all of which require a validation set (you fix the network weights, learn to modify the outputted confidence score based on the validation set and then hope for the model to stay well-calibrated also on the test set). They also introduce a very simple calibration method, named temperature scaling, in which you learn (optimize on the validation set) a single scalar T, which is used to scale the logits z outputted by the model (new_conf_score = max_k{softmax(z/T)_k}).

They compare these calibration methods on six different image classification datasets (e.g. ImageNet and CIFAR100) and 4 document classification datasets, using different CNNs (e.g. ResNet and DenseNet). Surprising to the authors, they find the simple temperature scaling method to achieve the best overall performance (most well-calibrated confidence scores), often having a significant positive effect on calibration.


Comments:
Quite interesting paper, and the effectiveness of temperature scaling is actually quite impressive. Since the authors assume that the train, val and test sets are drawn from the same data distribution, it would however be interesting to evaluate the calibration also on out-of-distribution data. If we train a model on MNIST train, use temperature scaling on MNIST val (and thus obtain quite well-calibrated confidence scores on MNIST test), would it then also be more well-calibrated on e.g. notMNIST?
```

##### [18-11-29] [paper23]
-  Evidential Deep Learning to Quantify Classification Uncertainty [[pdf]](https://arxiv.org/abs/1806.01768) [[poster]](https://muratsensoy.github.io/NIPS18_EDL_poster.pdf) [[code example]](https://muratsensoy.github.io/uncertainty.html) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Evidential%20Deep%20Learning%20to%20Quantify%20Classification%20Uncertainty.pdf)
- *Murat Sensoy, Lance Kaplan, Melih Kandemir*
- `NeurIPS 2018`
- [Uncertainty Estimation]
```
General comments on paper quality:
Well-written and very interesting paper. I had to read it a couple of times to really start understanding everything though.


Paper overview:
The authors present a classification model in which they replace the standard softmax output layer with an output layer that outputs parameters of a Dirichlet distribution (resource1, resource2). I.e., they assume a Dirichlet output distribution, similar to Gast and Roth. The authors interpret the behavior of this predictor from an evidential reasoning / subjective logic perspective (two terms which I am unfamiliar with): "By placing a Dirichlet distribution on the class probabilities, we treat predictions of a neural net as subjective opinions and learn the function that collects the evidence leading to these opinions by a deterministic neural net from data".

Instead of outputting just a point estimate of the class probabilities (the softmax scores), the network thus outputs the parameters of a distribution over the class probabilities (similar to how a network can output the parameters of a Gaussian instead of just a point estimate in the regression case).

The only difference in network architecture is that they replace the softmax layer with a ReLU layer (to get non-negative values) to obtain e_1, ..., e_K (K is the number of classes). The parameters alpha_1, ..., alpha_K of the Dirichlet distribution is then set to alpha_i = e_i + 1 (which means alpha_i >= 1, i.e., they are restricting the set of Dirichlet distributions their model can predict? They are setting a maximum value for the variance?). Given this, the Dirichlet mean, alpha/S (S = sum(alpha_i)), is taken as the class probabilities estimate.

The authors present three different possible loss functions (which are all different from the one used by Gast and Roth?), which all involve averaging over the predicted Dirichlet pdf, and choose one based on their empirical findings. They claim that this chosen loss corresponds to learned loss attenuation (but I struggle somewhat to actually see why that is so). They then also add a KL divergence term to this loss, penalizing divergence from a uniform distribution (which strikes me as somewhat ad hoc?).

They train their model on MNIST (digits) and then evaluate on notMNIST (letters), expecting a large proportion of predictions to have maximum entropy (maximum uncertainty). They also do a similar experiment using CIFAR10, training on images of the first 5 classes and then evaluating on images of the remaining 5 classes.

They compare their model with e.g. MC-dropout and Deep Ensembles, and find that their model achieves similar test set performance (on MNIST / the first 5 classes of CIFAR10), while producing significantly better uncertainty estimates (their model outputs maximum entropy predictions more frequently when being fed images of unseen classes).

They also do an experiment with adversarial inputs, finding that their model has a similar drop in prediction accuracy, while being less confident in its predictions (which is a good thing, you don't want the model to become overconfident, i.e., misclassify inputs but still being confident in its predictions).


Comments:
Really interesting paper. It also made me go back and read Gast and Roth much more carefully.

Just like I think it makes a lot of sense to assume a Gaussian/Laplacian output distribution in the regression case, it does intuitively seem reasonable to assume a Dirichlet output distribution in classification. As indicated by the fact that the authors and Gast and Roth choose different loss functions (and also estimate the Dirichlet parameters in different ways), it is however not at all as clear to me what actually should the best / most natural way of doing this.
```

##### [18-11-22] [paper22]
-  A Probabilistic U-Net for Segmentation of Ambiguous Images [[pdf]](https://arxiv.org/abs/1806.05034) [[code]](https://github.com/SimonKohl/probabilistic_unet) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/A%20Probabilistic%20U-Net%20for%20Segmentation%20of%20Ambiguous%20Images.pdf)
- *Simon A. A. Kohl, Bernardino Romera-Paredes, Clemens Meyer, Jeffrey De Fauw, Joseph R. Ledsam, Klaus H. Maier-Hein, S. M. Ali Eslami, Danilo Jimenez Rezende, Olaf Ronneberger*
- `NeurIPS 2018`
- [Uncertainty Estimation]
```
General comments on paper quality:
Well-written and interesting paper.


Paper overview:
The authors present a model for semantic image segmentation designed to produce a set of diverse but plausible/consistent segmentations. The model is intended for domains in which it is difficult to assign a unique ground truth segmentation mask to any given input image (there is some ambiguity), as in e.g. some medical imaging applications where it is standard practice to have a group of graders label each image (and the graders might produce quite different looking segmentations).

The model is a combination of a standard U-Net and a Conditional Variational Auto-Encoder (CVAE). The CVAE encodes the input image into a low-dimensional latent space (they use N=6), and a random sample from this space is injected into the final block of the U-Net to produce a corresponding segmentation mask.

A prior-net (essentially the U-Net encoder) takes the image as input and outputs mu_prior, sigma_prior (both in R^N) for a Gaussian (SIGMA = diag(sigma)) in the latent space. To sample z from the latent space, they simply sample from this Gaussian. The sample z in R^N is broadcasted to an N-channel feature map and concatenated with the last feature map of the U-Net. This new feature map is then processed by three 1x1 convolutions to map it to the number of classes (the feature map has the same spatial size as the input image). To output M segmentation masks, one thus only has to sample z_1, ..., z_M and follow the above procedure (prior-net and the U-Net only have to be evaluated once).

During training, each image - label pair (X, Y) is taken as input to a posterior-net (essentially the U-Net encoder) which outputs mu_post, sigma_post for a Gaussian in the latent space. A sample z is drawn from this Gaussian, the corresponding segmentation mask is produced (same procedure as above) and then compared with the label Y using the standard cross-entropy loss. To this loss we also add a KL loss term which penalizes differences between the posterior-net and prior-net Gaussians (the prior net only takes the image X as input).

They evaluate their method on two different datasets:

LIDC-IDRI, a medical dataset for lung abnormalities segmentation in which each lung CT scan has been independently labeled by four experts (each image has four corresponding ground truth labels, i.e., there is inherent ambiguity).
A modified version of Cityscapes. They here manually inject ambiguity into the dataset by e.g. changing the class "sidewalk" to "sidewalk2" (a class created by the authors) with some probability. They do this for 5 original classes, and thus end up with 2^5=32 possible modes with probabilities ranging from 10.9% to 0.5% (a given input image could thus correspond to any of these 32 modes, they have manually created some ambiguity).
Since they are not interested in comparing a deterministic prediction with a unique ground truth, but rather in comparing distributions of segmentations, they use a non-standard performance metric across their experiments.

They compare their method to number of baselines (a U-Net with MC-dropout, an ensemble of U-Nets, a U-Net with multiple heads) (same number of forward passes / ensemble networks / heads as the number of samples from the latent space), and basically find that their method outperforms all of them with respect to their performance metric.


Comments:
Interesting paper.

The method is mainly intended for the medical imaging domain, where I definitely can see why you might want a model that outputs a set of plausible segmentations which then can be further analyzed by medical professionals. For autonomous driving however, I guess what you ultimately want is just the most likely prediction and, crucially, the corresponding uncertainty of this prediction. Can we extract this from the proposed method?

If we take the mean of the prior-net Gaussian as our sample, I guess we would produce the most likely segmentation? And I guess sigma of this Gaussian is then a measure of the corresponding uncertainty? How about uncertainty estimates for the pixel-wise predictions, could you extract those as well somehow? Just treat the M maps of predicted class scores like you would when using MC-dropout or ensembles (e.g. take the sample variance as a measure of the epistemic uncertainty), or could you get this directly from the Gaussian?

Also, would this method not at all work if you only have one ground truth label per image?
```

##### [18-11-22] [paper21]
- When Recurrent Models Don't Need To Be Recurrent (a.k.a. Stable Recurrent Models) [[pdf]](https://arxiv.org/abs/1805.10369) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/When%20Recurrent%20Models%20Don%E2%80%99t%20Need%20To%20Be%20Recurrent.pdf)
- *John Miller, Moritz Hardt*
- `ICLR 2019`
- [Sequence Modeling]
```
General comments on paper quality:
Reasonably well-written and somewhat interesting paper. I do not think it is intended for publication in any conference/journal.


Paper overview:
The authors present a number of theorems, proving that stable Recurrent Neural Networks (RNNs) can be well-approximated by standard feed-forward networks. Moreover, if gradient descent succeeds in training a stable RNN, it will also succeed in training the corresponding feed-forward model. I.e., stable recurrent models do not actually need to be recurrent (which can be very convenient, since feed-forward models usually are easier and less computationally expensive to train).

For a vanilla RNN, h_t = rho(Wh_{t-1} + Ux_t), stability corresponds to requiring ||W|| < 1/L_rho (L_rho is the Lipschitz constant of rho).

You construct the corresponding feed-forward model approximation by moving over the input sequence with a sliding window of length k, producing an output every time the window advances by one step (auto-regressive model).

They show that stable recurrent models effectively do not have a long-term memory, and relate this to the concept of vanishing gradients (if the gradients of a recurrent model quickly vanish, then it could be well-approximated by a feed-forward model, even though the model was not explicitly constrained to be stable?).


Comments:
I find it difficult to judge how significant the presented results actually are, I think you need to be more familiar with the latest research within RNNs to properly appreciate the paper.
```

##### [18-11-16] [paper20]
- Uncertainty Estimates and Multi-Hypotheses Networks for Optical Flow [[pdf]](https://arxiv.org/abs/1802.07095) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20Estimates%20and%20Multi-Hypotheses%20Networks%20for%20Optical%20Flow_.pdf)
- *Eddy Ilg, Özgün Çiçek, Silvio Galesso, Aaron Klein, Osama Makansi, Frank Hutter, Thomas Brox*
- `ECCV 2018`
- [Uncertainty Estimation], [Ensembling]
```
General comments on paper quality:
Well written and very interesting paper. A recommended read.


Paper overview:
The authors study uncertainty estimation in the domain of optical flow estimation, which is a pixel-wise regression problem.

They compare multiple previously suggested uncertainty estimation methods:

Empirical ensembles (each model only outputs a single point estimate), using both MC-dropout, bootstrap ensembles and snapshot ensembles.
Predictive models (the model outputs the parameters (e.g. mean and variance) of an assumed output distribution, trained using the corresponding negative log likelihood).
Predictive ensembles (ensemble of predictive models), using both MC-dropout, bootstrap ensembles and snapshot ensembles.
(A bootstrap ensemble is created by independently training M models on different (partially overlapping) subsets of the training data, whereas a snapshot ensemble is essentially created by saving checkpoints during the training process of a single model).
For an empirical ensemble, the empirical mean and variance are taken as the mean and variance estimates (mu = (1/M)sum(mu_i), sigma^2 = (1/M)sum( (mu_i - mu)^2) )).

A predictive model directly outputs estimates of the mean and variance (the authors assume a Laplacian output distribution, which corresponds to an L1 loss).

For a predictive ensemble, the outputted mean and variance estimates are combined into the final estimates (mu = (1/M)sum(mu_i), sigma^2 = (1/M)sum( (mu_i - mu)^2) ) + (1/M)sum(sigma^2_i) )

Since all of the above methods require multiple forward passes to be computed during inference (obviously affecting the inference speed), the authors also propose a multi-headed predictive model architecture that yields multiple hypotheses (each hypothesis corresponds to an estimated mean and variance). They here use a loss that only penalizes the best hypothesis (the one which is closest to the ground truth), which encourages the model to yield a diverse set of hypotheses in case of doubt. A second network is then trained to optimally merge the hypotheses into a final mean and variance estimate. It is however not clear to me how this merging network actually is trained (did I miss something in the paper?).

They train their models on the FlyingChairs and FlyingThings3D datasets, and mostly evaluate on the Sintel dataset.

For all ensembles, they use M=8 networks (and M=8 hypotheses in the multi-headed model) (they find that more networks generally results in better performance, but are practically limited in terms of computation and memory).

To assess the quality of the obtained uncertainty estimates, they use sparsification plots as the main evaluation metric. In such a plot, you plot the average error as a function of the fraction of removed pixels, where the pixels are removed in order, starting with the pixels corresponding to the largest estimated uncertainty. This average error should thus monotonically decrease (as we remove more and more pixels) if the estimated uncertainty actually is a good representation of the true uncertainty/error. The obtained curve is compared to the "Oracle" sparsification curve, obtained by removing pixels according to the true error.

In their results, they find e.g. that:

Predictive ensembles have better performance than empirical ensembles. Even a single predictive model they claim to yield better uncertainty estimates than any empirical ensemble.
Predictive ensembles only yield slightly better performance than a single predictive model.
MC-dropout consistently performs worse than both bootstrap and snapshot ensembles (note that they also use just M=8 forward passes in MC-dropout).
The multi-headed predictive model yields the best performance among all models.


Comments:
Very interesting paper with a thorough comparison of various uncertainty estimation techniques.

I am however not completely convinced by the evaluation. I get how the sparsification plots measure the quality of the relative uncertainties (i.e., whether or not the model has learned what pixels are the most/least uncertain), but what about the absolute magnitude? Could it be that a model consistently under/over-estimates the uncertainties? If we were to create prediction intervals based on the estimated uncertainties, would they then have valid coverage?

The multi-headed network is definitely an interesting idea, I did not expect it to yield the best performance.
```

##### [18-11-15] [paper19]
- Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV) [[pdf]](https://arxiv.org/abs/1711.11279) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Interpretability%20Beyond%20Feature%20Attribution:%20Quantitative%20Testing%20with%20Concept%20Activation%20Vectors%20(TCAV)_.pdf)
- *Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, Rory Sayres*
- `ICML 2018`
- [Miscellaneous]
```
General comments on paper quality:
Quite well-written and fairly interesting paper, the authors do a pretty good job of giving an intuitive explanation of the proposed methods.


Paper overview:
The authors introduce a new method for interpreting the results of trained neural network classification models, in terms of user-defined high-level concepts.

They introduce Concept Activation Vectors (CAVs), which are vectors in the direction of the activations of a concept's set of example images, and the technique called Testing with CAVs (TCAV), that uses directional derivatives to quantify how important a user-defined concept is to a given classification result (e.g., how important the concept "striped" is to the classification of a given image as "Zebra").

To obtain a CAV for a given concept (e.g. "striped"), they collect a set of example images representing that concept (e.g. a set of images of various striped shirts and so on), train a linear classifier to distinguish between the activations produced by these concept example images and random images, and choose as a CAV the vector which is orthogonal to the classification boundary of this linear classifier (i.e., the CAV points in the direction of the activations of the concept example images).

By combining CAVs with directional derivatives, one can measure the sensitivity of a model's predictions to changes in the input towards the direction of a given concept. TCAV uses this to compute a model's conceptual sensitivity across entire classes of inputs, by computing the fraction of images for a given class which were positively influenced by a given concept (the directional derivatives were positive).

They qualitatively evaluate their method by e.g. sorting images of a given class based on how similar they are to various concepts (e.g. finding the images of "necktie" which are most similar to the concept "model woman"), and comparing the TCAV scores of different concepts for a given classification (e.g. finding that "red" is more important than "blue" for the classification of "fire engine").


Comments:
Quite interesting method which I suppose could be useful for some use-cases. I do however find it quite difficult to say how well the proposed method actually works, i.e., it is quite difficult to know whether the successful examples in the paper are just cherry-picked, or if the method consistently makes sense.
```

##### [18-11-12] [paper18]
- Large-Scale Visual Active Learning with Deep Probabilistic Ensembles [[pdf]](https://arxiv.org/abs/1811.03575) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Large-Scale%20Visual%20Active%20Learning%20with%20Deep%20Probabilistic%20Ensembles_.pdf)
- *Kashyap Chitta, Jose M. Alvarez, Adam Lesnikowski*
- `2018-11`
- [Uncertainty Estimation], [Ensembling]
```
General comments on paper quality:
Quite well-written and very interesting paper. Reasonably easy to read.


Paper overview:
The authors introduce Deep Probabilistic Ensembles (DPEs), a technique that utilizes a regularized ensemble to perform approximate variational inference in Bayesian Neural Networks (BNNs). They experimentally evaluate their method on the task of active learning for classification (CIFAR-10, CIFAR-100, ImageNet) and semantic segmentation (BDD100k), and somewhat outperform similar methods.

In variational inference, one restricts the problem to a family of distributions over the network weights w, q(w) ~ D. One then tries to optimize for the member of this family D that is closest to the true posterior distribution in terms of KL divergence. This optimization problem is equivalent to the maximization of the Evidence Lower Bound (ELBO), which contains expectations over all possible q(w) ~ D.

In this paper, the authors approximate these expectations by using an ensemble of E networks, which results in a loss function containing the standard cross-entropy term together with a regularization term OMEGA over the joint set of all parameters in the ensemble. Thus, the proposed method is an approximation of variational inference.

They chose Gaussians for both the prior p(w) and q(w), assume mutual independence between the network weights and can then compute the regularization term OMEGA by independently computing it for each network weight w_i (each network in the ensemble has a value for this weight w_i) using equation 9, and then summing this up over all network weights. I.e., for each weight w_i, you compute mu_q, sigma_q as the sample mean and variance across the E ensemble networks and then use equation 9. Equation 9 will penalize variances much larger than that of the prior (so that the ensemble members do not diverge completely from each other), penalize variances smaller than that of the prior (promoting diversity) and keep the mean close to that of the prior.

Note that the E ensemble networks have to be trained jointly, meaning that the memory requirement scales linearly with E.

They experienced some difficulties when trying to train an ensemble of just E=2, 3 networks, as the regularization term caused instability and divergence of the loss. This problem was mitigated by setting E >= 4, and they ended up using E=8 for all of their experiments (beyond E=8 they observed diminishing returns).

In the experiments, they e.g. compare DPEs to using an ensemble trained using standard L2 regularization on all four datasets. DPEs were found to consistently outperform the standard ensemble, but the performance gain is not very big.


Comments:
Definitely an interesting method. Nice to see more than just an intuitive argument for why ensembling seems to provide reasonable uncertainty estimates, even though the derivation contains multiple approximations (variational inference approximation, approximation of the expectations).

I'm not sure how significant the performance gain compared to standard ensembling actually is though, I would like to see more comparisons also outside of the active learning domain. Would also be interesting to compare with Bayesian ensembling.
```

##### [18-11-08] [paper17]
- The Lottery Ticket Hypothesis: Finding Small, Trainable Neural Networks [[pdf]](https://arxiv.org/abs/1803.03635) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/The%20Lottery%20Ticket%20Hypothesis:%20Finding%20Small%2C%20Trainable%20Neural%20Networks_.pdf)
- *Jonathan Frankle, Michael Carbin*
- `ICLR 2019`
- [Theoretical Properties of Deep Learning]
```
General comments on paper quality:
Well-written and very interesting paper. Not particularly heavy to read.


Paper overview:
Aiming to help and explain why it empirically seems easier to train large networks than small ones, the authors articulate the lottery ticket hypothesis: any large network that trains successfully contains a smaller subnetwork that, when initialized with the same initial parameter values again (i.e., the parameter values they had before the original training began), can be trained in isolation to match (or surpass) the accuracy of the original network, while converging in at most the same number of iterations. The authors call these subnetworks winning tickets.

When randomly re-initializing the parameters or randomly modifying the connections of winning tickets, they are no longer capable of matching the performance of the original network. Neither structure nor initialization alone is thus responsible for a winning ticket's success.

The authors find that a standard pruning technique (which essentially entails removing weights in increasing order of their magnitude (remove small-magnitude weights first)) can be used to automatically uncover such winning tickets.

They also extend their hypothesis into the conjecture (which they do not empirically test) that large networks are easier to train because, when randomly initialized, they contain more combinations of subnetworks and thus more potential winning tickets.

They find that winning tickets usually contain just 20% (or less) of the original network parameters. They find winning tickets for both fully-connected, convolutional and residual networks (MNIST, CIFAR10, CIFAR10).


Comments:
I actually found this paper a lot more interesting than I initially expected just from reading the title. Easy-to-grasp concept which still might help to improve our understanding of neural networks.
```

##### [18-10-26] [paper16]
- Towards Safe Autonomous Driving: Capture Uncertainty in the Deep Neural Network For Lidar 3D Vehicle Detection [[pdf]](https://arxiv.org/abs/1804.05132) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Towards%20Safe%20Autonomous%20Driving:%20Capture%20Uncertainty%20in%20the%20Deep%20Neural%20Network%20For%20Lidar%203D%20Vehicle%20Detection_.pdf)
- *Di Feng, Lars Rosenbaum, Klaus Dietmayer*
- `ITSC 2018`
- [Uncertainty Estimation], [3D Object Detection]
```
General comments on paper quality:
Fairly well-written paper. Interesting method.


Paper overview:
The authors present a two-stage, LiDAR-only (2D bird's eye view as input) model for 3D object detection (trained only on the Car class on KITTI) which attempts to model both epistemic uncertainty (model uncertainty) and aleatoric uncertainty (input-dependent data noise).

The aleatoric uncertainty is modeled for the regression task in the conventional way, i.e, by assuming a Gaussian distribution over the model output (the model outputs estimates for both the mean and variance) and minimizing the associated negative log-likelihood (actually, they seem to use an L1 or smoothL1 norm instead of L2). Aleatoric uncertainty is only modeled in the output layer, not in the RPN.

To estimate the epistemic uncertainty, they use MC-dropout in the three fully-connected layers in the refinement head (not in the RPN). They use N=40 forward passes. For classification, the softmax scores are averaged and the computed entropy and mutual information is used as epistemic uncertainty estimates. For regression, the sample variances are used.

Before the RPN, they use a ResNet-18(?) to extract a feature map. The model input has a spatial size of 1000x600 pixels. They use a discretization resolution of 0.1 m.

They train on 9918 training examples and evaluate on 2010 testing examples, both from the KITTI raw dataset. They evaluate their model by computing the F1 score for different IoU thresholds (0.1 to 0.8). I thus find it difficult to compare their 3DOD performance with models on the KITTI leaderboard.

They find that modeling the aleatoric uncertainty consistently improves 3DOD performance (compared to a fully deterministic baseline version), whereas modeling epistemic uncertainty actually degrades performance somewhat.

When the authors compute the average epistemic uncertainty for each predicted 3Dbbox, they find that predictions with large IoU values (good predictions, predictions which are close to a ground truth 3Dbbox) generally has smaller associated uncertainty than predictions with small IoU values (poor predictions).

For the aleatoric uncertainty, they did NOT see this relationship. Instead, they found that the uncertainty generally increased as the distance to the predicted 3Dbbox increased (which makes intuitive sense, distant objects may have just a few associated LiDAR points).


Comments:
First paper to apply the uncertainty estimation methods of Kendall and Gal to the task of 3DOD, which the authors definitely deserve credit for. Aleatoric uncertainty estimation adds negligible compute and improves performance, whereas the N=40 forward passes needed probably makes the epistemic uncertainty estimation method difficult to deploy in real-time applications.
```

##### [18-10-25] [paper15]
- Bayesian Convolutional Neural Networks with Many Channels are Gaussian Processes [[pdf]](https://arxiv.org/abs/1810.05148) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Bayesian%20Convolutional%20Neural%20Networks%20with%20Many%20Channels%20are%20Gaussian%20Processes_.pdf)
- *Roman Novak, Lechao Xiao, Jaehoon Lee, Yasaman Bahri, Daniel A. Abolafia, Jeffrey Pennington, Jascha Sohl-Dickstein*
- `ICLR 2019`
- [Theoretical Properties of Deep Learning]
```
General comments on paper quality:
Fairly well-written but rather heavy paper to read, I probably don't have the necessary background to fully appreciate its contributions.


Paper overview:
There is a known correspondence between fully Bayesian (with Gaussian prior), infinitely wide, fully connected, deep feedforward neural networks and Gaussian processes. The authors here derive an analogous correspondence between fully Bayesian (with Gaussian prior), deep CNNs with infinitely many channels and Gaussian Processes.

They also propose a method to find this corresponding GP (which has a 0 mean function), by estimating its kernel (which might be computationally impractical to compute analytically, or might have an unknown analytic form) using a Monte Carlo method. They show that this estimated kernel converges to the analytic kernel in probability as the number of channels.


Comments:
I always find these kind of papers interesting as they try to improve our understanding of the theoretical properties of neural networks. However, it's still not particularly clear to me what this GP correspondence in the infinite limit of fully Bayesian networks actually tells us about finite, non-Bayesian networks.
```

##### [18-10-19] [paper14]
- Uncertainty in Neural Networks: Bayesian Ensembling [[pdf]](https://arxiv.org/abs/1810.05546) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Uncertainty%20in%20Neural%20Networks:%20Bayesian%20Ensembling_.pdf)
- *Tim Pearce, Mohamed Zaki, Alexandra Brintrup, Andy Neel*
- `AISTATS 2020`
- [Uncertainty Estimation], [Ensembling]
```
General comments on paper quality:
Well-written and interesting paper. Compares different ensembling techniques and techniques for approximate Bayesian inference in neural networks.


Paper overview:
The authors present randomized anchored MAP sampling, anchored ensembles for short, a somewhat modified ensembling process aimed at estimating predictive uncertainty (epistemic/model uncertainty + aleatoric uncertainty) in neural networks.

They independently train M networks (they set M to just 5-10) on the entire training dataset, just like e.g. Lakshminarayanan et al.. The key difference is that they regularize the network parameters about values drawn from a prior distribution (instead of about 0 as in standard L2 regularization). In practice, each network in the ensemble is regularized, or "anchored", about its initialization parameter values (which are drawn from some prior Gaussian).

This procedure is motivated/inspired by the Bayesian inference method called randomized MAP sampling, which (roughly speaking) exploits the fact that adding a regularization term to the standard MLE loss function results in a MAP parameter estimate. Injecting noise into this loss (to the regularization term) and sampling repeatedly (i.e., ensembling) produces a distribution of MAP estimates which roughly corresponds to the true parameter posterior distribution.

What this injected noise should look like is however difficult to find in complex cases like NNs. What the authors do is that they study the special case of single-layer, wide NNs and claim that ensembling here will approximate the true posterior if the parameters theta of each network are L2-regularized about theta_0 ~ N(mu_prior, sigma_prior).

The authors do NOT let the network output an estimate of the aleatoric uncertainty like Lakshminarayanan et al. do, instead they assume a constant aleatoric uncertainty estimate sigma_epsilon^2. They then use their ensemble to compute the predictive mean y_hat = (1/M)*sum(y_hat_i) and variance sigma_y^2 = (1/(M-1))*sum((y_hat_i - y_hat)^2) + sigma_epsilon^2.

They evaluate their method on various regression tasks (no classification whatsoever) using single-layer NNs. For 1D regression they visually compare to the analytical GP ("gold standard" but not scalable), Hamiltonian MC ("gold standard" but not scalable), a Variational Inference method (scalable) and MC dropout (scalable). They find that their method here outperforms the other scalable methods.

They also compared their results on a regression benchmark with the method by Lakshminarayanan et al. ("normal" ensemble of the same size) which is the current state-of-the-art (i.e., it outperforms e.g. MC dropout). They find that their method outperforms the other in datasets with low aleatoric noise/uncertainty, whereas Lakshminarayanan et al. is better on datasets with high aleatoric noise. The authors say this is because Lakshminarayanan et al. explicitly tries to model the aleatoric noise (network outputs both mean and variance).


Comments:
Interesting method which is actually very similar to the very simple method by Lakshminarayanan et al.. The only real difference is that you add this regularization about the random initial parameter values, which shouldn't be too difficult to implement in e.g. PyTorch?

The fact that you just seem to need 5-10 networks in the ensemble (Lakshminarayanan et al. also used 5-10 networks) also makes the method somewhat practically useful even in real-time applications.

Would be very interesting to add this regularization to the Lakshminarayanan et al. method, apply to 3DOD and compare to Lakshminarayanan et al., MC dropout etc.

Perhaps the Bayesian motivation used in this paper doesn't really hold for large CNNs (and even so, how do you know that a Gaussian is the "correct" prior for Bayesian inference in this case?), but it still makes some intuitive sense that adding this random regularization could increase the ensemble diversity and thus improve the uncertainty estimate.
```

##### [18-10-18] [paper13]
- Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles [[pdf]](https://arxiv.org/abs/1612.01474) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Simple%20and%20Scalable%20Predictive%20Uncertainty%20Estimation%20using%20Deep%20Ensembles_.pdf)
- *Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell*
- `NeurIPS 2017`
- [Uncertainty Estimation], [Ensembling]
```
General comments on paper quality:
Well-written and interesting paper. The proposed method is simple and also very clearly explained.


Paper overview:
The authors present a simple, non-Bayesian method for estimating predictive uncertainty (epistemic/model uncertainty + aleatoric uncertainty) in neural networks, based on the concept of ensembling.

For regression, they train an ensemble of M networks which output both a mean and a variance (y given x is assumed to be Gaussian) by minimizing the corresponding negative log-likelihood (similar to how Kendall and Gal model aleatoric uncertainty).

For classification, they train an ensemble of M networks with the standard softmax output layer.

Each of the M networks in an ensemble is independently trained on the entire training dataset, using random initialization of the network weights and random shuffling of the training data. Typically, they set M=5.

For classification, the predicted probabilities are averaged over the M networks during inference.

For regression, the final mean is computed as the average of the means outputted by the individual networks (mu_final = (1/M)*sum(mu_i)), whereas the final variance sigma_final^2 = (1/M)*sum(mu_i^2 - mu_final^2) + (1/M)*sum(sigma_i^2).

The authors experimentally evaluate their method on various regression (1D toy problem as well as real-world datasets) and classification tasks (MNIST, SVHN and ImageNet). They find that their method generally outperforms (or a least matches the performance of) related methods, specifically MC-dropout.

They also find that when training a classification model on a certain dataset and then evaluating the model on a separate dataset containing unseen classes, their model generally outputs larger uncertainty (larger entropy) than the corresponding MC-dropout model (which is a good thing, we don't want our model to produce over-confident predictions, we want the model to "know what it doesn't know").


Comments:
Conceptually very simple, yet interesting method. The key drawback of using ensembling, especially in real-time applications, is of course that is requires running M networks to obtain a single prediction. However, if a relatively small ensemble size as e.g. M=5 is enough to obtain high-quality uncertainty estimates, it shouldn't really be impossible to still achieve real-time inference speed (50 Hz single-model is needed to obtain 10 Hz in that case). I do actually find this method quite interesting.
```

##### [18-10-18] [paper12]
- Reliable Uncertainty Estimates in Deep Neural Networks using Noise Contrastive Priors [[pdf]](https://arxiv.org/abs/1807.09289) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Reliable%20Uncertainty%20Estimates%20in%20Deep%20Neural%20Networks%20using%20Noise%20Contrastive%20Priors_.pdf)
- *Danijar Hafner, Dustin Tran, Alex Irpan, Timothy Lillicrap, James Davidson*
- `ICML Workshops 2018`
- [Uncertainty Estimation]
```
General comments on paper quality:
Well-written paper, the proposed method is fairly clearly explained.


Paper overview:
The authors present a method called Noise Contrastive Priors (NCPs). The key idea is to train a model to output high epistemic/model uncertainty for data points which lie outside of the training distribution (out-of-distribution data, OOD data). To do this, NCPs add noise to some of the inputs during training and, for these noisy inputs, try to minimize the KL divergence to a wide prior distribution.

NCPs do NOT try to add noise only to the subset of the training data which actually lie close to the boundary of the training data distribution, but instead add noise to any input data. Empirically, the authors saw no significant difference in performance between these two approaches.

The authors apply NCPs both to a small Bayesian Neural Network and to an OOD classifier model, and experimentally evaluate their models on regression tasks.

In the OOD classifier model, the network is trained to classify noise-upped inputs as OOD and non-modified inputs as "in distribution". During testing, whenever the model classifies an input as OOD, the model outputs the parameters of a fixed wide Gaussian N(0, sigma_y) instead of the mean and variance outputted by the neural network.


Comments:
Somewhat interesting method, although I must say it seems quite ad hoc. Not sure if just adding random noise to the LiDAR point clouds would be enough to simulate OOD data in the case of LiDAR-only 3DOD. Could be worth trying though I suppose.
```

##### [18-10-05] [paper11]
- VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection [[pdf]](https://arxiv.org/abs/1711.06396) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/VoxelNet:%20End-to-End%20Learning%20for%20Point%20Cloud%20Based%203D%20Object%20Detection_.pdf)
- *Yin Zhou, Oncel Tuzel*
- `CVPR 2018`
- [3D Object Detection]
```
General comments on paper quality:
Well-written and interesting paper, the proposed architecture is clearly described.


Paper overview:
The authors present a single-stage, LiDAR-only model for 3D object detection (3DOD) of a single object class (e.g. vehicles), and evaluate the model on the KITTI dataset.

They discretize 3D space into a 3D voxel grid of resolution 0.2x0.2x0.4 m, and the LiDAR points are grouped according to which voxel they reside in. If a voxel contains more than T (T = 35 for vehicles, T = 45 for pedestrians/cyclists) LiDAR points, T points are randomly sampled to represent that voxel. For each non-empty voxel, the corresponding LiDAR points are then fed through "Voxel Feature Encoding layers" (basically a PointNet) to extract a learned feature vector of dimension C (C = 128). The result of this process is thus a (sparse) CxD'xH'xW' (128x10x400x352 for vehicles) feature map representing the original LiDAR point cloud.

This 3D feature map is processed by 3D convolutions and flattened in order to obtain a 128xH'xW' 2D feature map, which is fed as input to a conventional (2D convolutions) region proposal network (RPN).

The RPN outputs a Kx(H'/2)x(W'/2) confidence/objectness score map, and a (7K)x(H'/2)x(W'/2) regression map, which contains the 7 regression outputs (x, y, z, h, w, l, theta) for each of the K anchors at each grid cell position.

The authors use K=2 anchors per grid cell, with theta = 0 deg or 90 deg, both with (w, h, l) set to the mean size from the training data and z set to -1. The grid is thus defined in a 2D bird's eye view, but still corresponds to anchor 3D bounding boxes on the plane z=-1 (which intuitively should work well in the application of autonomous driving where most cars lie on the same ground plane).

Anchors are assigned to either being positive, negative or don't-care based on their bird's eye view IoU with the ground truth bounding boxes. The confidence/classification loss is computed for both positive and negative anchors, while the regression loss is computed only for positive anchors.

The authors train three separate networks for detection of vehicles, pedestrians and cyclists, respectively.

They compare their networks' performance with other models on both the KITTI 3D and KITTI bird's eye view leaderboards, and find that VoxelNet outperforms all LiDAR-only methods across the board. Compared to PIXOR (which only submitted results for bird's eye view), VoxelNet has better performance but is significantly slower in inference. The VoxelNet inference time is dominated by the 3D convolutions.


Comments:
Interesting 3DOD model! Using (what is basically) a PointNet to extract feature vectors from groups of LiDAR points and thus obtain a learned 3D feature map is really rather clever, all though using 3D convolutions has a clear negative effect on inference time.

The remaining parts of the architecture seems well-designed (more so than e.g. PIXOR), and thus VoxelNet seems like a reasonable candidate to extend in future work on LiDAR-only 3DOD. Could you e.g. extend the architecture to perform multi-class detection (shouldn't be too difficult right, just add more anchors and output classification scores instead of a single confidence score?)?

I also think that their data augmentation scheme seems to make a lot of sense, could definitely be useful.
```

##### [18-10-04] [paper10]
- PIXOR: Real-time 3D Object Detection from Point Clouds [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf) [[annotated pdf]](https://github.com/fregu856/papers/blob/master/commented_pdfs/PIXOR:%20Real-time%203D%20Object%20Detection%20from%20Point%20Clouds_.pdf)
- *Bin Yang, Wenjie Luo, Raquel Urtasun*
- `CVPR 2018`
- [3D Object Detection]
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
- `NeurIPS 2018`
- [Miscellaneous]
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
- `ICML Workshops 2018`
- [Neural Processes]
```
General comments on paper quality:
Quite well-written overall, although I did find a couple of typos (and perhaps you should expect more for a paper with 7 authors?). Also, I don't think the proposed method is quite as clearly explained as in the Conditional Neural Processes paper.


Paper overview:
The authors introduce Neural Processes (NPs), a family of models that aim to combine the benefits of Gaussian Processes (GPs) and neural networks (NNs). NPs are an extension of Conditional Neural Processes (CNPs) by the same main authors.

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
- `ICML 2018`
- [Neural Processes]
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
- `ICML 2018`
- [Normalizing Flows]
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
- `Journal of Chemical Information and Modeling, 2019`
- [Uncertainty Estimation]
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
- `IEEE Intelligent Vehicles Symposium 2019`
- [Uncertainty Estimation]
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
- `CVPR 2018`
- [Uncertainty Estimation]
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
- `NeurIPS 2017`
- [Uncertainty Estimation]
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
- `ICLR 2018`
- [Theoretical Properties of Deep Learning]
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
