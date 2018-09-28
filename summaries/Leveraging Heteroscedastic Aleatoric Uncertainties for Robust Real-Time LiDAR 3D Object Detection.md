##### [18-09-25] [paper4]
- Leveraging Heteroscedastic Aleatoric Uncertainties for Robust Real-Time LiDAR 3D Object Detection [[pdf]](https://arxiv.org/abs/1809.05590) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Leveraging%20Heteroscedastic%20Aleatoric%20Uncertainties%20for%20Robust%20Real-Time%20LiDAR%203D%20Object%20Detection_.pdf)
- *Di Feng, Lars Rosenbaum, Fabian Timm, Klaus Dietmayer*
- `2018-09-14`

****

### General comments on paper quality:
- Fairly well-written paper. I did find a couple of typos though, and some concepts could definitely have been more carefully and clearly explained.

### Paper overview:
- The authors present a two-stage, LiDAR-only model for 3D object detection (trained only on the Car class on KITTI), which also models aleatoric (heteroscedastic) uncertainty by assuming a Gaussian distribution over the model output, similarly to how [Kendall and Gal](https://github.com/fregu856/papers/blob/master/summaries/What%20Uncertainties%20Do%20We%20Need%20in%20Bayesian%20Deep%20Learning%20for%20Computer%20Vision%3F.md) models aleatoric uncertainty. The 3DOD model takes as input LiDAR point clouds which have been projected to a 2D bird's eye view.

- The network outputs uncertainties in both the RPN and in the refinement head, for the anchor position regression, 3D location regression and orientation regression. They do NOT model uncertainty in the classification task, but instead rely on the conventional softmax scores.

- The deterministic version of the 3DOD model has fairly competitive AP3D performance on KITTI test (not as good as VoxelNet, but not bad for being a LiDAR-only method). What's actually interesting is however that modeling aleatoric uncertainty improves upon this performance with roughly 7% (Moderate class), while only increasing the inference time from 70 ms to 72 ms (TITAN X GPU).

- The authors conduct some experiments to try and understand their estimated aleatoric uncertainties: 
- - They find that the network generally outputs larger orientation uncertainty when the predicted orientation angle is far from the four most common angles {0, 90, 180, 270}.
- -  They find that the outputted uncertainty generally increases as the softmax score decreases.
- - They find that the outputted uncertainty generally increases as detection distance increases.

- The learned aleatoric uncertainty estimates thus seem to make intuitive sense in many cases.

### Comments:
- Just like in [Kendall and Gal](https://github.com/fregu856/papers/blob/master/summaries/What%20Uncertainties%20Do%20We%20Need%20in%20Bayesian%20Deep%20Learning%20for%20Computer%20Vision%3F.md) and [Gast and Roth](https://github.com/fregu856/papers/blob/master/summaries/Lightweight%20Probabilistic%20Deep%20Networks.md), modelling aleatoric uncertainty improves performance while adding negligible computational complexity. That this can be a practically useful tool is thus starting to become quite clear. 

- However, I still think we need to analyze the estimated uncertainties more carefully. Can they be used to form valid confidence intervals? 

- Also, it feels like this work is HEAVILY inspired by [Kendall and Gal](https://github.com/fregu856/papers/blob/master/summaries/What%20Uncertainties%20Do%20We%20Need%20in%20Bayesian%20Deep%20Learning%20for%20Computer%20Vision%3F.md) but it's not really explicitly mentioned anywhere in the paper. I personally think they could have given more credit.
