##### [18-10-26] [paper16]
- Towards Safe Autonomous Driving: Capture Uncertainty in the Deep Neural Network For Lidar 3D Vehicle Detection [[pdf]](https://arxiv.org/abs/1804.05132) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Towards%20Safe%20Autonomous%20Driving:%20Capture%20Uncertainty%20in%20the%20Deep%20Neural%20Network%20For%20Lidar%203D%20Vehicle%20Detection_.pdf)
- *Di Feng, Lars Rosenbaum, Klaus Dietmayer*
- `2018-09-08, ITSC2018`

****

### General comments on paper quality:
- Fairly well-written paper. Interesting method.

### Paper overview:
- The authors present a two-stage, LiDAR-only (2D bird's eye view as input) model for 3D object detection (trained only on the Car class on KITTI) which attempts to model both epistemic uncertainty (model uncertainty) and aleatoric uncertainty (input-dependent data noise).

- The aleatoric uncertainty is modeled for the regression task in the conventional way, i.e, by assuming a Gaussian distribution over the model output (the model outputs estimates for both the mean and variance) and minimizing the associated negative log-likelihood (actually, they seem to use an L1 or smoothL1 norm instead of L2). Aleatoric uncertainty is only modeled in the output layer, not in the RPN.

- To estimate the epistemic uncertainty, they use MC-dropout in the three fully-connected layers in the refinement head (not in the RPN). They use N=40 forward passes. For classification, the softmax scores are averaged and the computed entropy and mutual information is used as epistemic uncertainty estimates. For regression, the sample variances are used. 

- Before the RPN, they use a ResNet-18(?) to extract a feature map. The model input has a spatial size of 1000x600 pixels. They use a discretization resolution of 0.1 m.

- They train on 9918 training examples and evaluate on 2010 testing examples, both from the KITTI raw dataset. They evaluate their model by computing the F1 score for different IoU thresholds (0.1 to 0.8). I thus find it difficult to compare their 3DOD performance with models on the KITTI leaderboard.

- They find that modeling the aleatoric uncertainty consistently improves 3DOD performance (compared to a fully deterministic baseline version), whereas modeling epistemic uncertainty actually degrades performance somewhat.

- When the authors compute the average epistemic uncertainty for each predicted 3Dbbox, they find that predictions with large IoU values (good predictions, predictions which are close to a ground truth 3Dbbox) generally has smaller associated uncertainty than predictions with small IoU values (poor predictions).

- For the aleatoric uncertainty, they did NOT see this relationship. Instead, they found that the uncertainty generally increased as the distance to the predicted 3Dbbox increased (which makes intuitive sense, distant objects may have just a few associated LiDAR points).
 
### Comments:
- First paper to apply the uncertainty estimation methods of [Kendall and Gal](https://github.com/fregu856/papers/blob/master/summaries/What%20Uncertainties%20Do%20We%20Need%20in%20Bayesian%20Deep%20Learning%20for%20Computer%20Vision%3F.md) to the task of 3DOD, which the authors definitely deserve credit for. Aleatoric uncertainty estimation adds negligible compute and improves performance, whereas the N=40 forward passes needed probably makes the epistemic uncertainty estimation method difficult to deploy in real-time applications. 
