##### [19-07-10] [paper59]
- PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud [[pdf]](https://arxiv.org/abs/1812.04244) [[code]](https://github.com/sshaoshuai/PointRCNN) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/PointRCNN:%203D%20Object%20Proposal%20Generation%20and%20Detection%20from%20Point%20Cloud.pdf)
- *Shaoshuai Shi, Xiaogang Wang, Hongsheng Li*
- `2018-12-11, CVPR2019`

****

### General comments on paper quality:
- Interesting and quite well-written paper.

### Comments:
- I think I like this approach to 3DOD. Directly processing the point cloud and generating proposals by classifying each point as foreground/background makes sense, is quite simple and seems to perform well. Their bin-based regression losses seem somewhat strange to me though.
