##### [19-07-11] [paper60]
- Part-A^2 Net: 3D Part-Aware and Aggregation Neural Network for Object Detection from Point Cloud [[pdf]](https://arxiv.org/abs/1907.03670) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Part-A%5E2%20Net:%203D%20Part-Aware%20and%20Aggregation%20Neural%20Network%20for%20Object%20Detection%20from%20Point%20Cloud.pdf)
- *Shaoshuai Shi, Zhe Wang, Xiaogang Wang, Hongsheng Li*
- `2019-07-08`

****

### General comments on paper quality:
- Interesting and quite well-written paper.

### Comments:
- Same main authors as for the [PointRCNN](https://github.com/fregu856/papers/blob/master/summaries/PointRCNN:%203D%20Object%20Proposal%20Generation%20and%20Detection%20from%20Point%20Cloud.md) paper.

- The idea to use the intra-object point locations provided by the ground truth 3D bboxes as extra supervision makes a lot of sense, clever!

- In this paper, the bin-based losses from [PointRCNN](https://github.com/fregu856/papers/blob/master/summaries/PointRCNN:%203D%20Object%20Proposal%20Generation%20and%20Detection%20from%20Point%20Cloud.md) are NOT used.
