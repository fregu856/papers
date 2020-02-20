##### [20-02-18] [paper89]
- Probabilistic 3D Multi-Object Tracking for Autonomous Driving [[pdf]](https://arxiv.org/abs/2001.05673) [[code]](https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Probabilistic%203D%20Multi-Object%20Tracking%20for%20Autonomous%20Driving.pdf)
- *Hsu-kuang Chiu, Antonio Prioletti, Jie Li, Jeannette Bohg*
- `2020-01-16`

****

### General comments on paper quality:
- Interesting and well-written paper.

### Comments:
- They provide more details for the Kalman filter, which I appreciate.

- The design choices that differs compared to [AB3DMOT](https://github.com/fregu856/papers/blob/master/summaries/A%20Baseline%20for%203D%20Multi-Object%20Tracking.md) all make sense I think (e.g., Mahalanobis distance instead of 3D-IoU as the affinity measure in the data association), but the gain in performance in Table 1 does not seem overly significant, at least not compared to the huge gain seen when switching to the MEGVII 3D detector in AB3DMOT.
