##### [18-10-05] [paper11]
- VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection [[pdf]](https://arxiv.org/abs/1711.06396) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/VoxelNet:%20End-to-End%20Learning%20for%20Point%20Cloud%20Based%203D%20Object%20Detection_.pdf)
- *Yin Zhou, Oncel Tuzel*
- `2017-11-17, CVPR2018`

****

### General comments on paper quality:
- Well-written and interesting paper, the proposed architecture is clearly described.

### Paper overview:
- The authors present a single-stage, LiDAR-only model for 3D object detection (3DOD) of a single object class (e.g. vehicles), and evaluate the model on the KITTI dataset.

- They discretize 3D space into a 3D voxel grid of resolution 0.2x0.2x0.4 m, and the LiDAR points are grouped according to which voxel they reside in. If a voxel contains more than T (T = 35 for vehicles, T = 45 for pedestrians/cyclists) LiDAR points, T points are randomly sampled to represent that voxel. For each non-empty voxel, the corresponding LiDAR points are then fed through "Voxel Feature Encoding layers" (basically a PointNet) to extract a learned feature vector of dimension C (C = 128). The result of this process is thus a (sparse) CxD'xH'xW' (128x10x400x352 for vehicles) feature map representing the original LiDAR point cloud.

- This 3D feature map is processed by 3D convolutions and flattened in order to obtain a 128xH'xW' 2D feature map, which is fed as input to a conventional (2D convolutions) region proposal network (RPN).

- The RPN outputs a Kx(H'/2)x(W'/2) confidence/objectness score map, and a (7K)x(H'/2)x(W'/2) regression map, which contains the 7 regression outputs (x, y, z, h, w, l, theta) for each of the K anchors at each grid cell position.

- The authors use K=2 anchors per grid cell, with theta = 0 deg or 90 deg, both with (w, h, l) set to the mean size from the training data and z set to -1. The grid is thus defined in a 2D bird's eye view, but still corresponds to anchor 3D bounding boxes on the plane z=-1 (which intuitively should work well in the application of autonomous driving where most cars lie on the same ground plane).

- Anchors are assigned to either being positive, negative or don't-care based on their *bird's eye view* IoU with the ground truth bounding boxes. The confidence/classification loss is computed for both positive and negative anchors, while the regression loss is computed only for positive anchors.

- The authors train three separate networks for detection of vehicles, pedestrians and cyclists, respectively.

- They compare their networks' performance with other models on both the KITTI 3D and KITTI bird's eye view leaderboards, and find that VoxelNet outperforms all LiDAR-only methods across the board. Compared to [PIXOR](https://github.com/fregu856/papers/blob/master/summaries/PIXOR:%20Real-time%203D%20Object%20Detection%20from%20Point%20Clouds.md) (which only submitted results for bird's eye view), VoxelNet has better performance but is significantly slower in inference. The VoxelNet inference time is dominated by the 3D convolutions.  

### Comments:
- Interesting 3DOD model! Using (what is basically) a PointNet to extract feature vectors from groups of LiDAR points and thus obtain a learned 3D feature map is really rather clever, all though using 3D convolutions has a clear negative effect on inference time.

- The remaining parts of the architecture seems well-designed (more so than e.g. [PIXOR](https://github.com/fregu856/papers/blob/master/summaries/PIXOR:%20Real-time%203D%20Object%20Detection%20from%20Point%20Clouds.md)), and thus VoxelNet seems like a reasonable candidate to extend in future work on LiDAR-only 3DOD. Could you e.g. extend the architecture to perform multi-class detection (shouldn't be too difficult right, just add more anchors and output classification scores instead of a single confidence score?)? 

- I also think that their data augmentation scheme seems to make a lot of sense, could definitely be useful.

