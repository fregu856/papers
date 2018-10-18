##### [18-10-04] [paper10]
- PIXOR: Real-time 3D Object Detection from Point Clouds [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/PIXOR:%20Real-time%203D%20Object%20Detection%20from%20Point%20Clouds_.pdf)
- *Bin Yang, Wenjie Luo, Raquel Urtasun*
- `CVPR2018`

### General comments on paper quality:
- Fairly well-written paper, although there are a couple of typos (weird grammar). Quite interesting proposed 3D localization model.

### Paper overview:
- The authors present a single-stage, LiDAR-only model for 3D localization of vehicles (bird's eye view).

- The most interesting/novel contribution is probably the utilized LiDAR input representation: 
- - They discretize 3D space into a 3D voxel grid of resolution 0.1 m, each grid cell is then either assigned a value of 1 (if the grid cell contains any LiDAR points) or 0 (if the grid cell does NOT contain any LiDAR points), which results in a 3D occupancy map (e.g. a 800x700x35 tensor of 0s and 1s). 
- - This 3D tensor is then fed as input to a conventional (2D) CNN, i.e., the height dimension plays the role of the rgb channels in an image(!). 
- - The approach is thus very similar to models which first project the LiDAR point cloud onto a bird's eye view, in that we only need to use 2D convolutions (which is significantly more efficient than using 3D convolutions), but with the difference being that we in this approach don't need to extract any hand-crafted features in order to obtain a bird's eye view feature map. Thus, at least in theory, this approach should be comparable to bird's eye view based models in terms of efficiency, while being capable of learning a more rich bird's eye view feature representation.

- The model outputs a 200x175x7 tensor, i.e., 7 values (one objectness/confidence score + cos(theta) and sin(theta) + regression targets for x, y, w, and l) for each grid cell (when the feature map is spatially down-sampled in the CNN, this corresponds to an increasingly sparser grid). The authors say that their approach doesn't use any pre-defined object anchors, but actually I would say that it uses a single anchor per grid cell (centered at the cell, with width and length set to the mean of the training set, and the yaw angle set to zero).

- They use the focal loss to handle the large class imbalance between objects and background.

- In inference, only anchors whose confidence score exceeds a certain threshold are decoded (i.e., rotated, translated and resized according to the regressed values), and non-maximum-suppression (NMS) is then used to get the final detections.

- They evaluate their method on KITTI and compare to other entries on the bird's eye view leaderboard. They obtain very similar performance to the LiDAR-only version of MV3D and somewhat significantly worse than VoxelNet, i.e., not OVERLY impressive performance but still pretty good. The method is also significantly faster in inference than both MV3D and VoxelNet.


### Comments:
- Pretty interesting paper. The approach of creating a 3D occupancy map using discretization and then processing this with 2D convolutions seems rather clever indeed. One would think this should be quite efficient while also being able to learn a pretty rich feature map.

- I don't think the model should be particularly difficult to extend to full 3D object detection either, you would just need to also regress values for z (relative to some default ground plane z value, i.e., we assume that all anchor 3dbboxes sit on the ground plane) and h (relative to the mean h in the training set). I think this is basically what is done in VoxelNet?

- There are however some design choices which I find somewhat peculiar, e.g. the way they assign anchors (the authors just talk about "pixels") to being either positive (object) or negative (background).
