##### [20-06-12] [paper101]
- End-to-End Object Detection with Transformers [[pdf]](https://arxiv.org/abs/2005.12872) [[code]](https://github.com/facebookresearch/detr) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/End-to-End%20Object%20Detection%20with%20Transformers.pdf)
- *Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko*
- `2020-05-26`

****

### General comments on paper quality:
- Interesting and well-written paper.

### Comments:
- Interesting and quite neat idea. Impressive results on object detection, and panoptic segmentation.

- It seems like the model requires longer training (500 vs 109 epochs?), and might be somewhat more difficult to train? Would be interesting to play around with the code.

- The "decoder output slot analysis" in Figure 7 is quite interesting. Would be interesting to further study what information has been captured in the object queries (which are just N vectors?) during training.
