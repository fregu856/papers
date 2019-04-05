##### [19-03-29] [paper52]
- Generating High Fidelity Images with Subscale Pixel Networks and Multidimensional Upscaling [[pdf]](https://openreview.net/forum?id=HylzTiC5Km) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/GENERATING%20HIGH%20FIDELITY%20IMAGES%20WITH%20SUBSCALE%20PIXEL%20NETWORKS%20AND%20MULTIDIMENSIONAL%20UPSCALING.pdf)
- *Jacob Menick, Nal Kalchbrenner*
- `2018-12-04, ICLR2019`

****

### General comments on paper quality:
- Quite interesting paper. I do however think that the proposed method could be more clearly explained, the paper actually left me somewhat confused (I am however not particularly familiar with this specific sub-field).

### Comments:
- For e.g. the images in Figure 5, it is not clear to me how these are actually generated? Do you take a random image from ImageNet, choose a random slice of this image and then generate the image by size- and depth-upscaling?

- For training, I guess that they (for each image in the dataset) choose a random image slice, condition on the previous true image slices (according to their ordering), predict/generate the next image slice and compare this with the ground truth to compute an unbiased estimator of the NLL loss. But what do they do during evaluation? I.e., how are the NLL scores in Table 1-3 computed?

- The experimental results do not seem overly impressive/convincing to me.
