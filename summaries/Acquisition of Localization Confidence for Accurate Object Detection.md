##### [19-06-12] [paper56]
- Acquisition of Localization Confidence for Accurate Object Detection [[pdf]](https://arxiv.org/abs/1807.11590) [[code]](https://github.com/vacancy/PreciseRoIPooling) [[oral presentation]](https://youtu.be/SNCsXOFr_Ug) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Acquisition%20of%20Localization%20Confidence%20for%20Accurate%20Object%20Detection.pdf)
- *Borui Jiang, Ruixuan Luo, Jiayuan Mao, Tete Xiao, Yuning Jiang*
- `2018-07-30, ECCV2018`

****

- Interesting idea that intuitively makes a lot of sense, neat to see that it actually seems to work quite well.

- While the predicted IoU is a measure of "localization confidence", it is not an ideal measure of localization uncertainty. Having an estimated variance each for (x, y, w, h) would provide more information.
