##### [19-06-05] [paper55]
- LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving [[pdf]](https://arxiv.org/abs/1903.08701) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/LaserNet:%20An%20Efficient%20Probabilistic%203D%20Object%20Detector%20for%20Autonomous%20Driving.pdf)
- *Gregory P. Meyer, Ankit Laddha, Eric Kee, Carlos Vallespi-Gonzalez, Carl K. Wellington*
- `2019-03-20, CVPR2019`

****

### General comments on paper quality:
- Quite well-written and interesting paper. It was however quite difficult to fully grasp their proposed method.

### Comments:
- I struggled to understand some steps of their method, it is e.g. not completely clear to me why both mean shift clustering and adaptive NMS has to be performed.

- I find the used probabilistic model somewhat strange. They say that "our proposed method is the first to capture the uncertainty of a detection by modeling the distribution of bounding box corners", but actually they just predict a single variance value per bounding box (at least when K=1, which is the case for pedestrians and bikes)? 

- Overall, the method seems rather complicated. It is probably not the streamlined and intuitive 3DOD architecture I have been looking for.
