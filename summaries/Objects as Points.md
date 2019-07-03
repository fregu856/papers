##### [19-07-03] [paper58]
- Objects as Points [[pdf]](https://arxiv.org/abs/1904.07850) [[code]](https://github.com/xingyizhou/CenterNet) [[pdf with comments]]()
- *Xingyi Zhou, Dequan Wang, Philipp Krähenbühl*
- `2019-04-16`

****

### General comments on paper quality:
- Quite well-written and interesting paper.

### Comments:
- Multiple objects (of the same class) having the same (low-resolution) center point is apparently not very common in MS-COCO, but is that true also in real life in automotive applications? And in these cases, would only detecting one of these objects be a major issue? I do not really know, I find it somewhat difficult to even visualize cases where multiple objects would share center points.

- It is an interesting point that this method essentially corresponds to anchor-based one-stage detectors, but with just one shape-agnostic anchor. Perhaps having multiple anchors per location is not super important then?
