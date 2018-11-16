##### [18-11-15] [paper19]
- Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV) [[pdf]](https://arxiv.org/abs/1711.11279) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Interpretability%20Beyond%20Feature%20Attribution:%20Quantitative%20Testing%20with%20Concept%20Activation%20Vectors%20(TCAV)_.pdf)
- *Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, Rory Sayres*
- `2018-06-07, ICML2018`

****

### General comments on paper quality:
- Quite well-written and fairly interesting paper, the authors do a pretty good job of giving an intuitive explanation of the proposed methods. 

### Paper overview:
- The authors introduce a new method for interpreting the results of trained neural network classification models, in terms of user-defined high-level concepts.

- They introduce Concept Activation Vectors (CAVs), which are vectors in the direction of the activations of a concept's set of example images, and the technique called Testing with CAVs (TCAV), that uses directional derivatives to quantify how important a user-defined concept is to a given classification result (e.g., how important the concept "striped" is to the classification of a given image as "Zebra").

- To obtain a CAV for a given concept (e.g. "striped"), they collect a set of example images representing that concept (e.g. a set of images of various striped shirts and so on), train a linear classifier to distinguish between the activations produced by these concept example images and random images, and choose as a CAV the vector which is orthogonal to the classification boundary of this linear classifier (i.e., the CAV points in the direction of the activations of the concept example images).

- By combining CAVs with directional derivatives, one can measure the sensitivity of a model's predictions to changes in the input towards the direction of a given concept. TCAV uses this to compute a model's conceptual sensitivity across entire classes of inputs, by computing the fraction of images for a given class which were positively influenced by a given concept (the directional derivatives were positive).

- They qualitatively evaluate their method by e.g. sorting images of a given class based on how similar they are to various concepts (e.g. finding the images of "necktie" which are most similar to the concept "model woman"), and comparing the TCAV scores of different concepts for a given classification (e.g. finding that "red" is more important than "blue" for the classification of "fire engine").

### Comments:
- Quite interesting method which I suppose could be useful for some use-cases. I do however find it quite difficult to say how well the proposed method actually works, i.e., it is quite difficult to know whether the successful examples in the paper are just cherry-picked, or if the method consistently makes sense.
