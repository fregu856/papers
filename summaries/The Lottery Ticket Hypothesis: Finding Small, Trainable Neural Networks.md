##### [18-11-08] [paper17]
- The Lottery Ticket Hypothesis: Finding Small, Trainable Neural Networks [[pdf]](https://arxiv.org/abs/1803.03635) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/The%20Lottery%20Ticket%20Hypothesis:%20Finding%20Small%2C%20Trainable%20Neural%20Networks_.pdf)
- *Jonathan Frankle, Michael Carbin*
- `2018-03-09, ICLR2019 submission`

****

### General comments on paper quality:
- Well-written and very interesting paper. Not particularly heavy to read.

### Paper overview:
- Aiming to help and explain why it empirically seems easier to train large networks than small ones, the authors articulate the *lottery ticket hypothesis*: any large network that trains successfully contains a smaller subnetwork that, when initialized with the same initial parameter values again (i.e., the parameter values they had before the original training began), can be trained in isolation to match (or surpass) the accuracy of the original network, while converging in at most the same number of iterations. The authors call these subnetworks *winning tickets*. 

- When randomly re-initializing the parameters or randomly modifying the connections of winning tickets, they are no longer capable of matching the performance of the original network. Neither structure nor initialization alone is thus responsible for a winning ticket's success. 

- The authors find that a standard pruning technique (which essentially entails removing weights in increasing order of their magnitude (remove small-magnitude weights first)) can be used to automatically uncover such winning tickets.

- They also extend their hypothesis into the conjecture (which they do **not** empirically test) that large networks are easier to train because, when randomly initialized, they contain more combinations of subnetworks and thus more potential winning tickets. 

- They find that winning tickets usually contain just 20% (or less) of the original network parameters. They find winning tickets for both fully-connected, convolutional and residual networks (MNIST, CIFAR10, CIFAR10).

### Comments:
- I actually found this paper a lot more interesting than I initially expected just from reading the title. Easy-to-grasp concept which still might help to improve our understanding of neural networks. 
