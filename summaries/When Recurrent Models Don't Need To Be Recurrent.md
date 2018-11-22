##### [18-11-22] [paper21]
- When Recurrent Models Don't Need To Be Recurrent [[pdf]](https://arxiv.org/abs/1805.10369) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/When%20Recurrent%20Models%20Don%E2%80%99t%20Need%20To%20Be%20Recurrent.pdf)
- *John Miller, Moritz Hardt*
- `2018-05-29`

****

### General comments on paper quality:
- Reasonably well-written and somewhat interesting paper. I do not think it is intended for publication in any conference/journal.

### Paper overview:
- The authors present a number of theorems, proving that *stable* Recurrent Neural Networks (RNNs) can be well-approximated by standard feed-forward networks. Moreover, if gradient descent succeeds in training a stable RNN, it will also succeed in training the corresponding feed-forward model. I.e., *stable* recurrent models do not actually need to be recurrent (which can be very convenient, since feed-forward models usually are easier and less computationally expensive to train).  

- For a vanilla RNN, h_t = rho(W*h_{t-1} + U*x_t), stability corresponds to requiring ||W|| < 1/L_rho (L_rho is the Lipschitz constant of rho).

- You construct the corresponding feed-forward model approximation by moving over the input sequence with a sliding window of length k, producing an output every time the window advances by one step (auto-regressive model). 

- They show that stable recurrent models effectively do not have a long-term memory, and relate this to the concept of vanishing gradients (if the gradients of a recurrent model quickly vanish, then it could be well-approximated by a feed-forward model, even though the model was not explicitly constrained to be stable?). 

### Comments:
- I find it difficult to judge how significant the presented results actually are, I think you need to be more familiar with the latest research within RNNs to properly appreciate the paper. 
