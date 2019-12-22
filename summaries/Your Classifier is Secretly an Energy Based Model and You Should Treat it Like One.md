##### [19-12-22] [paper74]
- Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One [[pdf]](https://arxiv.org/abs/1912.03263) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Your%20Classifier%20is%20Secretly%20an%20Energy%20Based%20Model%20and%20You%20Should%20Treat%20it%20Like%20One.pdf)
- *Will Grathwohl, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, Kevin Swersky*
- `2019-12-06, ICLR 2020`

****

### General comments on paper quality:
- Interesting and very well-written paper. I recommend actually going through the appendix as well, as it contains some interesting details.

### Comments:
- The idea to create an energy-based model for p(x) by marginalizing out y is really neat and makes a lot of sense in this classification setting (in which this corresponds to just summing the logits for all K classes). This EBM for p(x) is then trained using the MCMC-based ML learning method employed in other recent work. Simultaneously, a model for p(y|x) is also trained using the standard approach (softmax / cross entropy), thus training p(x, y) = p(y | x)*p(x).

- I am however not overly impressed/convinced by their experimental results. All experiments are conducted on relatively small and "toy-ish" datasets (CIFAR10, CIFAR100, SVHN etc), but they still seemed to have experienced A LOT of problems with training instability. Would be interesting to see results e.g. for semantic segmentation on Cityscapes (a more "real-world" task and dataset).

- Moreover, like the authors also point out themselves, training p(x) using SGLD-based sampling with L steps (they mainly use L=20 steps, but sometimes also have to restart training with L=40 to mitigate instability issues) basically makes training L times slower. I am just not sure if the empirically observed improvements are strong/significant enough to justify this computational overhead.
