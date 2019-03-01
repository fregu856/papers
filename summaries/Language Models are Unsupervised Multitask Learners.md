##### [19-03-01] [paper48]
- Language Models are Unsupervised Multitask Learners [[pdf]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) [[blog post]](https://blog.openai.com/better-language-models/) [[code]](https://github.com/openai/gpt-2) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Language%20Models%20are%20Unsupervised%20Multitask%20Learners.pdf)
- *Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever*
- `2019-02-14`

****

### General comments on paper quality:
- Interesting and quite well-written paper. There are not that many technical details, one would probably have to read previous work for that. One probably needs to be somewhat familiar with NLP.

### Comments:
- Very impressive work from an infrastructure perspective.

- Just as context to their model with 1.5 billion parameters: a ResNet101 has 45 million parameters, which takes up 180 Mb when saved to disk. DeepLabV3 for semantic segmentation has roughly 75 million parameters.
 
- This has become a pretty hyped paper, and I agree that the work is impressive, but it still seems to me like their model is performing roughly as one would expect. It performs really well on general language modeling tasks, which is exactly what it was trained for (although it was not fine-tuned on the specific benchmark datasets), but performs rather poorly on translation and question-answering. The fact that the model has been able to learn some basic translation in this fully unsupervised setting is still quite impressive and interesting though.
