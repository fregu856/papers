##### [20-10-09] [paper107]
- Satellite Conjunction Analysis and the False Confidence Theorem [[pdf]](https://royalsocietypublishing.org/doi/pdf/10.1098/rspa.2018.0565) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Satellite%20conjunction%20analysis%20and%20the%20false%20confidence%20theorem.pdf)
- *Michael Scott Balch, Ryan Martin, Scott Ferson*
- `2018-03-21`

****

### General comments on paper quality:
- Quite well-written and somewhat interesting paper.

### Comments:
- Section 6 (Future and on-going work) is IMO the most interesting part of the paper (_"We recognize the natural desire to balance the goal of preventing collisions against the goal of keeping manoeuvres at a reasonable level, and we further recognize that it may not be possible to achieve an acceptable balance between these two goals using present tracking resources"_).

- To me, it seems like the difference between their proposed approach and the standard approach is mainly just a change in how to interpret very uncertain satellite trajectories. In the standard approach, two very uncertain trajectories are deemed NOT likely to collide (the two satellites could be basically anywhere, so what are the chances they will collide?) . In their approach, they seem to instead say: "the two satellites could be basically anywhere, so they COULD collide!". They argue their approach prioritize safety (which I guess they do, they will check more trajectories since they COULD collide), but it must also actually be useful in practice. I mean, the safest way to drive a car is to just remain stationary at all times, otherwise you risk colliding with something. 
