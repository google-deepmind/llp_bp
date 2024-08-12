# llp_bp

This repository provides the code for implementing the LLP-BP algorithm
described in the paper [Learning from Label Proportions: Bootstrapping
Supervised Learners via Belief Propagation](https://arxiv.org/abs/2310.08056).

## Abstract

Learning from Label Proportions (LLP) is a learning problem where only aggregate
level labels are available for groups of instances, called bags, during
training, and the aim is to get the best performance at the instance-level on
the test data. This setting arises in domains like advertising and medicine due
to privacy considerations. We propose a novel algorithmic framework foßr this
problem that iteratively performs two main steps. For the first step (Pseudo
Labeling) in every iteration, we define a Gibbs distribution over binary
instance labels that incorporates a. covariate information through the
constraint that instances with similar covariates should have similar labels and
b. the bag level aggregated label. We then use Belief Propagation (BP) to
marginalize the Gibbs distribution to obtain pseudo labels. In the second step
(Embedding Refinement), we use the pseudo labels to provide supervision for a
learner that yields a better embedding. Further, we iterate on the two steps
again by using the second step's embeddings as new covariates for the next
iteration. In the final iteration, a classifier is trained using the pseudo
labels. Our algorithm displays strong gains against several SOTA baselines (up
to 15%) for the LLP Binary Classification problem on various dataset types -
tabular and Image. We achieve these improvements with minimal computational
overhead above standard supervised learning due to Belief Propagation, for large
bag sizes, even for a million samples.

## Installation

Please install the required packages as listed in train.py .

## Usage

Please run the train file from your terminal as a default command. The
hyper-parameter values are saved as default in the file.

`python3 train.py`

## Citing this work

If you use any snippet from our code, or found the paper relevant, please cite
our ICLR '24 work:

```latex
@misc{havaldar2024learninglabelproportionsbootstrapping,
      title={Learning from Label Proportions: Bootstrapping Supervised Learners via Belief Propagation},
      author={Shreyas Havaldar and Navodita Sharma and Shubhi Sareen and Karthikeyan Shanmugam and Aravindan Raghuveer},
      year={2024},
      eprint={2310.08056},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2310.08056},
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you
may not use this file except in compliance with the Apache 2.0 license. You may
obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.

## License and disclaimer

Please contact shreyasjh[at]google.com or navoditasharma[at]google.com for any
questions or suggestions.
