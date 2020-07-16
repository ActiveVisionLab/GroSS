---
layout: index
title: "GroSS: Group-Size Series Decomposition for Grouped Architecture Search"
---

This is the landing page for paper **GroSS: Group-Size Series Decomposition for Grouped Architecture Search**.

## Abstract
We present a novel approach which is able to explore the configuration of grouped convolutions within neural networks. Group-size Series (GroSS) decomposition is a mathematical formulation of tensor factorisation into a series of approximations of increasing rank terms. GroSS allows for dynamic and differentiable selection of factorisation rank, which is analogous to a grouped convolution. Therefore, to the best of our knowledge, GroSS is the first method to enable simultaneously train
differing numbers of groups within a single layer, as well as all possible combinations between layers. In doing so, GroSS is able to train an entire grouped convolution architecture search-space concurrently. We demonstrate this through architecture searches with performance objectives and evaluate its performance against conventional Block Term Decomposition. GroSS enables more effective and efficient search for grouped convolutional architectures. 

## Code

The code for reproducing results in the paper can be obtained from the [GitHub repository](https://github.com/ActiveVisionLab/GroSS).

## Citation

BiBTeX:

```
@misc{howardjenkins2019gross,
    title={GroSS: Group-Size Series Decomposition for Grouped Architecture Search},
    author={Henry Howard-Jenkins and Yiwen Li and Victor A. Prisacariu},
    year={2019},
    eprint={1912.00673},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

Plain text:
Henry Howard-Jenkins, Yiwen Li, Victor A. Prisacariu, "GroSS: Group-Size Series Decomposition for Grouped Architecture Search", in [arXiv:1923.00673v2](https://arxiv.org/abs/1912.00673v2)
