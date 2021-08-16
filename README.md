# MC Dropout Notebooks

Source code for the ICML2021 workshop paper *[Notes on the behavior of MC
Dropout](https://arxiv.org/abs/2008.02627)*

**Authors:** Francesco Verdoja\
**Maintainer:** Francesco Verdoja, francesco.verdoja@aalto.fi\
**Affiliation:** Intelligent Robotics Lab, Aalto University

## Notebooks

This repository contains all Jupyter notebooks needed to replicate the
experiments in the original paper.

Each notebook is named according to the following convention:

`f_{shape}_{bias}_{dropout_rate}.ipynb`

* `shape` describes the function used in the test (among the ones presented in
  the paper)
* `bias` can be either `bias` or `nobias` if the network has bias on the last
  layer or not.
* `dropout_rate` is either `02p` if $p_d = 0.2$, or `05p` if $p_d = 0.5$

## Reference

If you end up using parts or the entirety of the code in this repository for
academic research, please reference the original paper:

> F. Verdoja and V. Kyrki, “Notes on the behavior of MC Dropout,” presented at
the ICML 2021 Workshop on Uncertainty and Robustness in Deep Learning, Jul. 23,
2021 [Online]. Available: https://arxiv.org/abs/2008.02627
