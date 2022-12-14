# Numerical experiments for BEER

This repository contains numerical experiments for "BEER: Fast O(1/T) Rate for Decentralized Nonconvex Optimization with Communication Compression" [[PDF]](https://arxiv.org/pdf/2201.13320.pdf).

If you find this repo useful, please cite our paper
```
@article{zhao2022beer,
  title={BEER: Fast O (1/T) Rate for Decentralized Nonconvex Optimization with Communication Compression},
  author={Zhao, Haoyu and Li, Boyue and Li, Zhize and Richt{\'a}rik, Peter and Chi, Yuejie},
  journal={Advances in Neural Information Processing Systems},
  volume = {35},
  year={2022}
}
```

## 1. Folder structure

- `beer/`: framework for convolutional neural network experiments.

- `experiments/experiments.ipynb`: code for synthetic numerical experiments.

- `experiments/mnist/`: code and scripts for convolutional neural network experiments.

## 2. Installation

Please install [[this package]](https://github.com/liboyue/Network-Distributed-Algorithm) first.

Then run `pip install git+https://github.com/liboyue/beer.git` to install this package.
