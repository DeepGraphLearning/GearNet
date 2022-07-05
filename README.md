# GearNet: Geometry-Aware Relational Graph Neural Network


This is the official codebase of the paper

[Protein Representation Learning by Geometric Structure Pretraining](https://arxiv.org/abs/2203.06125)

[Zuobai Zhang](https://oxer11.github.io/), [Minghao Xu](https://chrisallenming.github.io/), [Arian Jamasb](https://jamasb.io/), [Vijil Chenthamarakshan](https://researcher.watson.ibm.com/researcher/view.php?person=us-ecvijil), [Aurelie Lozano](https://researcher.watson.ibm.com/researcher/view.php?person=us-aclozano), [Payel Das](https://researcher.watson.ibm.com/researcher/view.php?person=us-daspa), [Jian Tang](https://jian-tang.com/)

## Overview

This codebase is based on PyTorch and [TorchDrug] ([TorchProtein](https://torchprotein.ai)). It supports training and inference
with multiple GPUs or multiple machines.

[TorchDrug]: https://github.com/DeepGraphLearning/torchdrug

## Installation

You may install the dependencies via either conda or pip. Generally, NBFNet works
with Python 3.7/3.8 and PyTorch version >= 1.8.0.

### From Conda

```bash
conda install pytorch=1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pyg -c pyg
conda install rdkit easydict pyyaml -c conda-forge
```


## Reproduction

To reproduce the results of GearBind, use the following command. Alternatively, you
may use `--gpus null` to run GearBind on a CPU. All the datasets will be automatically
downloaded in the code.

We provide the hyperparameters for each experiment in configuration files.
All the configuration files can be found in `config/*.yaml`.

To run GearBind with multiple GPUs, use the following commands

```bash
python -m torch.distributed.launch --nproc_per_node=4 script/run.py -c config/downstream/gearnet.yaml --gpus [0,1,2,3]
```

