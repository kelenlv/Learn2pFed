# Learn What You Need in Personalized Federated Learning
Codes for the paper `Learn What You Need in Personalized Federated Learning` submitted to NeurIPS 2023.



## Table of Content
  - [1. File descriptions](#1file-descriptions)
  - [2. Train and attack](#2train-and-attack)

## 1. File descriptions

A brief description for the files in this repo:
- `Layers.py` constructions of *Learn2pFed* network for regression in synthetic data and forecasting in ELD dataset
- 
- `modelv.py` definitions of the variant of the GRFF model for image data
- `data_loader.py` scripts on loading the data
- `train.sh` & `train.py` scripts on training the GRFF model on *synthetic* data and real-world *benchmark* data
- `train_attack_mnist.sh` & `train_mnist.py` & `attack_mnist.py` scripts on training and attacking the GRFF variant on MNIST

## 2. Run

### Generalization

To see the improved generalization performance of the GRFF model on the synthetic data and the real-world benchmark data, run
```
sh train.sh
```
Comment or uncomment specific lines in `train.sh` to run the corresponding experiments.



This repo is keeping on updating.
