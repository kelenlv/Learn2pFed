# Learn What You Need in Personalized Federated Learning
Codes for the paper `Learn What You Need in Personalized Federated Learning` submitted to NeurIPS 2023.



## Table of Content
  - [1. File descriptions](#1file-descriptions)
  - [2. Train *Learn2pFed*](#2train)

## 1. File descriptions

A brief description for the files in this repo:
- `Layers.py` constructions of *Learn2pFed* network for regression in synthetic data and forecasting in ELD dataset
- `demo.sh` & `demo.py` scripts on training *Learn2pFed* on *synthetic* data and real-world data

## 2. Run

### Generalization

To see the improved generalization performance of the GRFF model on the synthetic data and the real-world benchmark data, run
```
sh train.sh
```
Comment or uncomment specific lines in `train.sh` to run the corresponding experiments.



This repo is keeping on updating.
