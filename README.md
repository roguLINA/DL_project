# Adversarial radius is a confidence measure
This repository contains all code for final project on Skoltech Deep Learning course. Authors: Alina Rogulina, Nikita Baramiia, Sergey Petrakov, Valerii Kornilov, 
Nikita Balabin, Alexey Zaytsev.

**Abstract:**

We propose a novel approach to uncertainty estimation, motivated with concepts from adversarial attacks research.
In particular, we provide an explanation of how adversarial radius can be used in task of uncertainty estimation and why it is a good instrument for measuring uncertainty. To support our approach we present empirical evidence of its validity and compare it with other popular uncertainty estimators.

**The structure of the repository is as follows**:
- *folder utils*
  - attacks.py - basic functions for gradient attacks
  - adevrsarial_radius.py - functions for estimation of adversarial radii/ step radii on single model/ensemble
  - uncertainty_baselines.py - functions for calculating baseline uncertainty measures on single model/ensemble
  - data.py - dataloading utils
  - model.py - training utils
- *training for ensembles:*
  - bagging-ensemble-training.ipynb
  - snapshot-ensemble-training.ipynb
- *uncertainty metrics estimation and their comparison:*
  - bagging-ensemble-experiments.ipynb
  - snapshot-ensemble-experiments.ipynb
  - one-model-experiments.ipynb
 - *folder results* - contains all calculated metrics from 3 above mentioned notebooks with experiments and saved versions of comparison plots.
