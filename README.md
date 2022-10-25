## OlfactionAD: Expansive Linguistic Representations to Predict Interpretable Odor Mixture Discriminability

Model can be run to predict 21 olfactory percepts for any molecule using the dataset published in the Dream olfaction dataset (Keller et. al. Science, 2017) using dragons. You can train and then predict the same or different molecules that you have provided dragon features for.

---

## Installation
Git clone a copy of code:
```
git clone https://github.com/jeriscience/OlfactionAD.git
```
## Required dependencies

* [python](https://www.python.org) (3.9.12)
* [numpy](http://www.numpy.org) (1.21.5). It comes pre-packaged in Anaconda.
* [sklearn](https://scikit-learn.org) (1.0.2). It comes pre-packaged in Anaconda.

## 1. Preprocess datasets, including dragon chemoinformatic features, Bushdid, Snitz, and Ravia mixture datasets
* python process_dragon.py

## 2. Train Elastic Network models for single-molecule perceptual predictions
* python train_percept_single_elastic.py

## 3. Predict perceptual values for single molecules and calculate perceptual valuse for mixtures
* python pred_percept_single_elastic.py

## 4. Train Lasso models for mixtures on the Bushdid dataset and predict across cohorts
* python train_pred_mixture_lasso.py

## 5. Use maximum values instead of average when calculating perceptual values for mixtures
* python pred_percept_single_elastic_max.py
* python train_pred_mixture_lasso_max.py

## 6. Investigate the relationship between number of features (alpha in Lasso) and predictive performance
* python train_pred_mixture_lasso_alpha.py


