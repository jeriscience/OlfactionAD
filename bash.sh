#!/bin/bash

set -e

## 1. clean up and combine dragon features
python process_dragon.py

## 2. train perception model for single molecules 
python train_percept_single_elastic.py

## 3. predict perceptual values for single molecules; calculate perceptual valuse for mixtures
python pred_percept_single_elastic.py

## 4. train models for mixture on bushdid and predict across cohorts
python train_pred_mixture_lasso.py

## 5. use maximum values instead of average when calculating perceptual values for mixtures
python pred_percept_single_elastic_max.py
python train_pred_mixture_lasso_max.py

## 6. investigate the relationship between number of features (alpha in Lasso) and predictive performance
python train_pred_mixture_lasso_alpha.py

