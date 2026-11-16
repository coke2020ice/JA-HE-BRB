# Experimental protocol notes

These notes preserve protocol details moved out of the main experimental narrative. They should accompany the experimental code and submission materials.

## Table 5 repetition and aggregation

Metrics are averaged over ten training runs within each outer fold, except for CDC, EVFeX and AnonGBDT, which use one run. Reported means and standard deviations are computed across the five outer folds.

## CDC evaluation

The CDC results in Table 5 use additional training-fold validation and class-balancing settings. They are exploratory because the outer-fold results had been examined during development and additional tuning budgets differed across methods.

The previous manuscript also stated that CDC uses no additional label flips and selects class weighting and undersampling by inner-validation BA, retaining all test records. Its exact scope across the Table 5 and local-to-joint experiments needs to be matched to the saved experiment records before submission; it should not be assumed to describe all CDC experiments.

## Saved configurations

Predictive comparisons and supplementary analyses use their respective saved configurations. Outside the exploratory CDC search, saved configurations and tree-round counts remain fixed.
