# Iterative Deepening Hyperband
Hyperparameter optimization (HPO) is concerned with the automated search for the most appropriate hyperparameter configuration (HPC) of a parameterized machine learning algorithm. A state-of-the-art HPO method is Hyperband, which, however, has its own parameters that influence its performance. One of these parameters, the maximal budget, is especially problematic: If chosen too small, the budget needs to be increased in hindsight and, as Hyperband is not incremental by design, the entire algorithm must be re-run. This is not only costly but also comes with a loss of valuable knowledge already accumulated. In this paper, we propose incremental variants of Hyperband that eliminate these drawbacks, and show that these variants satisfy theoretical guarantees qualitatively similar to those for the original Hyperband with the "right" budget. Moreover, we demonstrate their practical utility in experiments with benchmark data sets.

## Setup

Install the dependencies of IDHB via ```pip install -r requirements.txt```

## Jupyter Notebooks

The experiment runs can be started via the Jupyter notebooks. Before please fill in the user credentials in ```config/database_credentials.cfg```.

## Experiment Data

An export of our raw data for the experimental results can be found in ```idhb.sql```.