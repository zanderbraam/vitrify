vitrify
==============================

This repository contains all the code for the Master's thesis, *Interpreting Decision Boundaries of Deep Neural Networks*.
An abstract of the thesis follows:

> As deep learning methods are becoming the front runner among machine learning techniques, the importance of
> interpreting and understanding these methods grows. Deep neural networks are known for their highly competitive
> prediction accuracies, but also infamously known for their ``black box'' properties when it comes to their decision
> making process. Tree-based models on the other end of the spectrum, are highly interpretable models, but lack the
> predictive power with high dimensional data. The proposed solution of this thesis is to combine these two methods and
> get the predictive accuracy from the complex learner, but also the explainability from the interpretable learner.
> The suggested method is a continuation of the work done by the Google Brain Team in their paper
> *Distilling a Neural Network Into a Soft Decision Tree* (Frosst and Hinton, 2017). Frosst and Hinton argue that the
> reason why it is difficult to understand how a neural network model comes to a particular decision, is due to the
> learner being reliant on distributed hierarchical representations. If the knowledge gained by the deep learner were
> to be transferred to a model based on hierarchical decisions instead, interpretability would be much easier.
> Their proposed solution is to use a "deep neural network to train a soft decision tree that mimics the input-output
> function discovered by the neural network". This thesis tries to expand upon this by using generative models
> (Goodfellow et al., 2016), in particular VAE's (variational autoencoders), to generate additional data from the
> training data distribution. This synthetic data can then labelled by the complex learner we wish to approximate.
> By artificially growing our training set, we can overcome the statistical inefficiencies of decision trees and
> improve model accuracy.

==============================

# Installation on Linux

After cloning the repository, create a virtual environment.
A virtual environment creates a sandbox that does not affect the system Python.

```
virtualenv ve -p `which python3.6`
source ./ve/bin/activate
pip install numpy
pip install -r requirements.txt
```

Also, please do the following for plotting decision trees:

```
sudo apt-get install graphviz
```

Git Large File Storage ([git-lfs](https://git-lfs.github.com/)) is used to store model checkpoints. To install it, run
```
git lfs install
```

Code was tested on [Python](https://www.python.org/) 3.5 with [TensorFlow](https://www.tensorflow.org/) version 1.10.0 and [matplotlib](https://matplotlib.org/) version 2.1.0.
==============================

Example notebooks can be found at:

1. [vitrify_mnist.ipynb](./notebooks/vitrify_mnist.ipynb)

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── EMNIST_Letter             <- Data from the EMNIST-Letter dataset (numpy arrays).
    │   ├── EMNIST_Letter_Uppercase   <- Data from the EMNIST-Letter dataset, containing only the uppercase letters (numpy arrays).
    │   ├── FashionMNIST              <- Data from the Fashion-MNIST dataset (numpy arrays).
    │   └── MNIST                     <- Data from the MNIST dataset (numpy arrays).
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Code for the Soft Decision Trees was adapted from [https://github.com/lmartak/distill-nn-tree](https://github.com/lmartak/distill-nn-tree)
