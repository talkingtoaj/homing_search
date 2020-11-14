# Homing Search

Homing Search is a python module for smart hyperparameter optimization in Python.

## Why
Typically for hyperparameter optimization programmers use sklearn's GridSearch or RandomSearch algorithms. 

GridSearch has the advantage of being exhaustive, and the disadvantage of taking impractically long for large search spaces.

RandomSearch has the advantage of being managable to a timeframe of the developer's choosing, but the disadvantage of being non-adaptable to results as they are discovered.

Homing Search seeks to bring together the best of both of these two approaches and to improve upon them by adding adaptation based on the results discovered so far.

The library is written to be compatible with multiple Machine Learning Libaries (e.g. Keras, Sci-kit Learn, Pytorch), however to minimize required dependencies, each libary will have its own forked package.

## Features
* You provide a time-limit within which it is guaranteed to finish.
* If the search area is small enough, it will perform an exhaustive grid search in approximately the same time as a GridSearch. There is no reason NOT to choose Homing Search in preference to GridSearch.
* If the search area is very large, it will produce better results than both GridSearch and RandomSearch.
* Backend agnostic. Currently supporting functions for Keras are provided to process pandas dataframes into tensorflow datasets, in the future additional supporting functions for sklearn and pytorch databunch will be added.

## Getting started

Install Homing Search from PyPI

```
$ pip install homing-search-keras
```

To run your first example:

Write a build_fn that takes a combination of parameters as inputs and a compiled model as output
* build_fn (**kwargs)

Define a dictionary of parameters to search. values provided as lists are statically searched, but values of int or float provided as sets are ranged (ie when a promising result is found, it will begin exploring other values near to the mean best score)

```
params = {
    'layer1_dim':{800, 2000, 4000}, 
    'layer2_dim':{100,300,500}, 
    'layer1_dropout':{0.1, 0.3, 0.5}, 
    'layer2_dropout':{0.1, 0.3, 0.5},
    'activation':['relu', 'tanh', 'softplus', 'elu'], 
    'optimizer':['Adagrad','SGD','RMSprop','adam', 'Adadelta'], 
    'learning_rate':{0.1, 0.01}, 
    'batch_size': [64, 128, 256, 512],
}    
repeats = 2 # number of times to repeat each parameter option to get an average score
epochs = 200
time_limit = 30 # will optimize search to end no later than 30 minutes
NN_builder # a function that 

from homing_search import HomingSearchKeras

hsk = HomingSearchKeras(build_fn=build_fn, data=pandas_df, label='price', batch_size=256, save_tf_logs=False)
hsk.start(params, repeats, epochs, time_limit)

```

## Contributing
If you're a developer and wish to contribute:

1. Create an account on GitHub if you do not already have one.

2. Fork the project repository: click on the ‘Fork’ button near the top of the page. This creates a copy of the code under your account on the GitHub user account. For more details on how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

3. Clone your fork of the homing_search repo from your GitHub account to your local disk:

```
$ git clone https://github.com/<your github username>/homing_search.git
$ cd homing_search
```