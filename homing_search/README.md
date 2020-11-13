# Homing Search

Homing Search is a python module for smart hyperparameter optimization in Python.

## Why
Typically for hyperparameter optimization programmers use sklearn's GridSearch or RandomSearch algorithms. 

GridSearch has the advantage of being exhaustive, and the disadvantage of taking impractically long for large search spaces.

RandomSearch has the advantage of being managable to a timeframe of the developer's choosing, but the disadvantage of being non-adaptable to results as they are discovered.

Homing Search seeks to bring together the best of both of these two approaches and to improve upon them by adding adaptation based on the results discovered so far.

## Features
* You provide a time-limit within which it is guaranteed to finish.
* If the search area is small enough, it will perform an exhaustive grid search in approximately the same time as a GridSearch. There is no reason NOT to choose Homing Search in preference to GridSearch.
* If the search area is very large, it will produce better results than both GridSearch and RandomSearch.
* Backend agnostic. Currently supporting functions for Keras are provided to process pandas dataframes into tensorflow datasets, in the future additional supporting functions for sklearn and pytorch databunch will be added.

## Getting started

Install Homing Search from PyPI

```
$ pip install homing_search
```

To run your first example:

```
# TODO: Provide an example using a toy dataset!
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