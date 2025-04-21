# Objectives

The learning objectives of this assignment are to:
1. get familiar with the TensorFlow Keras framework for training neural networks.
2. experiment with the various hyper-parameter choices of feedforward networks.



# Write your code

You will be editing the file `nn.py`.
**Do not add or edit any other files.**

You will implement several feedforward neural networks using the
[TensorFlow Keras API](https://www.tensorflow.org/guide/keras/).
You should read the documentation strings (docstrings) in each of methods in
`nn.py`, and implement the methods as described.
Write your code below the docstring of each method;
**do not delete the docstrings**.

The following objects and functions may come in handy:
* [Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)
* [Sequential.compile](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile)
* [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
* [Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)
* [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)

# Test your code for correctness

Tests have been provided for you in the `test_nn.py` file.
The tests show how each of the methods is expected to be used.
To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_nn.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
...
collected 3 items

test_nn.py FFF                                                          [100%]

=================================== FAILURES ===================================
...
============================== 3 failed in 5.33s ===============================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
...
collected 3 items

test_nn.py
8.2 RMSE for baseline on Auto MPG
6.2 RMSE for deep on Auto MPG
3.9 RMSE for wide on Auto MPG
.

18.2% accuracy for baseline on UCI-HAR
93.8% accuracy for dropout on UCI-HAR
91.7% accuracy for no dropout on UCI-HAR
.
25.4% accuracy for baseline on census income
79.0% accuracy for early on census income
77.8% accuracy for late on census income
.                                                          [100%]

============================== 3 passed in 23.16s ==============================
```
**Warning**: The performance of your models may change somewhat from run to run,
especially when moving from one machine to another, since neural network models
are randomly initialized.
A correct solution to this assignment should pass the tests on any machine.

otherwise, try different hyper-parameters for your model.

