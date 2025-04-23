# Objectives

The learning objectives of this assignment are to:
1. learn the TensorFlow Keras APIs for convolutional and recurrent neural
   networks.
2. explore the space of hyper-parameters for convolutional and recurrent
   networks.


# Write your code

You will be editing the file `nn.py`.
**Do not add or edit any other files.**

You will implement several convolutional and recurrent neural networks using the
[TensorFlow Keras API](https://www.tensorflow.org/guide/keras/).
You should read the documentation strings (docstrings) in each of methods in
`nn.py`, and implement the methods as described.
Write your code below the docstring of each method;
**do not delete the docstrings**.

The following objects and functions may come in handy:
* [Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)
* [Sequential.compile](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile)
* [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
* [Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding)
* [Conv1D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D)
* [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)
* [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)
* [GlobalMaxPool1D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalMaxPool1D)
* [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten)
* [SimpleRNN](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN)
* [LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
* [GRU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)
* [Bidirectional](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional)

# Test your code for correctness

Tests have been provided for you in the `test_nn.ipynb` file.
The tests show how each of the methods is expected to be used.
To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_nn.ipynb``.
Initially, you will see output like:
```
============================= test session starts ==============================
...
collected 4 items

test_nn.py FFFF                                                          [100%]

=================================== FAILURES ===================================
...
============================== 4 failed in 6.04s ===============================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
...
collected 4 items

test_nn.py
0.6 RMSE for RNN on toy problem
.
89.0% accuracy for CNN on MNIST sample
.
88.9% accuracy for RNN on Youtube comments
.
85.4% accuracy for CNN on Youtube comments
.                                                          [100%]

============================== 4 passed in 56.97s ==============================
```
**Warning**: The performance of your models may change somewhat from run to run,
especially when moving from one machine to another, since neural network models
are randomly initialized.
A correct solution to this assignment should pass the tests on any machine.
otherwise, try different hyper-parameters for your model.
