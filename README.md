# Neural Network Implementation

This repository contains an implementation of a neural network algorithm for the Machine Learning class at UFRGS.

## Installation

Install the requirements

```
pip install -r requirements.txt
```

## Validation

Run the unit tests (validating the benchmark data):

```
make run_test
```

In order to validate the backpropagation the following command can be used:

```
python backpropagation.py network.txt weights.txt dataset.txt
```

with the network architecture, weights and data specified through the arguments.

The numeric gradient check can also be validated in the same way, with the following command:

```
python gradient_check.py network.txt weights.txt dataset.txt
```


## Usage

In order to run the experiment which attempts to optimize the network architecture, learning rate, lambda and batch size for the three datasets used, the following command is available

```
make run_experiment
```
