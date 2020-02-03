# MNIST
Various different methods applied on the MNIST database

Data is taken from http://yann.lecun.com/exdb/mnist/index.html

For a quick demo you may check out the [Notebook](Notebook.ipynb).

## Fully Connected Neural Network Structures
Linear regression with momentum achieves 9% error rate in 300 iterations

Linear regression with ADAM optimization and mini-batch size of 256 achieves 9.8% error rate in 300 iterations

Linear regression with ADAM optimization and batch-gradient descent achieves 9.2% error rate in 300 iterations

3-Layer NN with hidden units: (300,ReLU), (40, ReLU), ADAM optimization and mini-batches of 128 achieves 5.4% error rate in 1200 iterations

7-Layer NN with hidden units: (2000,ReLU), (1500,ReLU), (1000,ReLU), (500,ReLU), (100,ReLU), (40, ReLU) and ADAM optimization (with various different minibatch sizes during the whole training) achieves 1.86% error rate. The NN requires about 1.1GB of disk space.

Sample cost function graph:
![Cost](3-Layer-300r40r.png)

## k-Nearest Neighbor

| k | acc |
|---|-----|
| 1 |96.9%|
| 2 |96.0%|
| 3 |94.4%|
| 5 |91.6%|
| 9 |86.7%|
|15 |80.5%|
|21 |75.3%|

