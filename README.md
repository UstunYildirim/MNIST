# MNIST
In this project, I practice implementing various different methods for digit recognition (using the MNIST database) without use of high level libraries. Apart from visualization libraries, only numpy is used.

Data is taken from http://yann.lecun.com/exdb/mnist/index.html

For a quick demo you may check out the [Notebook](Notebook.ipynb).

## Fully Connected Neural Network Structures
Linear regression with momentum achieves 9% error rate in 300 iterations

Linear regression with ADAM optimization and mini-batch size of 256 achieves 9.8% error rate in 300 iterations

Linear regression with ADAM optimization and batch-gradient descent achieves 9.2% error rate in 300 iterations

3-Layer NN with hidden units: (300,ReLU), (40, ReLU), ADAM optimization and mini-batches of 128 achieves 5.4% error rate in 1200 iterations

7-Layer NN with hidden units: (2000,ReLU), (1500,ReLU), (1000,ReLU), (500,ReLU), (100,ReLU), (40, ReLU) and ADAM optimization (with various different minibatch sizes during the whole training) achieves 1.86% error rate. 

Sample cost function graph:
![Cost](3-Layer-300r40r.png)

## k-Nearest Neighbor

| k         | accuracy      | accuracy (w/ weights)   | accuracy (w/ PCA)       | accuracy (w/ PCA & weights)   |
|-----------|---------------|-------------------------|-------------------------|-------------------------------|
| 1         | 96.91%        | 96.91%                  | 97.18%                  | 97.18%                        |
| 2         | 96.64%        | 96.91%                  | 96.83%                  | 97.18%                        |
| 3         | 97.14%        | 97.06%                  | 97.55%                  | 97.30%                        |
| 4         | 96.98%        | 97.24%                  | 97.35%                  | 97.55%                        |
| 5         | 96.91%        | 97.16%                  | 97.58%                  | 97.65%                        |
| 6         | 96.93%        | 97.18%                  | 97.44%                  | 97.64%                        |
| 7         | 96.92%        | 97.16%                  | 97.51%                  | 97.56%                        |
| 8         | 96.85%        | 97.19%                  | 97.37%                  | 97.55%                        |
| 9         | 96.66%        | 97.09%                  | 97.47%                  | 97.56%                        |
| 10        | 96.73%        | 96.96%                  | 97.26%                  | 97.55%                        |
| 11        | 96.69%        | 96.98%                  | 97.39%                  | 97.54%                        |
| 12        | 96.60%        | 96.94%                  | 97.24%                  | 97.54%                        |
| 13        | 96.56%        | 96.89%                  | 97.21%                  | 97.58%                        |
| 14        | 96.42%        | 96.82%                  | 97.19%                  | 97.54%                        |
| 15        | 96.35%        | 96.81%                  | 97.20%                  | 97.52%                        |
| 16        | 96.35%        | 96.71%                  | 97.22%                  | 97.52%                        |
| 17        | 96.33%        | 96.68%                  | 97.21%                  | 97.50%                        |
| 18        | 96.35%        | 96.63%                  | 97.19%                  | 97.50%                        |
| 19        | 96.30%        | 96.64%                  | 97.15%                  | 97.38%                        |
| 20        | 96.22%        | 96.61%                  | 97.13%                  | 97.37%                        |
| Optimal k | 3             | 4                       | 5                       | 5                             |

We applied PCA to reduce the number of dimensions from 784 to 30. In this case, dimensionality reduction improved the accuracy a little bit.
However, of course, here the main advantage of PCA is it speeds up the run time by about 2.4 times.

The weights we use are k for the nearest neighbor, (k-1) for the second nearest neighbor and so on.
