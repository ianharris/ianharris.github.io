---
title:  "Introduction to Linear Regression"
date:   2018-06-11 12:00:00 +0000
categories: [ Machine Learning ]
---

# Introduction

In our last article we [introduced Supervised Learning]({% post_url 2018-05-19-introduction-to-supervised-learning %}) and used as examples of Supervised Learning, equations relevant to Linear Regression with Gradient Descent. In this post we will look at using Linear Regression with Gradient Descent to learn a relationship between a set of features and labels. 


# Defining our Data Set

To apply Linear Regression we will first need a data set. However, instead of using an open source data set, we will instead generate a data set from a known mathematical function. In this scenario, the function we use to generate the data corresponds to the target function, \\(h(X)\\), that we want to learn. As a result this method of data generation will allow us to very clearly demonstrate that the machine learning algorithm is 'learning' the correct function.

It should be noted that while generating data from a known mathematical function is beneficial to demonstrating how a Supervised Learning algorithm learns a relationship between features and labels, it will not produce a very realistic data set. We will address more complex data sets in a future post.
The function we will use to generate the data set is a linear function of two independent variables \\((X\_{1}, X\_{2})\\). Specifically, we will look at the function:

$$
\begin{equation}
f(X_{1}, X_{2}) = h(X) = 0.5 \cdot X_{1} + 0.5 \cdot X_{2}\label{eqn:target}\tag{1}
\end{equation}
$$

A visualisation of the function \\(\ref{eqn:target}\\) is presented in Figure 1 below. As we have two features (our independent variables \\((X\_{1}, X\_{2})\\)) our hypothesis function will be defined by:

$$
\begin{equation}
h_{\theta}(X) = \theta_{0} + \theta_{1} \cdot{} X_{1} + \theta_{2} \cdot{} X_{2}\label{eqn:hyp}\tag{2}
\end{equation}
$$

If we compare equations \\(\ref{eqn:target}\\) and \\(\ref{eqn:hyp}\\) we see that the values of \\((\theta\_{0}, \theta\_{1}, \theta\_{2})\\) that our algorithm should learn are \\((\theta\_{0} = 0, \theta\_{1} = 0.5, \theta\_{2} = 0.5)\\).

The data set used for this tutorial was generated using a random number generator to create \\((X\_{1}, X\_{2})\\) pairs and then a \\(y\\) value was calculated using equation \\(\ref{eqn:target}\\). The data set is included with the code for the project (which can be found at [linear-regression-gradient-descent-numpy](https://github.com/ianharris/linear-regression-gradient-descent-numpy)) in 'data/training-plane.csv'.

<img src='/assets/linear-function.png'/>
<br/>
Figure 1: Linear function defined by \\(\ref{eqn:target}\\).

# Implementing Linear Regression with Gradient Descent

It is common to use a machine learning package such as [scikit-learn](http://scikit-learn.org/stable/index.html) when performing machine learning. However, in this tutorial we will implement the algorithm for Linear Regression with Gradient Descent without the use of such a package. While this is slightly more work, it is instructive to code the algorithm ourselves. Although as the problems we address in the tutorial get more complex using a machine learning package will be a much more efficient strategy.

We do use two Python packages in the project; specifically, [pandas](https://pandas.pydata.org/) and [numpy](http://www.numpy.org/). Pandas is used for creating and manipulating data strucutes and numpy is used for scientific computation. They are both very commonly used packages in Data Science so if you aren't yet familiar with them, it would be good to look into them a bit.

All the code for the project is contained in a single file; 'linear\_regression.py'. The aptly named 'main' function is the main function of the code; it controls reading data, executing multiple iterations of gradient descent, periodically outputting cost calculations to screen and finally writing a file with the predicitons. The remainder of the code contains a class for reading data as well as a number of functions for:

* performing an iteration of gradient descent,
* calculating the cost function,
* calculating the hypothesis function.

The code for these three functions is shown below with some small rearrangement for clarity.

```python
# The array theta has the shape [1, 3]. That is a single row 
# with [theta0, theta1, theta2].
# 
# The array features has the shape [-1, 2], where the -1 in 
# the shape is used to denote an unknown number of rows. 
# This number will be dictated by number of samples in our 
# data set.
#
# The array labels has the shape [-1], where again the -1 
# is a placeholder that is indicative of the number of 
# samples in the data set.

import numpy as np

def hypothesis(theta, features):

    m = np.matmul(features, theta[1:, :])
    hyp = np.add(theta[0, :], m)
    return hyp

def cost(theta, features, labels):

    s = np.subtract(hypothesis(theta, features), labels)
    p = np.power(s, 2.0)
    c = 0.5*np.sum(p)
    return c

def gradient_descent(theta, features, labels, numSamples):

    # alpha is the learning rate
    alpha = np.power(10.0, -2)

    # incorporating the 2/N term from our equations 
    # into the alpha term for convenience
    alpha = 2*np.divide(alpha, np.shape(features)[0])

    ntheta = np.matrix([[0.0], [0.0], [0.0]])

    hyp = hypothesis(theta, features)

    ntheta[0, :] = theta[0, :] - \
      alpha * np.sum(np.subtract(hyp, labels))
    ntheta[1, :] = theta[1, :] - \
      alpha * np.sum(np.multiply(np.subtract(hyp, labels), \
                                 np.reshape(features[:, 0], \
                                            (numSamples, 1))))
    ntheta[2, :] = theta[2, :] - \
      alpha * np.sum(np.multiply(np.subtract(hyp, labels), \
                                 np.reshape(features[:, 1], \
                                            (numSamples, 1))))

    return ntheta
```

# Results

In the case of this specific experiment we knew in advance the target function \\(h(X)\\) defined by \\(\ref{eqn:target}\\). As a result we can test the efficacy of our model by comparing the actual values for \\((\theta\_{0}, \theta\_{1}, \theta\_{2})\\) with our learned values. From one run of the code the reported values for \\((\theta\_{0}, \theta\_{1}, \theta\_{2})\\) were \\((4.68178075\cdot{}10^{-17}, 5.00000000\cdot{}10^{-1}, 5.00000000\cdot{}10^{-1})\\). The values of \\(\theta\_{1}\\) and \\(\theta\_{2}\\) are exact; at least up to the level of accuracy reported by the program. The value \\(\theta\_{0}\\) isn't exactly 0 as we expected it to be. However, \\(4.68178075\cdot{}10^{-17}\\) is very, very small. To put it in context, if \\(4.68178075\cdot{}10^{-17}\\) expressed a probability, it would be a probablility approximately 100,000,000 times less likely than winning the main prize US Powerball or European Euro Millions Lotteries - and there are many of us who know all too well how unlikely that is.

# Wrap up

In this tutorial we demonstrated how Linear Regression with Gradient Descent could be applied. However, it was to a very simplified data set. Specifically, the data set was formed from a well known and very simple mathematical function. In reality, it would be extremely unlikely to have a data set that could be modeled by a single function for all samples. When we do have more complex data there are issues that we must address that weren't a concern in this case, e.g. model undefitting or overfitting. We will look at these issues in a future post.


