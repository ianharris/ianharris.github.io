---
title:  "Introduction to Supervised Learning"
date:   2018-05-19 12:00:00 +0000
categories: [ Machine Learning ]
---

# Introduction to Supervised Learning

In our last post [Introduction to Machine Learning]({% post_url 2018-05-13-introduction-to-machine-learning %}) we were introduced to Supervised Learning - an area of machine learning that focuses on learning a relationship between a set of features and a set of labels. In this post we will build on that knowledge to provide a more complete understanding of the area of Supervised Learning.

# Learning

A Supervised Learning algorithm will compare a set of features and labels many times to learn the relationship between them. These many comparisons allow the algorithm to iteratively improve its understanding of the underlying relationship. At a high level that iterative process follows the following structure:

1. The algorithm starts with a random guess of the relationship. <br/>
2. This relationship along with the featues is used to create a prediction of the labels. <br/> 
3. A measure of the error (known as the loss or cost) between the predicted and actual labels is then produced. <br/>
4. Based on this error an update is made to the relationship. <br/>
5. Repeat from step 2 until the relationship can sufficiently accurately (it will never be exact) predict the labels from the features.

But how does the algorithm know how to update the relationship in a way that guarantees improvement? To answer that question let's frame the problem more mathematically.

Mathematically speaking when we refer to learning the relationship, we mean that we want to learn a function that correctly maps the features of every instance to their corresponding labels. We call this function the target function and denote it by \\(h(X)\\), where \\(X\\) is the set of features in the data set. As we saw above, to learn \\(h(X)\\) we will iterate through a number of approximate relationships that will become increasingly better approximations of \\(h(X)\\). We call these approximate relationships hypothesis functions and denote them by \\(h\_{\theta}(X)\\) - where \\(\theta\\) represent the parameters that define the hypothesis function.

But we still don't understand how our hypothesis functions can be iterated towards the target function reliably. Why would successive approximations always get better? Why not become increasingly worse approximations? For completeness it should be noted that it would be incorrect to say that successive iterations are always better; in truth, it is actually that over many iterations there is a net improvement in approximation. This net improvement is provided by a clever use of a cost function and an optimization algorithm. Let's look at an overview of cost functions and optimization algorithms before taking a specific example to help illustrate.

## Cost/Loss Function

The Cost (sometimes referred to as the Loss) function, which we denote by \\(J(\theta)\\), is an error measure that represents the level of inaccuracy between the actual labels \\(y\\) and the predicted labels \\(y\_{pred}\\). The cost function is chosen so that there is a single global minimum (functions with this property are known as convex) when \\(h\_{\theta}(X) = h(X)\\). Hence, we can learn the target function by finding the global minimum of the cost function. Finding the global minimum can be achieved by applying an optimization algorithm to the cost function.

## Optimzation Algorithm

An optimization algorithm is an algorithm used to identify where maximum/minimum values appear in a function. In the case of supervised learning, we want to apply an optimization algorithm to identify the global minimum of the cost function.

# Exmaple: Linear Regression

To better illustrate how a cost function and an optimization algorithm can be used for supervised learning, we will look at the example Linear Regression using a Mean Squared Error cost function and Gradient Descent as the optimization algorithm.

## Mean Squared Error Cost

The Mean Squared Error (MSE) cost is defined as:

$$ 
\begin{eqnarray}
J(\theta) & = & \frac{1}{N}\sum_{i=0}^{N} (y^{(i)} - y_{pred}^{(i)})^2 \\
 & = & \frac{1}{N}\sum_{i=0}^{N} (y^{(i)} - h_{\theta}(X^{(i)}))^2\label{eqn:mse}\tag{1}
\end{eqnarray}
$$

where \\(N\\) is the number of instances in the data set, \\(y^{(i)}\\) is the label of the *i*-th instance in the data set, \\(y^{(i)}\_{pred}\\) is the predicted label for the *i*-th instance of the data set and \\(X^{(i)}\\) denotes the features of the *i*-th instance of the data set.

From equation \\(\ref{eqn:mse}\\) we note that \\(J(\theta)\\) has a minimum at \\(J(\theta) = 0\\) and at that minimum the hypothesis function is equal to the target function; to see this, recall that \\(y^{(i)} = h(X^{(i)})\\) and \\(y^{(i)}\_{pred} = h\_{\theta}(X^{(i)})\\). Furthermore, we can prove that \\(J(\theta)\\) is a convex function (see [appendix]({% post_url 2018-05-26-introduction-to-supervised-learning-appendix %})) meaning that the minimum is a global minimum.

<img src='/assets/mean-square-error-two-dims.png'/>
<br/>
Figure 1: Mean squared error for two dimensions.

## Gradient Descent

To optimize the mean squared error we will use gradient descent. Gradient descent finds a minimum of a function by following the gradient of the function until it reaches the minimum. To help illustrate this, let's consider Figure 1 above. Figure 1 shows an approximation of equation \\(\ref{eqn:mse}\\) where the hypothesis function, \\(h\_{\theta}(X)\\), is defined as:

$$
\begin{equation}
h_{\theta}(X^{(i)}) = \theta_{0} + \theta_{1} \cdot X^{(i)}\label{eqn:hyp}\tag{2}.
\end{equation}
$$

This hypothesis function can be used to perform a Linear Regression; we'll discuss more about what Linear Regression is in a future post. During the optimization process we are looking to find the \\((\theta\_{0, min}, \theta\_{1, min})\\) that will minimize \\(J(\theta)\\). Clearly, in the example shown in the figure that will be \\((\theta\_{0, min}, \theta\_{1, min}) = (0, 0)\\). However, that will not genearlly be the case.

To find this \\((\theta\_{0, min}, \theta\_{1, min})\\), let's suppose we start at some random initial values \\((\theta\_{0, 0}, \theta\_{1, 0})\\). The gradients of \\(J(\theta)\\) with respect to \\(\theta\_{0}\\) and \\(\theta\_{1}\\) at any point \\((\theta\_{0, j}, \theta\_{1, j})\\) will always point away from the global minimum. Hence, if we update the values of \\((\theta\_{0, j}, \theta\_{1, j})\\) with an increment proportional to the negative of the gradient, our update will be closer to the minimum \\((\theta\_{0, min}, \theta\_{1, min})\\). Expressed as a set of equations we have: 

$$
\begin{eqnarray}
\theta_{0, j+1} & := & \theta_{0, j} - \alpha \cdot \frac{\partial{}J}{\partial\theta_{0}}(\theta_{0, j}, \theta_{1, j}) \label{eqn:updtht0}\tag{3} \\ 
\theta_{1, j+1} & := & \theta_{1, j} - \alpha \cdot \frac{\partial{}J}{\partial\theta_{1}}(\theta_{0, j}, \theta_{1, j}) \label{eqn:updtht1}\tag{4}
\end{eqnarray}
$$

where \\(\alpha\\) is a constant (known as the learning rate) that controls how quickly the algorithm descends towards the minimum. Using equations \\(\ref{eqn:mse}\\) and \\(\ref{eqn:hyp}\\) we can expand equations \\(\ref{eqn:updtht0}\\) and \\(\ref{eqn:updtht1}\\) to give:  

$$
\begin{eqnarray}
\theta_{0, j+1} & := & \theta_{0, j} - \alpha \cdot \frac{1}{N} \sum_{i=1}^{N} (\theta_{0, j} + \theta_{1, j}\cdot{}X_{1}^{i} - y^{(i)}) \label{eqn:updtht0final}\tag{5} \\ 
\theta_{1, j+1} & := & \theta_{1, j} - \alpha \cdot \frac{1}{N} \sum_{i=1}^{N} (\theta_{0, j} + \theta_{1, j}\cdot{}X_{1}^{i} - y^{(i)})\cdot X^{(i)} \label{eqn:updtht1final}\tag{6}
\end{eqnarray}
$$

Equations \\(\ref{eqn:updtht0final}\\)  and \\(\ref{eqn:updtht1final}\\) can be used to update \\((\theta_{0, j}, \theta_{1, j})\\) at each iteration to try to learn \\((\theta_{0, min}, \theta_{1,min})\\) and thus the target function \\(h(X)\\).

<hr/>

In the [next post]({% post_url 2018-06-11-introduction-to-linear-regression %}) we will look at an application of Supervised Learning; specifically, Linear Regression using Mean Squared Error and Gradient Descent. This will give us a much better view of a practical application of Supervised Learning as well as indroduce us to some of the common machine learning libraries.
