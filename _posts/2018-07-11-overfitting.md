---
title:  "Overfitting Your Model"
date:   2018-07-11 12:00:00 +0000
categories: [ Machine Learning ]
---

# Overfitting Your Model

In this post we'll look at a problem facing machine learning models, overfitting our model. We alluded briefly to overfitting in our [last post]({% post_url 2018-07-03-train-test-validate %}) but will describe it in greater detail in this post.

# Overfitting

Overfitting occurs when a model fits the training data set that was used to produce it too precisely - due to some noise or detail that occurs in the training set but not in other data samples. When this occurs the model typically doesn't perform well when trying to predict on new data; data scientists commonly refer to such a model as one that doesn't generalize well. Clearly, a model that performs well on a training data set but can't generalize to perform well on unknown data isn't very useful - as the aim of such a model is to provide insights relative to that unknown data. But how do we know when our model has overfitted and how can we prevent this?

## Identifying Overfitting

The primary method for identifying overfitting is to compare the cost function calculated on the training and validation sets throughout training. When overfitting occurs the training cost will continuously decrease while the validation cost may initially decrease but will diverge from the training cost at the point where the model starts to overfit. Figure 1 below shows an example of a comparison of training cost and validation cost plotted against training iteration when the model is overfitting.

<img src='/assets/overfitting.png'/><br/>
Figure 1: An example of a comparison of training cost and validation cost plotted against training iteration when the model is overfitting.

## Preventing Overfitting

There are a number of methods that can prevent overfitting from occurring. The following is an overview of some of the most common examples.

### Simpler Model

Overfitting might occur because the algorithm we have chosen to model our data is too complex. A classic demonstration of this is consider trying to fit a higher order polynomial model to a data set that is actually linear. Figure 2 below depicts such a scenario. In the figure we have a labeled data set \\((X\_{1}, y)\\) with six samples. Plotted alongside the data set are curves depicting a linear fit (in green) and a fifth order polynomial (in red). 

The linear fit line does not pass through the center of any of the points. In contrast, the fifth order polynomial fit passes through each point exactly. This would seem to suggest that the polynomial fit is the better model for this data set. However, this is not the case. In this case the fifth order polynomial has actually overfit the training set. 

Note: actually identifying that the model has overfit would require an inspection as outlined in [Identifying Overfitting](#identifying-overfitting) above. However, Figure 2 does give a nice visualization of what overfitting looks like.

<img src="/assets/fifth-order-overfit.png"/><br/>
Figure 2: An example of a data set correctly modelled by a linear fit and incorrectly modelled by a fifth order polynomical fit.

### Regularization

Regularization refers to a set of methods that prevent overfitting by directly restricting the weights learned from a specific training set. Two of the most common regularization techniques are L1-norm and L2-norm regularization. In this post we will look at L2-norm regularization (abbreviated to L2 regularization) as an example.

_Note: There are other regularizaition techniques in machine learning that are related to specific areas of machine learning; for example, Dropout regularization in Deep Neural Networks. We will discuss these techniques when we look at the corresponding area of machine learning._

L2 regularization is achieved by adding a regularization term to our cost function. If we have some existing cost function \\(J(\theta)\\) then the L2 regularized cost function is defined by:

$$
\begin{equation}
J_{reg}(\theta) = J(\theta) + \lambda \cdot \sum_{i}^{N} \theta_{i}^{2}\label{eqn:regcost}\tag{1}
\end{equation}
$$

where \\(\lambda\\) is a parameter that controls the strength of regularization. The choice of \\(\lambda\\) is important. If \\(\lambda = 0\\) then no regularization occurs and our model may overfit. If \\(\lambda \gg 1\\) then the regularization will drive the \\(\theta_{i}\\) towards zero and our model may underfit - we will discuss underfitting in our next post.

The inclusion of a regularization term ensures that the model weights \\(\theta\_{i}\\) remain small; to see this consider that the effect of large weights \\(\theta\_{i}\\) would lead to a large cost which our optimizer is trying to minimize. This restriction of the model weights prevents the model from disproportionally favouring specific features in the learned model and will prevent the model being skewed by features that have particular importance in the training set only.

If we return to our earlier post [introducing supervised learning]({% post_url 2018-05-19-introduction-to-supervised-learning %}) and add L2 regularization to the equations defining the updates to the model weights \\((\theta\_{0}, \theta\_{1})\\), Linear Regression with a Gradient Descent optimizer the regularized equations would be:

$$
\begin{eqnarray}
\theta_{0, j+1} & := & \theta_{0, j} - \alpha \cdot \frac{1}{N} \sum_{i=1}^{N} (\theta_{0, j} + \theta_{1, j}\cdot{}X_{1}^{i} - y^{(i)}) \label{eqn:updtht0}\tag{2} \\ 
\theta_{1, j+1} & := & \theta_{1, j} - \alpha \cdot \left\lbrace \frac{1}{N} \sum_{i=1}^{N} (\theta_{0, j} + \theta_{1, j}\cdot{}X_{1}^{i} - y^{(i)})\cdot X^{(i)} + \lambda \cdot \theta_{i} \right\rbrace \label{eqn:updtht1}\tag{3}
\end{eqnarray}
$$

We will use equations \\(\ref{eqn:updtht0}\\) and \\(\ref{eqn:updtht1}\\) in a future post demonstrating L2 regularization.

### Ensemble Methods

Ensemble methods involve training multiple models with the same algorithm and then combining the models into one for a prediction. A good example of an ensemble method is Random Forests - we haven't discussed Random Forests yet so we will describe them at the end of this section. Random Forests are an example of a group of ensemble methods known as Bagging (short for Boostrap Aggregating) methods. Bagging methods consist of creating \\(m\\) training subsets from the original training set by uniformly sampling with replacement - 'with replacement' means that each subset is sampled from the entire original training set. Each of these subsets is then used to train a model with the same learning algorithm. A prediction for new data is then achieved by averaging the individual model predictions for regression problems, or by a voting method for classification problems.

Let's now return to a description of Random Forests. Random Forests are a Bagging method where the learning algorithm is a Decision Tree. A Decision Tree is a tree like structure, where each node in the tree corresponds to a feature of the data set. At each node, the training data set is split into two parts based on the node's feature. Each new part passes to a new node, which itself will have a feature to again split on, until finally the data reaches a "Terminal Node"; a node where splitting does not occur. Figure 3 below shows and example of a Decision Tree.

<img src="/assets/decision-tree.png" /><br/>
Figure 3: An example of a Decision Tree. The nodes of the tree are labelled with the feature on which the split or "Terminal Node" if the node is a terminal node of the tree.

### More Training Data

Since the noise and/or details of the specific data set are being overfit, obtaining more data samples and adding them to the training set may reduce overfitting. Of course, this solution only works if the additional data does not suffer from the same noise and/or details as the original training set. It should also be noted that while in theory this is an easy solution, in practice it may not be. While having more data to train on is beneficial to a machine learning problem, in reality, additional training data might just not be available; particularly in supervised learning where data must be labelled to be useful.

### Augmented Training Data

In some cases it is possible to augment your training data set. Augmenting your data set involves modifying existing samples from your data set to create 'new' samples. It must be noted that augmentation won't suit all machine learning tasks. It is commonly seen in computer vision problems where augmentation such as illumination normalization will provide a new sample but without modifying the content captured by the image. Clearly, great care should be taken when applying augmentation as incorrectly augmenting your data could introduce artificial relationships between features and hence cause your algorithm to learn a relationship that doesn't really exist. 

### Early Stopping

Another method to solve the overfitting problem is known as early stopping. And early stopping means exactly that. We stop the training the model at the point at which it begins to overfit. Practically, speaking this involves monitoring the training and validation cost as we train and when they start to diverge we interrupt the learning. The exact conditions on what constitutes sufficient divergence to interrupt training will be based on the specific use case and data set. 

Of course, the early stopping method assumes that when we stop learning that the model will have learned enough to perform well on the test set. If the learned model doesn't perform sufficiently well, then an alternate strategy would be required.

# Wrap up

And that's it. An overview of the problem of overfitting in machine learning. The problems of a model that has overfit its training data cannot be overstated. With that in mind, all aspiring data scientists should become familiar with identifying and mitigating overfitting in a variety of machine learning scenarios.

Up next we'll discuss the problem of underfitting a data set.


