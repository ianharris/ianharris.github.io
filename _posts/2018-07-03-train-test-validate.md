---
title:  "Real World Datasets - Train, Test, Validate"
date:   2018-07-03 12:00:00 +0000
categories: [ Machine Learning ]
---

# Real World Datasets - Train, Test, Validate

In our [last post]({% post_url 2018-06-11-introduction-to-linear-regression %}) we demonstrated how to train a Linear Regression model using Gradient Descent. In that post we created a data set from a known mathematical function and demonstrated that we could 'learn' that function very accurately. This demonstration, though effective, was based on a very simple data set and not one that we could reasonably expect to see in a real world application of machine learning. As a result of this simplicity, how we handled the data was also simplified. Specifically, we used the entire data set for training. This isn't typical for machine learning problems as we usually reserve some data for testing and validation tasks. In this post we'll discuss why we would use test and validation sets, which will give us an insight into why we didn't need them for our demonstration.

# Training Data, Test Data and Validation Data

Let's begin by discussing the motivation for each of our data sets. 

The training data set needs little explanation. Machine learning requires a set of data to learn from. In our Linear Regression demonstration, the training set was used to learn the relationship between our set of features \\((X\_{0}, X\_{1})\\) and our set of labels \\(y\\).

The test data set is used to test the relationship we learn. Without a test data set we run the risk of 'overfitting' our model on our training set. We'll discuss the problem of overfitting in a future post but at a high level overfitting occurs when the relationship the model learns performs well in predicting labels for the training data set but performs poorly when predicting labels for new data. Obviously if our model doesn't provide accurate predictions on data that isn't in our training set it isn't very useful.

The validation set is used to help tune parameters of our algorithm. Many algorithms have parameters that can be set independent of our data set; e.g. the learning rate \\(\alpha\\) in our Linear Regression demonstration. These parameters can be tuned to improve model performance. Tuning the parameters is achieved by training with a specific set of parameters and then assessing the effect of that parameter set using the validation set.

At first it may seem like the validation set is redundant; that the test set could be used for tuning. But this is not the case. If an algorithm's parameters need to be tuned, it is vital to have a separate validation set. Consider what would happen if we used the test set for tuning. We would train our model a number of times using various parameter sets and then choose the parameter set that had the best model performance when evaluated against test set. But this selection will ensure the model performs well specifically on the test set (as that is how we selected the parameters) and so the model may appear to be more performant than it actually is when we assess it.

# Simplified for our Demonstration

In our Linear Regression with Gradient Descent example we didn't use a test or validation set. We instead used all our data for training. Let's look at why. In the case of both data sets, the answer comes down to simplicity.

We'll consider the use of a test set first. Usually we will use our test set to assess the accuracy of our trained model. However, in the case of our demonstration we knew the target function \\(h(X)\\). Hence, assessing the accuracy of the model could be achieved by simply comparing the final hypothesis function \\(h\_{\theta}(X)\\) to the known target function \\(h(X)\\). 

We next consider the use of a validation set. The only parameter we could have tuned in our demonstration was the learning rate \\(\alpha\\). As the demonstration was such a simple case any reasonable choice of learning rate would have guaranteed the model would learn the relationship. With that in mind, we selected the learning rate without the use of a validation set.

# Sourcing the Datasets

Now that we understnd the functions of the three data sets (training, test and validate) we need to address the question of where we source each of them. When presented with a machine learning problem, a data scientist will usually have a single data set. It is the responsibility of the data scientist to then divide the data set into training, test and validation sets. Typically, the majority of the data set will go to the training data set, with a smaller portion being taken for testing and validation. There is no strict rule on the exact proportions of the splits as those proportions will be dependent on the data set and the problem being addressed. For someone trying to learn machine learning the lack of a clear rule isn't ideal. However, if you read on we'll discuss strategies for dividing up data in future posts; and we will try to present as diverse a set of strategies as possible.

# Up Next

In our next post we'll discuss two common problems faced in machine learning; namely overfitting and underfitting as well as methods to identify and solve both problems.

