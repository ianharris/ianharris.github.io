---
title:  "Introduction to Machine Learning"
date:   2018-05-13 12:00:00 +0000
categories: [ Machine Learning ]
---

# Introduction to Machine Learning


Machine Learning (ML) is the most widely recognized portion of the more general field of Data Science. Following a number of high profile succseses (e.g. ad recommender systems, self-driving cars, improvements in speech recognition systems) machine learning has become an extremely popular technology. This in turn has led to new opportunities to work in the area of machine learning. However, if like me, you haven't come from an education that explicitly taught you machine learning, you'll need to do some catching up to get involved in the industry. This series of posts will cover an introduction to some of the main areas of machine learning. This post aims to give a broad overview of the field and introduce some of the terminology that will be used for future posts.

# Terminology

Let's start by introducing terminology that we will use throughout the series. When working in machine learning we will be processing data sets. A Data Set is a collection of data instances. A data instance (sometimes referred to as a member of the data set) is a grouping of data that refers to a single entity; for example, in a data set comprised of house purchase data, an instance would be all data referring to a single house (see the table below for an example of a house purchase data set). Each data instance will contain a collection of properties/characteristics that represent the entity as well as a collection of observations/outcomes. If we return to the example of house purchase data, the properties/characteristics could include things like the number of bedrooms in the house, the square footage of the house, the house's location, etc. The observations/outcomes would include only a single item in this case; specifically, the purchase price of the house. In machine learning terms we generally refer to the properties/characteristics of a data set as features and the observations/outcomes as labels. Furthermore, we generally represent the set of features by X and the set of labels as y when using mathematical notation.


| House ID | Number of Bedrooms | Square Footage (ft * ft) | Location | ... | Purchase Price (EUR) |
|:--------:|:------------------:|:------------------------:|:--------:|:---:|:--------------------:|
| 1        |  2                 | 1091                     | Dublin   | ... | 290000               |
| 2        |  3                 | 1287                     | Dublin   | ... | 345000               |
| ...      |  ...               | ...                      | ...      | ... | ...                  |
| N        |  2                 | 1085                     | Dublin   | ... | 301000               |


Now that we have an understanding of the fundamental terminology, we will explore some of the key types of machine learning.

# Types of Learning

Within machine learning there are two main types of learning. Specifically, 

* Supervised Learning,
* Unsupervised Learning.

There are other types of learning (for example, Reinforcement Learning) but these are more advanced and are better explained once we have covered Supervised Learning and Unsupervised Learning.

## Supervised Learning

Supervised learning assumes there exists a data set that consists of features and label. We further assume that a relationship exists between the features and labels; although, we don't need to know what this relationship is. A machine learning algorithm is applied to the data to try to 'learn' this relationship. 

As an example, supervised learning could be applied to the house purchase data set in order to learn a relationship between the features of the house data set and it's purchase price. In order to learn the relationship between houses features and prices we could apply an algorithm such as Linear Regression. We'll discuss particular algorithms and their implementation later in the series.

## Unsupervised Learning

Unsupervised learning assumes there exists a data set that consists of features but in this case we have no corresponding labels. In some applications (e.g. clustering) we will further assume that a relationship exists between items in the data set such that the features can be used to group these common items. As in the case of supervised learning, we don't need to know what this relationship is, nor do we need to know what the groupings are.

To demonstrate unsupervised learning let's consider clustering of a customer base. In this problem you have a large amount of data on your customers and you want to group them to understand how you can better serve them. You don't know before hand what these groups are but you aim to learn these groupings using an unsupervised learning algorithm such as K-means clustering.

In the [next post]({% post_url 2018-05-19-introduction-to-supervised-learning %}) we will explore the fundamentals of Supervised Learning in more detail. Getting an understanding of what is required to learn a relationship between a data set's features and labels.

