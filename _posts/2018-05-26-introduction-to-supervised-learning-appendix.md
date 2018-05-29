---
title:  "Introduction to Supervised Learning - Appendix"
date:   2018-05-26 12:00:00 +0000
categories: [ Machine Learning ]
---

$$
\begin{equation}
H = \left\lbrack 
\begin{array}{cc}
\frac{\partial^{2}J}{\partial\theta_{0}^{2}} \frac{\partial^{2}J}{\partial\theta_{0}\partial\theta_{1}} \\ 
\frac{\partial^{2}J}{\partial\theta_{1}\partial\theta{0}} \frac{\partial^{2}J}{\partial\theta_{1}^{2}} 
\end{array} 
\right\rbrack
\end{equation}
$$

Recall the definition of \\(J(\theta)\\) that we used in [Introduction to Supervised Learning]({ post_url 2018-05-19-introduction-to-supervised-learning }):

$$
\begin{equation}
J(\theta) = \frac{1}{N} \sum_{i}^{N} (h_{\theta}(X^{(i)}) - y^{(i)})
\end{equation}
$$


$$
\begin{equation}
h_{\theta}(X^{(i)}) = \theta_{0} + \theta_{1} \cdot X_{1}^{(i)}
\end{equation}
$$
