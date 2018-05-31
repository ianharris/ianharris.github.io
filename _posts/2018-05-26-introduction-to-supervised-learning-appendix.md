---
title:  "Introduction to Supervised Learning - Appendix"
date:   2018-05-26 12:00:00 +0000
categories: [ Machine Learning ]
---

# Appendix

In the post [Introduction to Supervised Learning]({% post_url 2018-05-19-introduction-to-supervised-learning %}) we claimed to be able to prove the convexity of a Mean Squared Error function with a linear hypothesis function. That is with a loss defined by equation \\(\ref{eqn:mse}\\) and hypothesis function \\(\ref{eqn:hyp}\\) below.

$$
\begin{equation}
J(\theta) = \frac{1}{N} \sum_{i}^{N} (h_{\theta}(X^{(i)}) - y^{(i)})\label{eqn:mse}\tag{1}
\end{equation}
$$

$$
\begin{equation}
h_{\theta}(X^{(i)}) = \theta_{0} + \theta_{1} \cdot X_{1}^{(i)}\label{eqn:hyp}\tag{2}
\end{equation}
$$

A method to prove the convexity of a function of multiple variables is to show that the Hessian matrix is positive semidefinite.

## Hessian Matrix

The Hessian matrix is an \\(n\times{}n\\) matrix that consists of the second order partial derivatives of the function with respect of each of the variables; where \\(n\\) is the number of independent variables of the function. That is:

$$
\begin{equation}
H = \left\lbrack 
\begin{array}{cccc}
\frac{\partial^{2}f}{\partial{}x_{0}^{2}} & \frac{\partial^{2}f}{\partial{}x_{0}\partial{}x_{1}} & \cdots & \frac{\partial^{2}f}{\partial{}x_{0}\partial{}x_{n}}\\ 
\frac{\partial^{2}f}{\partial{}x_{1}\partial{}x_{0}} & \frac{\partial^{2}f}{\partial{}x_{1}^{2}} & \cdots & \frac{\partial^{2}f}{\partial{}x_{1}\partial{}x_{n}}\\ 
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^{2}f}{\partial{}x_{n}\partial{}x_{0}} & \frac{\partial^{2}}{\partial{}x_{n}\partial{}x_{1}} & \cdots & \frac{\partial^{2}f}{\partial{}x_{n}^{2}} 
\end{array}
\right\rbrack
\end{equation}
$$

Equations \\(\ref{eqn:mse}\\) and \\(\ref{eqn:hyp}\\) define a function, \\(J(\theta)\\) with two independent variables \\(\theta\_{0}\\) and \\(\theta\_{1}\\). Hence, the Hessian matrix for equations \\(\ref{eqn:mse}\\) and \\(\ref{eqn:hyp}\\) is:

$$
\begin{eqnarray}
H  &=&  \left\lbrack 
\begin{array}{cc}
\frac{\partial^{2}J}{\partial\theta_{0}^{2}} \frac{\partial^{2}J}{\partial\theta_{0}\partial\theta_{1}} \\ 
\frac{\partial^{2}J}{\partial\theta_{1}\partial\theta_{0}} \frac{\partial^{2}J}{\partial\theta_{1}^{2}} 
\end{array}\right\rbrack \\
&=& \left\lbrack\begin{array}{cc}
\sum_{i}^{N} 1 & \sum_{i}^N X_{1}^{(i)} \\
\sum_{i}^N X_{1}^{(i)}  & \sum_{i}^N \left(X_{1}^{(i)}\right)^{2} 
\end{array}\right\rbrack
\label{eqn:hsscost}\tag{3}
\end{eqnarray}
$$

## Positive Semidefinite

A matrix \\(A\\) is positive semidefinite if:

$$
\begin{equation}
v^T\cdot{}A\cdot{}v \ge 0\label{eqn:psd}\tag{4}
\end{equation}
$$

where \\(v \in \mathbb{R}^{n}\\). To check if our Hessian matrix is positive semidefinite we use the Hessian matrix defined in \\(\ref{eqn:hsscost}\\) in equation \\(\ref{eqn:psd}\\) to give:

$$
\begin{eqnarray}
v^T\cdot{}A\cdot{}v & = & \left\lbrack{}\begin{array}{cc}a & b\end{array}\right\rbrack \cdot \left\lbrack\begin{array}{cc}
\sum_{i}^{N} 1 & \sum_{i}^N X_{1}^{(i)} \\
\sum_{i}^N X_{1}^{(i)}  & \sum_{i}^N \left(X_{1}^{(i)}\right)^{2} 
\end{array}\right\rbrack
\left\lbrack \begin{array}{c} a \\ b \end{array}\right\rbrack

\\

& = &\left\lbrack\begin{array}{cc}
a\cdot\sum_{i}^{N} 1 + b\cdot\sum_{i}^{N}X_{1}^{(i)} & a\cdot\sum_{i}^{N}X_{1}^{(i)} & b\cdot\sum_{i}^{N}\left(X_{1}^{(i)}\right)^{2}
\end{array}\right\rbrack
\left\lbrack \begin{array}{c} a \\ b \end{array}\right\rbrack

\\

& = & a^{2}\sum_{i}^{N}1 + a\cdot{}b\sum_{i}^{N}X_{1}^{(i)} + a\cdot{}b\sum_{i}^{N}X_{1}^{(i)} + b^{2}\sum_{i}^{N}\left(X_{1}^{(i)}\right)^{2} \\
& = & a^{2}\sum_{i}^{N}1 + 2\cdot{}a\cdot{}b\sum_{i}^{N}X_{1}^{(i)} + b^{2}\sum_{i}^{N}\left(X_{1}^{(i)}\right)^{2} \\ 
& = & a^{2}\sum_{i}^{N}1 + 2\cdot{}a\cdot{}b\sum_{i}^{N}X_{1}^{(i)} + b^{2}\sum_{i}^{N}\left(X_{1}^{(i)}\right)^{2} \\
& = & \sum_{i}^{N} a^{2} + 2\cdot{}a\cdot{}b\cdot{}X_{1}^{(i)} + b^{2}\left(X_{1}^{(i)}\right)^{2} \\
& = & \sum_{i}^{N} \left(a + b\cdot{}X_{1}^{(i)}\right)^{2} \ge 0


\end{eqnarray}
$$

Hence our Hessian defined by equation \\(\ref{eqn:hsscost}\\) is positive semidefinite. Thus our Cost function defined by equations \\(\ref{eqn:mse}\\) and \\(\ref{eqn:hyp}\\) is convex.

