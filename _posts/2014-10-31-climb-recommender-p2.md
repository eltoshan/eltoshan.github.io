---
layout: post
title: "Building a Climbing Route Recommender - Part 2"
date: 2014-10-31T09:02:21-04:00
---

In [Part 1]({% post_url 2014-10-27-climb-recommender-p1 %}) of the series, we looked at gathering the data necessary for the recommender. In this post, we will implement the collaborative filtering recommender locally in Python.  The next post will cover distributed implementation in Apache Spark, so stay tuned.

### Collaborative Filtering Using Non-Negative Matrix Factorization

The general idea behind collaborative filtering using non-negative matrix factorization (NMF) is that the user/item/rating tuples can be represented using an $ m \times n $ matrix, where $m$ is the number of users and $n$ is the number of items, the values in the matrix are the ratings.

The intuition behind collaborative filtering is that similar users prefer similar products, and that we can predict the preference a given user has for a product using other similar users.  In the context of NMF, we can try to extract hidden preference features by decomposing the original matrix into an user matrix ($X$) and an item matrix ($Y$). $X$ is then an $ m \times f $ matrix, and $Y$ is an $ f \times n $ matrix, where $f$ is the number of hidden features.

![matrix_factorization](/assets/climb_recommender_files/matrix_factorization.png)

The goal is then to minimize the error of the predicted ratings from the factored matrices compared to the actual ratings:  

$$ \arg \min_{x*, y*}  \sum\limits_{u,i} w_{ui}(r_{ui} - x_u^Ty_i)^2 + \lambda\left(\sum\limits_{u} \|x_u\|^2 + \sum\limits_{i} \|y_i\|^2\right) $$

Where $w_{ui}$ is an indicator of whether the user rated the item.

One approach is to simply solve this using gradient descent, but to scale to a fast parallel algorithm we can use the Alternating Least Square (ALS) method, where we alternately minimize $X$ and $Y$ over many iterations until the error converges to a stable value. The especially nice property of ALS is that each iteration of $X$ and $Y$ can be solved analytically.

By taking the loss function above and setting the derivative with respect to $X$ to $0$, we obtain the solution for $X$:  

$$ X_u = (Y^TW^uY + \lambda I)^{-1}Y^TW^ur_u $$

Where $W^u$ is a diagonal matrix where the diagonal entries are $w_{ui}$.

Repeating the same for $Y$ we get:

$$ Y_i = (X^TW^iX + \lambda I)^{-1}X^TW^ir_i $$

With that laid out, we can construct our ALS algorithm.

{% highlight python linenos %}
import numpy as np
import pandas as pd

class ALSModel():

    def __init__(self, ratings, num_factors=20, num_iterations=10, reg_param=0.30):
        self.Q = ratings
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param

    def train_model(self):
        m = self.Q.shape[0]
        n = self.Q.shape[1]
        nRatings = np.count_nonzero(self.Q)

        W = self.Q>0.5
        W[W == True] = 1
        W[W == False] = 0
        W = W.astype(np.float64, copy=False)
        Nu = np.sum(W, axis=1)
        Ni = np.sum(W, axis=0)

        X = np.random.rand(m, self.num_factors) 
        Y = np.random.rand(self.num_factors, n)
        lambda_eye = self.reg_param * np.eye(self.num_factors)

        for ii in range(self.num_iterations):
            for u, Wu in enumerate(W):
                X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + max(1, Nu[u]) * lambda_eye,
                                       np.dot(Y, np.dot(np.diag(Wu), self.Q[u].T))).T
            for i, Wi in enumerate(W.T):
                Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + max(1, Ni[i]) * lambda_eye,
                                         np.dot(X.T, np.dot(np.diag(Wi), self.Q[:, i])))
            print "%dth iteration is completed" % (ii+1)
        return factoredModel(X, Y)

class factoredModel():

    def __init__(self, userMat, itemMat):
        self.X = userMat
        self.Y = itemMat

    def computeRMSE(self, ratings):
        Q = ratings
        nRatings = np.count_nonzero(ratings)
        W = ratings>0.5
        W[W == True] = 1
        W[W == False] = 0
        W = W.astype(np.float64, copy=False)
        Qhat = W * np.dot(self.X, self.Y)
        RMSE = np.sqrt(np.sum((W * (Q - Qhat)**2)) / nRatings)
        return Qhat, RMSE
{% endhighlight %}

Full Python script available on [GitHub](https://github.com/eltoshan/CF/blob/master/mf.py).

<br>