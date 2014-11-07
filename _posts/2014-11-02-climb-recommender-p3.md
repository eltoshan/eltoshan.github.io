---
layout: post
title: "Building a Climbing Route Recommender - Part 3"
date: 2014-11-02T10:37:32-05:00
---

In [Part 2]({% post_url 2014-10-31-climb-recommender-p2 %}) of the series we implemented the Alternating Least Square with Weighted Regularization (ALS-WR) algorithm in Python. In this post, we will test out a distributed implementation in Apache Spark. The ALS-WR implementation in Spark takes advantage of some computational shortcuts as well as parallelization. The distributed algorithm achieves nearly linear speed up with the number of additional processors in the cluster.

We can spin up an on-demand cluster on Amazon EC2 using the spark-ec2 script provided in the Spark distribution.  Once the cluster is up and running the data can be uploaded to the persistent-hdfs created by the script. Another option is just to run Spark locally on one node.

Full implementation available on [GitHub](https://github.com/eltoshan/CF/blob/master/MPsparkALS.py).

Something interesting that became obvious to me is that the weighted regularization method has much better out-of-sample prediction than the vanilla ALS method, but since there can be very large penalties placed on very active users or very popular products, a large regularization parameter can lead to recommendations of very obscure products with very few high positive ratings. In the case of this experiment where I used the ratings of climbs in the New River Gorge, the best model used a regularization parameter of $0.30$, and the recommendations tended to favor 4 star routes with only a handful of votes over 3.7 star routes with hundreds of votes. I call this the hipster effect. Perhaps applying a TF-IDF style multiplier to the regularization parameter would strike a better balance of recommendations.

In practice, choosing the right regularization parameter is hotly debated, and sometimes can be more of an art than science. Below are the top 20 routes recommended to me using a regularization parameter of $0.10$.

{% highlight bash %}
Routes recommended for user 109673281:
 1: Dial 911 5.13a
 2: Narcissus 5.12a
 3: The Project 5.13c
 4: Shang 5.12+
 5: New Traditionalist 5.12b
 6: The Beckoning 5.12a
 7: Legacy 5.11a
 8: Lactic Acid Bath 5.12d
 9: Masterpiece Theatre 5.12d
10: Skylore Engine 5.13a
11: Coffindaffer's Dream 5.11b
12: Ly'n and Stealin' 5.12b
13: Under the Milky Way 5.11d
14: Blackhappy 5.12b
15: Sanctified 5.12d
16: Green Envy 5.12c
17: Scenic Adult 5.11c
18: Sloth 5.12c
19: Disturbance 5.11d
20: Massacre 5.13a
{% endhighlight %}

Without filtering to my appropriate climbing level, the recommendations are far too hard for the most part!