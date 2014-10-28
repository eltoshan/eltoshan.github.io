---
layout: post
title: "Multi-Armed Bandit for Risk Revenue"
tags: [multi-armed bandit, machine learning, bayesian methods]
date: 2014-10-28T10:44:53-04:00
---

This is an exercise to demonstrate the application of multi-armed bandit algorithm for maximizing risk revenue. Specifically, given finite resources and several potential missing CC's to go after, we want to find a process for allocating our chart reviews so that it maximizes the overall boost to risk scores.

The classical approach is to go after all of the types of charts with potentially missing CC's equally, and then perform A/B testing afterwards to determine the missing CC that had the highest conversion rate and had the biggest boost to risk score. The result is that a large number of the chart reviews would have been spent on charts that did not provide the optimal boost to risk scores.

The Bayesian bandit algorithm outlined below is a process that balances the trade-off between exploiting the currently "most profitable" option, and exploring the other options that may prove to be potentially even more profitable.

On top of the classical Multi-Armed Bandit with binary outcomes, we are going to incorporate a reward function that is dependent on the specific CC.

### Basics of bandits

From a Bayesian point of view, bandits with binary outcomes $k$ can be easily modeled using the beta-binomial predictive distribution. That is, if  

$$ k|n,p \sim Binom(n,p) $$

where $ Binom(n,p) $ is the binomial distribution, and $p$ is a random variable with a beta distribution,  

$$ \pi(p|\alpha,\beta) = Beta(\alpha,\beta) $$

And the posteriod predictive distribution for the probability of success $p$, given $n$ trials and $k$ successes, is  

$$ p|n,k,\alpha,\beta \sim Beta(\alpha + k, \beta + n - k) $$

### Modeling our bandits

So we can exploit the properties above and model our bandits using the beta-binomial preditive distribution. Python implementation below:  


{% highlight python linenos %}
%matplotlib inline
import numpy as np
from pymc import rbeta

np.random.seed(123123)
rand = np.random.rand
"""
class representing bandit machines.
parameters:
    p_array: (n,) numpy array between 0 and 1.
methods:
    pull( i ): binary result of ith bandit.
"""

class Bandits(object):
    def __init__(self, p_array, r_array):
        self.p = p_array
        self.r = r_array
        self.optimal = np.argmax(p_array)
    def pull(self, i):
        # i is which arm to pull
        return rand() < self.p[i]
    def __len__(self):
        return len(self.p)

"""
Multi armed bandit with online reinforcement learning.
parameters:
    bandits: from Bandit class with .pull method.
methods:
    sample_bandits(n): sample and train on n pulls.
attributes:
    N: cumulative number of sampels
    choices: (N,) array of choice history
    bb_score: (N,) array of score history
"""

class BayesianStrategy(object):
    def __init__(self, bandits):
        self.bandits = bandits
        n_bandits = len(self.bandits)
        self.wins = np.zeros(n_bandits)
        self.trials = np.zeros(n_bandits)
        self.N = 0
        self.choices = []
        self.bb_score = []

    def sample_bandits(self, n=1):
        bb_score = np.zeros(n)
        choices = np.zeros(n)

        for k in range(n):
            # sample from the bandit's priors, then select largest sample * reward
            choice = np.argmax( rbeta( 1 + self.wins, 1 + self.trials - self.wins) * bandits.r)
            # sample the chosen bandit
            result = self.bandits.pull( choice )
            # update priors and score
            self.wins[ choice ] += result
            self.trials[ choice ] += 1
            bb_score[ k ] = result
            self.N += 1
            choices[ k ] = choice

        self.bb_score = np.r_[ self.bb_score, bb_score ]
        self.choices = np.r_[ self.choices, choices ]
        return
{% endhighlight %}

### Visualizing the learning

Suppose that we have 3 different potentially missing CC's each with different impact on a member's risk score, and an unknown fraction of the missing CC's can be found in the chart reviews, we want to identify the highest impact missing CC to go after first in order to maximize risk revenue.

In the example below, the 3 missing CC's have hidden probabilities for being found on chart reviews of 10%, 30%, and 50%, and each boost the risk score of a member by 5, 4, and 1 respectively. The hidden probabilities are not observed and thus are not known to us. The algorithm starts by allocating the first chart review with equal probability to all 3 options. On each subsequent chart, the results from the previous charts are taken into account to update our beliefs of the missing CC's, and the next chart is chosen proportionally with respect to the expected increase in risk score.

Our posterior beliefs about the probability of finding each missing CC on the chart reviews is plotted after various number of charts.  The process gravitates to the CC with the greatest expected reward.  It's important to note that the Multi-Armed Bandit process is an optimization process, and we do not need to make exact inferences about the hidden probabilities of each of the CC's.

{% highlight python linenos %}
%matplotlib inline
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import scipy.stats as stats

figsize(11.0, 10)
beta = stats.beta
x = np.linspace(0.001, .999, 200)

def plot_priors(bayesian_strategy, prob, lw=3, alpha=0.2, plt_vlines=True):
    # plotting function
    wins = bayesian_strategy.wins
    trials = bayesian_strategy.trials
    for i in range(prob.shape[0]):
        y = beta(1 + wins[i], 1 + trials[i] - wins [i])
        p = plt.plot(x, y.pdf(x), lw=lw)
        c = p[0].get_markeredgecolor()
        plt.fill_between(x, y.pdf(x), 0, color=c, alpha=alpha,
                         label="underlying probability: %.2f" % prob[i])
        if plt_vlines:
            plt.vlines(prob[i], 0, y.pdf(prob[i]),
                       colors=c, linestyles="--", lw=2)
        plt.autoscale(tight="True")
        plt.title("Posteriors After %d chart" % bayesian_strategy.N +
                  "s" * (bayesian_strategy.N > 1))
        plt.autoscale(tight=True)
    return

hidden_prob = np.array([0.1, 0.3, 0.5])
reward_scalar = np.array([5.0, 4.0, 1.0])
bandits = Bandits(hidden_prob, reward_scalar)
bayesian_strat = BayesianStrategy(bandits)

draw_samples = [1,1,3,10,10,25,50,100,200,600]

for j, i in enumerate(draw_samples):
    plt.subplot(5, 2, j + 1)
    bayesian_strat.sample_bandits(i)
    plot_priors(bayesian_strat, hidden_prob)
    plt.autoscale(tight=True)
plt.tight_layout()
{% endhighlight %}

![bandit_learning](/assets/Bayesian_Multi_Armed_Bandit_files/Bayesian_Multi_Armed_Bandit_10_0.png)

What we see above is that our beliefs about the 3 different options start in a neutral and uninformative state, with a flat prior. After each chart is read, we update our beliefs about the probability of identifying a missing condition on each type of chart. As the algorithm progresses, we become more and more certain that the type of charts in green have the best reward. A sanity check multiplying the hidden probability by the reward scalar confirms that our belief is indeed true.

### Reward and regret

Since we didn't know which CC was optimal before starting the process, we undershot the best case performance. We can define the notion of Regret as the difference between the total reward had we picked the optimal CC to begin with, and the actual reward from the strategy:

$$ R_{T} = \sum\limits_{i=1}^T (w_{optimal} - w_{Bandit_{i}}) $$  

Where $ R\_{T} $ is the total regret, $ w\_{optimal} $ is our winning for one arm had we picked the optimal arm, and $ w\_{Bandit\_{i}} $ is our winning for the chosen arm.

We can visualize the regret of the Multi-Armed Bandit strategy compared to the standard case where we allocate resources equally. Lower the regret indicates better performance.

{% highlight python linenos %}
figsize(11.0, 5)

# define regret
def regret(probabilities, rewards, choices):
    w_opt = (probabilities * rewards).max()
    return (w_opt - (probabilities * rewards)[choices.astype(int)]).cumsum()

choices_std = np.tile( np.array([0, 1, 2]), 333)

regret_MAB = regret(hidden_prob, reward_scalar, bayesian_strat.choices)
regret_std = regret(hidden_prob, reward_scalar, choices_std)


plt.plot(regret_MAB, label="Bayesian MAB", lw=3)
plt.plot(regret_std, label="Equal Allocation", lw=3)

plt.title("Total Regret of Bayesian Bandits Strategy vs. Equal Allocation")
plt.xlabel("Number of charts")
plt.ylabel("Regret after $n$ charts");
plt.legend(loc="upper left");
{% endhighlight %}

![regret](/assets/Bayesian_Multi_Armed_Bandit_files/Bayesian_Multi_Armed_Bandit_14_0.png)

### Next steps

Reward payoffs for chart reviews often is not constant, due to comorbidities in conditions we may find multiple missing conditions on one chart. An approach to this problem is to apply a function on the reward functions based on the correlation of comorbidities. This of course introduces a calculation complexity where the posterior predictive distribution may not be able to be solved analytically.


<br>
