# bandit.py
# parsons/25-mar-2017
#
# A program to run experiments with epsilon-greedy action selection for n-armed
# bandits.
#
# This is an implementation of the bandit problem discussed by Sutton and Barto. The
# problem is stated here:
#
# http://incompleteideas.net/sutton/book/ebook/node16.html
#
# in the HTML version of the book

import numpy as np
import matplotlib.pyplot as plt

#
# Parameters
#

epsilon     = 0.1  # learning parameter
num_actions = 10   # number of arms on the bandit
iterations  = 100 # number of times we repeat the experiment
plays       = 1000 # number of plays within each experiment

#
# Class definition
#
class bandit_problem:

    # Initialization
    def __init__ (self):
        # Create the actual reward for each of the actions. This is a draw from
        # a normal distribution with mean 0 and variance (and standard deviation) 1.
        self.q_star = []
        for k in range(num_actions):
            self.q_star.append(np.random.randn())

    # When an action is selected, provide the payoff for the action.
    #
    # This is a draw from a normal distribution with mean q_star and
    # variance 1. That is the same as q_star plus a draw from a normal
    # distribution with mean 0 and variance 1
    def provide_payoff(self, action):
        return self.q_star[pick] + np.random.randn()

    def get_optimal_action(self):
        return greedy_pick(self.q_star)

#
# Functions
#

# From a list of utilities, pick the "greedy" option, just the max of
# the list.
def greedy_pick(utility_list):
    index = np.random.randint(0, len(utility_list))
    max = utility_list[index]
    for i in range(len(utility_list)):
        if utility_list[i] > max:
            index = i
            max =  utility_list[i]
    return index

# Setup lists to collect metric data
average_rewards = []
proportions_optimal_action = []

#
# Main loop --- repeated for each iteration
#
for i in range(iterations):

    # Create a new bandit with num_actions arms
    bp = bandit_problem()

    # Initialise the estimated reward for each action, Q_t(a), and 
    # the count of the times it has been chosen.
    q_t = []
    a_count = []
    for k in range(num_actions):
            q_t.append(0)
            a_count.append(0)

    #
    # Inner loop --- repeated for each play in each iteration
    #
    
    # Variable to collect data on each play. We reinitialise for each
    # iteration.
    rewards = []
    actions = []
    for j in range(plays):
        # Pick an action
        choice = np.random.random()
        if choice <= (1 - epsilon):
            pick = greedy_pick(q_t)
        else:
            # np.random.randint generates a number < upper bound
            pick = np.random.randint(0, num_actions)

        # Record which action we selected, both in a running list and
        # count of how many times each action was picked.
        actions.append(pick)
        a_count[pick] += 1
        # Get the reward for the action we picked, from the bandit problem
        rewards.append(bp.provide_payoff(pick))
                
        # Update utility estimate of the action we picked
        q_t[pick] = q_t[pick] + (1/float(a_count[pick]))*(rewards[j] - q_t[pick])

    #
    # End of inner loop
    #
    
    # Compute metrics over a run.
    counter = []
    average_reward = []
    proportion_optimal_action = []
    optimal_action = bp.get_optimal_action()
    for j in range(plays):
        counter.append(j)
        total_reward = 0
        optimal_action_count = 0
        for i in range(j):
            total_reward += rewards[i]
            if actions[i] == optimal_action:
                optimal_action_count += 1
                
        average_reward.append(total_reward/(j+1))
        proportion_optimal_action.append(float(optimal_action_count)/(j+1))

    # Stash metrics for later analysis
    average_rewards.append(average_reward)
    proportions_optimal_action.append(proportion_optimal_action)

#
# End of main loop
#

# Now average the results

averaged_reward = []
averaged_proportion = []
for i in range(plays):
    total_reward_per_step = 0
    total_proportion_per_step = 0
    for j in range(iterations):
        total_reward_per_step += average_rewards[j][i]
        total_proportion_per_step += proportions_optimal_action[j][i]
        
    averaged_reward.append(total_reward_per_step/iterations)
    averaged_proportion.append(float(total_proportion_per_step)/iterations)

# Make proportion a %age

for i in range(len(averaged_proportion)):
    averaged_proportion[i] = averaged_proportion[i]*100
    
# And plot
# Plot parameters are optimised for figures that are used in slides.
plt.subplot(1, 2, 1)
plt.title("Epsilon = " + str(epsilon), fontsize=20)
plt.plot(counter, averaged_reward, color = 'blue', linewidth = 4)
plt.xlabel("Plays", fontsize=20 )
plt.ylabel("Average reward", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0, 1.5)
plt.xlim(0, plays)

plt.subplot(1, 2, 2)
plt.title("Epsilon = " + str(epsilon), fontsize=20)
plt.plot(counter, averaged_proportion, color = 'green', linewidth=4)
plt.xlabel("Plays", fontsize=20 )
plt.ylabel("Percent optimal action", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0, 100)
plt.xlim(0, plays)

plt.tight_layout()
plt.show()

