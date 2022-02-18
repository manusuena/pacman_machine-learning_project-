# python 2.7 script
# Chen/13-apr-2017
# based on the script written by Parsons
#
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
import random
import math
import game
import util

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # Q-values
        self.q_value = util.Counter()
        # current score
        self.score = 0
        # last state
        self.lastState = []
        # last action
        self.lastAction = []



    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts


    def get_max_q(self, state):  # return the maximum Q of state
        q_list = []  # makes an empty list
        for actions in state.getLegalPacmanActions():   # fills the list with all possible legal action in this state
            q = self.q_value[(state, actions)]
            q_list.append(q)
        if len(q_list) == 0:  # check if the list is empty
            return 0
        else:
            return max(q_list)  # output the best q value

    def update_q(self, state, action, reward, q_max):  # updates Q values
        q = self.q_value[(state, action)]  # gets the q values
        self.q_value[(state, action)] = q + self.alpha*(reward + self.gamma*q_max - q)  # the new q values are calculated

    def q_learning(self, state):  # return the action that maximises Q of state
        legal = state.getLegalPacmanActions()
        if self.getEpisodesSoFar()*1.0/self.getNumTraining() < 0.5:
            if Directions.STOP in legal:
                legal.remove(Directions.STOP)
            if len(self.lastAction) > 0:
                last_action = self.lastAction[-1]  # get the last action
                distance0 = state.getPacmanPosition()[0] - state.getGhostPosition(1)[0]  # get the distance between pacmman and ghost
                distance1 = state.getPacmanPosition()[1] - state.getGhostPosition(1)[1]
                if math.sqrt(distance0**2 + distance1**2) > 2:  # calculated the distance and check if its grater then 2
                    if (Directions.REVERSE[last_action] in legal) and len(legal) > 1:
                        legal.remove(Directions.REVERSE[last_action])
        tmp = util.Counter()
        for action in legal:
          tmp[action] = self.q_value[(state, action)]
        return tmp.argMax()

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # The data we have about the state of the game
        # the legal action of this state
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        reward = state.getScore()-self.score  # the reward is calculated
        if len(self.lastState) > 0:
            last_state = self.lastState[-1]   # last state is saved
            last_action = self.lastAction[-1]  # last action is saved
            max_q = self.get_max_q(state)  # get the highest q value
            self.update_q(last_state, last_action, reward, max_q)  # update Q-value

        probability = random.random()
        if probability < self.epsilon:  # e-greedy is used to decide which action to make
            action = random.choice(legal)  # pick random legal action to explore
        else:
            action = self.q_learning(state)  # pick an action from the q-list to exploit

        self.score = state.getScore()  # update score
        self.lastState.append(state)   # update state
        self.lastAction.append(action)  # update action

        pick = action
        return pick

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        reward = state.getScore()-self.score  # calculated reward
        last_state = self.lastState[-1]  # get the last state
        last_action = self.lastAction[-1]  # get the last action
        self.update_q(last_state, last_action, reward, 0)  # update Q-values

        self.score = 0  # reset score
        self.lastState = []  # reset state
        self.lastAction = []  # reset action

        ep = 1 - self.getEpisodesSoFar()*1.0/self.getNumTraining()  # decrease epsilon during the training
        self.setEpsilon(ep*0.1)


        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() % 100 == 0:
            print "Completed %s runs of training" % self.getEpisodesSoFar()

        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)