# banditAgents.py
# parsons/25-mar-2017
#
# A bandit agent to work with the Pacman AI projects, reinforcement
# learning edition:
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

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util

# BanditAgent
#
# An agent that uses an n-armed bandit to choose which action to
# take. Not a great way to do things, but it shows some learning.
class BanditAgent(Agent):

    # Build initial internal state.
    def __init__ (self):
        # Values for each possible action
        self.qValues = [['North', 0], ['East', 0], ['South', 0], ['West', 0]]
        # Local copy of the game score
        self.scoreTracker = 0
        # Epsilon to control explore/exploit
        self.epsilon = 0.5
        # How long have we been running?
        self.steps = 0

    # Update values. Given value and action, update the average value
    # of the action.
    def updateQValue(self, direction, value):
        for i in range(len(self.qValues)):
            if direction == self.qValues[i][0]:
                self.qValues[i][1] += 1.0/(self.getSteps()) * (value - self.qValues[i][1])

    # Access the value of actions.
    def getQValue(self, direction):
        for i in range(len(self.qValues)):
            if direction == self.qValues[i][0]:
                return self.qValues[i][1]

    def setScoreTracker(self, score):
        self.scoreTracker = score

    def getScoreTracker(self):
        return self.scoreTracker

    def getEpsilon(self):
        return self.epsilon

    def incrementSteps(self):
        self.steps += 1
        
    def getSteps(self):
        return self.steps

    # Given a list of values, return the index of the biggest value
    def greedyPick(self, list):
        # Note random.randint generates a number <= upper bound
        index = random.randint(0, (len(list) - 1))
        max = list[index]
        for i in range(len(list)):
            if list[i] > max:
                index = i
                max =  list[i]
        return index

    def getAction(self, state):
        # Increase the counter
        self.incrementSteps()
        # Get the last action
        last = state.getPacmanState().configuration.direction
        # Calculate the change in score due to the last action, and
        # update our local copy of the score
        current_score = state.getScore()
        change_in_score = current_score - self.getScoreTracker()
        self.setScoreTracker(current_score)
        
        # Since we are still running, attribute change in score to the last action
        self.updateQValue(last, change_in_score)
        
        # Get the actions we can try, and remove "STOP" if that is one of them.
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # For each action in legal, build a list with the current q-values:
        q_values = []
        for i in range(len(legal)):
           q_values.append(self.getQValue(legal[i]))

        # Make the epsilon-greedy choice
        choice = random.random()
        if choice <= (1 - self.getEpsilon()):
            pickIndex = self.greedyPick(q_values)
            pick = legal[pickIndex]
        else:
            pick = random.choice(legal)
        return pick
