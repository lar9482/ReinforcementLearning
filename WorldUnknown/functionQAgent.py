from MDP.hashStates import hashState
from WorldUnknown.QAgent import QAgent, learnType

import sys
import copy
import random

class functionQAgent(QAgent):
    def __init__(
        self, 
        mdpSimulator, 
        learnType,
        discount = 0.1, 
        maxExpectedReward = 1, 
        maxNumTries = 10, 
        epsilon = 0.5,
        radius = 1
    ):
        self.mdpSimulator = mdpSimulator
        self.learnType = learnType

        self.N = {}
        self.radius = radius
        self.prevState = None
        self.prevAction = None

        self.discount = discount
        self.maxExpectedReward = maxExpectedReward
        self.maxNumTries = maxNumTries
        self.epsilon = epsilon
    
    def learn(self, currState, currReward):
        pass

    def QLearn(self, state, reward):
        pass

    def SARSA(self, state, reward):
        pass