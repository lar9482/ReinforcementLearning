from enum import Enum
from MDP.hashStates import hashState

import sys

class agentType(Enum):
    QLearn = 0
    SASRA = 1

class tableQAgent:
    def __init__(self, mdpSimulator, agentType, maxExpectedReward, maxNumTries):
        self.mdpSimulator = mdpSimulator
        self.agentType = agentType
        self.N = {}
        self.Q = {}

        self.prevState = None
        self.prevAction = None

        self.maxExpectedReward = maxExpectedReward
        self.maxNumTries = maxNumTries

    def learn(self, currState, currReward):
        if (self.prevState != None):
            pass

        self.prevState = currState
        self.prevAction = self.argMaxExplore(currState)

        return self.prevAction

    def argMaxExplore(self, currState):
        possibleActions = self.mdpSimulator.actions_at(currState)
        argmaxAction = None
        maxExploreValue = -sys.maxsize

        for actionPrime in possibleActions:
            exploreValuePrime = self.explore(
                self.lookUpQTable(currState, actionPrime),
                self.lookUpNTable(currState, actionPrime)
            )
            if (exploreValuePrime > maxExploreValue):
                maxExploreValue = exploreValuePrime
                argmaxAction = actionPrime

        return argmaxAction
    
    def explore(self, U, N):
        if (N < self.maxNumTries):
            return self.maxExpectedReward
        else:
            return U
    
    def incrementNTable(self):
        hashedPrevState = hashState(self.prevState)
        self.N[(hashedPrevState, hashedPrevState)] += 1

    def lookUpNTable(self, state, action):
        hashedState = hashState(state)
        if (self.N.get((hashedState, action)) == None):
            self.N[(hashedState, action)] = 0

        return self.N[(hashedState, action)]
    
    def lookUpQTable(self, state, action):
        hashedState = hashState(state)
        if (self.Q.get((hashedState, action)) == None):
            self.Q[(hashedState, action)] = 0
        
        return self.Q[(hashedState, action)]
    
    def learningRate(self, N):
        return 1 / N