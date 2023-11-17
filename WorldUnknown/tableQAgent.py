from MDP.hashStates import hashState
from WorldUnknown.QAgent import QAgent, learnType

import sys
import copy
import random

class tableQAgent(QAgent):
    def __init__(
        self, 
        mdpSimulator, 
        learnType, 
        discount = 0.1, 
        maxExpectedReward = 1, 
        maxNumTries = 10, 
        epsilon = 0.5
    ):
        self.mdpSimulator = mdpSimulator
        self.learnType = learnType
        self.N = {}
        self.Q = {}

        self.prevState = None
        self.prevAction = None

        self.discount = discount
        self.maxExpectedReward = maxExpectedReward
        self.maxNumTries = maxNumTries
        self.epsilon = epsilon

    def learn(self, currState, currReward):
        if (self.prevState is not None):
            self.__incrementNTable()

            self.Q[(
                hashState(self.prevState),
                self.prevAction
            )] += (
                self.QLearn(currState, currReward) if self.learnType == learnType.QLearn else
                self.SARSA(currState, currReward)
            )

        self.prevState = copy.deepcopy(currState)
        self.prevAction = self.__argMaxExplore(currState)

        return copy.deepcopy(self.prevAction)

    def QLearn(self, state, reward):
        alpha = self.learningRate(
            self.__lookUpNTable(self.prevState, self.prevAction)
        )
        
        maxDiscountedQValue = -sys.maxsize
        for actionPrime in self.mdpSimulator.actions_at(state):
            currDiscountedQValue = self.__lookUpQTable(state, actionPrime)
            maxDiscountedQValue = max(maxDiscountedQValue, currDiscountedQValue)
        
        return alpha * (
            reward + 
            (self.discount * maxDiscountedQValue) -
            self.__lookUpQTable(self.prevState, self.prevAction)
        )

    def SARSA(self, state, reward):
        alpha = self.learningRate(
            self.__lookUpNTable(self.prevState, self.prevAction)
        )
        
        actionPrime = (
            self.__argMaxExploit(state) if random.uniform(0, 1) < self.epsilon else
            self.__argMaxExplore(state)
        )
        
        return alpha * (
            reward + 
            (self.discount * self.__lookUpQTable(state, actionPrime)) -
            self.__lookUpQTable(self.prevState, self.prevAction)
        )

    def __argMaxExploit(self, currState):
        possibleActions = self.mdpSimulator.actions_at(currState)
        argmaxActions = []
        maxQValue = -sys.maxsize
        for actionPrime in possibleActions:

            QValuePrime = self.__lookUpQTable(currState, actionPrime)
            if (QValuePrime > maxQValue):
                maxQValue = QValuePrime
                argmaxActions = [actionPrime]
            elif (QValuePrime == maxQValue):
                argmaxActions.append(actionPrime)
        
        return random.choice(argmaxActions)
    
    def __argMaxExplore(self, currState):
        possibleActions = self.mdpSimulator.actions_at(currState)
        argmaxActions = []
        maxExploreValue = -sys.maxsize

        for actionPrime in possibleActions:
            exploreValuePrime = self.__explore(
                self.__lookUpQTable(currState, actionPrime),
                self.__lookUpNTable(currState, actionPrime)
            )
            if (exploreValuePrime > maxExploreValue):
                maxExploreValue = exploreValuePrime
                argmaxActions = [actionPrime]
            elif (exploreValuePrime == maxExploreValue):
                argmaxActions.append(actionPrime)

        return random.choice(argmaxActions)
    
    def __explore(self, U, N):
        if (N < self.maxNumTries):
            return self.maxExpectedReward
        else:
            return U
    
    def __incrementNTable(self):
        hashedPrevState = hashState(self.prevState)
        self.N[(hashedPrevState, self.prevAction)] += 1

    def __lookUpNTable(self, state, action):
        hashedState = hashState(state)
        if (self.N.get((hashedState, action)) == None):
            self.N[(hashedState, action)] = 0

        return self.N[(hashedState, action)]
    
    def __lookUpQTable(self, state, action):
        hashedState = hashState(state)
        if (self.Q.get((hashedState, action)) == None):
            self.Q[(hashedState, action)] = 0
        
        return self.Q[(hashedState, action)]