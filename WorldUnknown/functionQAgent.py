from MDP.hashStates import hashState
from WorldUnknown.QAgent import QAgent, learnType

import sys
import copy
import random
import math

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
        self.prevState = None
        self.prevAction = None

        self.discount = discount
        self.maxExpectedReward = maxExpectedReward
        self.maxNumTries = maxNumTries
        self.epsilon = epsilon
        self.radius = radius

        self.theta1 = random.uniform(0, 1)
        self.theta2 = random.uniform(0, 1)
        self.theta3 = random.uniform(0, 1)
    
    def learn(self, currState, currReward):
        if (self.prevState is not None):
            pass

        self.prevState = copy.deepcopy(currState)
        self.prevAction = self.__argMaxExplore(currState)

        return copy.deepcopy(self.prevAction)

    def QLearn(self, state, reward):
        pass

    def SARSA(self, state, reward):
        pass

    def __argMaxExplore(self, currState):
        possibleActions = self.mdpSimulator.actions_at(currState)
        maxExploreValue = -sys.maxsize

        selectedActions = []
        for actionPrime in possibleActions:
            exploreValuePrime = self.explore(
                self.calculateQValue(currState, actionPrime),
                self.__lookUpNTable(currState, actionPrime),
                self.maxNumTries, 
                self.maxExpectedReward
            )
            if (exploreValuePrime > maxExploreValue):
                maxExploreValue = exploreValuePrime
                selectedActions = [actionPrime]

            elif (exploreValuePrime == maxExploreValue):
                selectedActions.append(actionPrime)
        
        return random.choice(selectedActions)

    def calculateQValue(self, state, action):
        nextState, _ = self.mdpSimulator.act(state, action)
        hashedNextState = hashState(nextState)

        X = hashedNextState[0]
        Y = hashedNextState[1]

        return (
            self.theta1 +
            (self.theta2 * X) +
            (self.theta3 * Y)
        )
    
    def __lookUpNTable(self, state, action):
        nextState, _ = self.mdpSimulator.act(state, action)
        hashedNextState = hashState(nextState)
        discreteNextState = self.__getDiscreteState(hashedNextState[0], hashedNextState[1])

        if (self.N.get((discreteNextState, action)) == None):
            self.N[(discreteNextState, action)] = 0

        return self.N[(discreteNextState, action)]

    def __getDiscreteState(self, X, Y):
        integerBaseX = int(X)
        integerBaseY = int(Y)
        roundX = math.ceil(X) if ((X - integerBaseX) >= 0.5) else math.floor(X)
        roundY = math.ceil(Y) if ((Y - integerBaseY) >= 0.5) else math.floor(Y)

        return (roundX, roundY)
        
        