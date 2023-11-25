from MDP.hashStates import hashState
from WorldUnknown.QAgent import QAgent, learnType

import sys
import copy
import random
import math
import numpy as np

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

        self.theta1 = random.uniform(-1, 1)
        self.theta2 = random.uniform(-1, 1)
        self.theta3 = random.uniform(-1, 1)
    
    def learn(self, currState, currReward):
        if (self.prevState is not None):
            self.__incrementNTable()
            
            if (self.learnType == learnType.QLearn):
                self.QLearn(currState, currReward) 
            else:
                self.SARSA(currState, currReward)

        self.prevState = copy.deepcopy(currState)
        self.prevAction = self.__argMaxExplore(currState)

        return copy.deepcopy(self.prevAction)

    def QLearn(self, state, reward):
        alpha = self.learningRate(
            self.__lookUpNTable(self.prevState, self.prevAction)
        )

        maxDiscountedQValue = -sys.maxsize
        for actionPrime in self.mdpSimulator.actions_at(state):
            currDiscountedQValue = self.calculateQValue(state, actionPrime)
            maxDiscountedQValue = max(maxDiscountedQValue, currDiscountedQValue)
        
        sample = alpha * np.clip((
            reward + 
            (self.discount * maxDiscountedQValue) -
            (self.calculateQValue(self.prevState, self.prevAction))
        ), -1, 1)
        
        hashedPrevState = hashState(self.prevState)
        prevStateX = hashedPrevState[0]
        prevStateY = hashedPrevState[1]

        self.theta1 = np.clip(self.theta1 + sample, -1, 1)
        self.theta2 = np.clip(self.theta2 + (sample * prevStateX), -1, 1)
        self.theta3 = np.clip(self.theta3 + (sample * prevStateY), -1, 1)

    def SARSA(self, state, reward):
        alpha = self.learningRate(
            self.__lookUpNTable(self.prevState, self.prevAction)
        )

        actionPrime = (
            self.__argMaxExploit(state) if random.uniform(0, 1) < self.epsilon else
            self.__argMaxExplore(state)
        )

        sample = alpha * np.clip((
            reward + 
            (self.discount * self.calculateQValue(state, actionPrime)) -
            (self.calculateQValue(self.prevState, self.prevAction))
        ), -5, 5)

        hashedPrevState = hashState(self.prevState)
        prevStateX = hashedPrevState[0]
        prevStateY = hashedPrevState[1]

        self.theta1 = np.clip(self.theta1 + sample, -5, 5)
        self.theta2 = np.clip(self.theta2 + (sample * prevStateX), -5, 5)
        self.theta3 = np.clip(self.theta3 + (sample * prevStateY), -5, 5)
    
    def __argMaxExploit(self, currState):
        possibleActions = self.mdpSimulator.actions_at(currState)
        argmaxActions = []
        maxQValue = -sys.maxsize
        for actionPrime in possibleActions:

            QValuePrime = self.calculateQValue(currState, actionPrime)
            if (QValuePrime > maxQValue):
                maxQValue = QValuePrime
                argmaxActions = [actionPrime]
            elif (QValuePrime == maxQValue):
                argmaxActions.append(actionPrime)
        
        return random.choice(argmaxActions)

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
            (self.theta2 * (X)) +
            (self.theta3 * (Y))
        )
    
    def __incrementNTable(self):
        (closestXY, seenActions) = self.__getClosestLocationWithActions(self.prevState)

        # Case where no close location has been seen yet.
        if (closestXY == None):
            hashedPrevState = hashState(self.prevState)
            NTableEntry_PrevState = (
                (hashedPrevState[0], hashedPrevState[1]), 
                self.prevAction
            )

            self.N[NTableEntry_PrevState] = 1
            return self.N[NTableEntry_PrevState]
        
        # Case where no association between the previous action and the closest location 
        # has been associated yet.
        if (not self.prevAction in seenActions):
            self.N[(
                closestXY,
                self.prevAction
            )] = 1
            
        else:
            self.N[(
                closestXY,
                self.prevAction
            )] += 1

        return self.N[(
            closestXY,
            self.prevAction
        )]

    def __lookUpNTable(self, state, action):
        (closestXY, seenActions) = self.__getClosestLocationWithActions(state)

        # Case where no close location has been seen yet.
        if (closestXY == None):
            hashedCurrState = hashState(state)
            NTableEntry_CurrState = (
                (hashedCurrState[0], hashedCurrState[1]), 
                action
            )

            self.N[NTableEntry_CurrState] = 0
            return self.N[NTableEntry_CurrState]
        
        # Case where no association between the current action and the closest location 
        # has been associated yet.
        if (not action in seenActions):
            self.N[(
                closestXY,
                action
            )] = 0

        return self.N[(
            closestXY,
            action
        )]

    def __getClosestLocationWithActions(self, state):
        hashedState = hashState(state)

        X = hashedState[0]
        Y = hashedState[1]

        shortestDist = sys.maxsize
        closestXY = None
        seenActions = []
        
        for seenState in self.N:
            seenStateX = seenState[0][0]
            seenStateY = seenState[0][1]

            dist = ((seenStateX - X) ** 2) + ((seenStateY - Y) ** 2)
            if (dist < self.radius):
                if (dist < shortestDist):
                    closestXY = (seenStateX, seenStateY)
                    shortestDist = dist
                    seenActions = [seenState[1]]

                elif (dist == shortestDist):
                    seenActions.append(seenState[1])
        
        return (closestXY, seenActions)