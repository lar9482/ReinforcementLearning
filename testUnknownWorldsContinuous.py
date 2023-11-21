from WorldUnknown.domains.world1 import getWorld1Continuous, getWorld1Discrete
from WorldUnknown.domains.world2 import getWorld2Continuous, getWorld2Discrete
from WorldUnknown.domains.world3 import getWorld3Continuous, getWorld3Discrete

from WorldUnknown.QAgent import learnType
from WorldUnknown.functionQAgent import functionQAgent

from MDP.hashStates import hashState
from MDP.gridworld_c import Actions
# from MDP.gridworld import Actions, GridState
import random
import numpy as np

def testUnknownWorldsQLearnContinuous():
    world = getWorld3Continuous()
    
    maxExpectedReward = 1
    maxNumTries = 10
    discount = 0.1
    epsilon = 0.2
    radius = 1

    typeAgent = learnType.QLearn
    agent = functionQAgent(world, typeAgent, discount, maxExpectedReward, maxNumTries, epsilon, radius)

    convergeCount = 0
    for _ in range(0, 1000):

        state = world.initial_state
        action = random.choice(list(world.actions))
        totalR = 0
        totalRuns = 0

        while (not world.is_terminal(state) and totalRuns < 250):
            state, r = world.act(state, action)
            action = agent.learn(state, r)
            totalR += r
            totalRuns += 1

        if (convergeCount >= 5):
            break

        if ((totalR / totalRuns) == -0.009999999999999875):
            convergeCount += 1
        else:
            convergeCount = 0

        print(totalR / totalRuns)
    
    print('#####################')
    test1 = np.array([4, 4])
    test2 = np.array([6, 6])
    print(agent.calculateQValue(test1, Actions.UP))
    print(agent.calculateQValue(test1, Actions.DOWN))
    print(agent.calculateQValue(test1, Actions.LEFT))
    print(agent.calculateQValue(test1, Actions.RIGHT))
    print('#####################')
    print(agent.calculateQValue(test2, Actions.UP))
    print(agent.calculateQValue(test2, Actions.DOWN))
    print(agent.calculateQValue(test2, Actions.LEFT))
    print(agent.calculateQValue(test2, Actions.RIGHT))
    print()

    # test1 = GridState(4, 4, 10, 10)
    # test2 = GridState(6, 6, 10, 10)

    # print(agent.calculateQValue(test1, Actions.UP))
    # print(agent.calculateQValue(test1, Actions.DOWN))
    # print(agent.calculateQValue(test1, Actions.LEFT))
    # print(agent.calculateQValue(test1, Actions.RIGHT))
    # print('#####################')
    # print(agent.calculateQValue(test2, Actions.UP))
    # print(agent.calculateQValue(test2, Actions.DOWN))
    # print(agent.calculateQValue(test2, Actions.LEFT))
    # print(agent.calculateQValue(test2, Actions.RIGHT))
    # print()

def testUnknownWorldsSARSAContinuous():
    world = getWorld3Continuous()
    
    maxExpectedReward = 10
    maxNumTries = 50
    discount = 0.1
    epsilon = 0.3
    radius = 1

    typeAgent = learnType.SASRA
    agent = functionQAgent(world, typeAgent, discount, maxExpectedReward, maxNumTries, epsilon, radius)

    convergeCount = 0
    for _ in range(0, 50):

        state = world.initial_state
        action = random.choice(list(world.actions))
        totalR = 0
        totalRuns = 0

        while (not world.is_terminal(state) and totalRuns < 5000):
            state, r = world.act(state, action)
            action = agent.learn(state, r)
            totalR += r
            totalRuns += 1

        if (convergeCount >= 5):
            break

        if ((totalR / totalRuns) == -0.09):
            convergeCount += 1
        else:
            convergeCount = 0

        print(totalR / totalRuns)
    
    print('#####################')
    test1 = np.array([1, 1])
    print(agent.calculateQValue(test1, Actions.UP))
    print(agent.calculateQValue(test1, Actions.DOWN))
    print(agent.calculateQValue(test1, Actions.LEFT))
    print(agent.calculateQValue(test1, Actions.RIGHT))
    print('#####################')
    test2 = np.array([3, 3])
    print(agent.calculateQValue(test2, Actions.UP))
    print(agent.calculateQValue(test2, Actions.DOWN))
    print(agent.calculateQValue(test2, Actions.LEFT))
    print(agent.calculateQValue(test2, Actions.RIGHT))
    print('#####################')
    test3 = np.array([7, 7])
    print(agent.calculateQValue(test3, Actions.UP))
    print(agent.calculateQValue(test3, Actions.DOWN))
    print(agent.calculateQValue(test3, Actions.LEFT))
    print(agent.calculateQValue(test3, Actions.RIGHT))
    print('#####################')
    test4 = np.array([9, 9])
    print(agent.calculateQValue(test4, Actions.UP))
    print(agent.calculateQValue(test4, Actions.DOWN))
    print(agent.calculateQValue(test4, Actions.LEFT))
    print(agent.calculateQValue(test4, Actions.RIGHT))

    # test1 = GridState(3, 3, 10, 10)
    # test2 = GridState(7, 7, 10, 10)

    # print(agent.calculateQValue(test1, Actions.UP))
    # print(agent.calculateQValue(test1, Actions.DOWN))
    # print(agent.calculateQValue(test1, Actions.LEFT))
    # print(agent.calculateQValue(test1, Actions.RIGHT))
    # print('#####################')
    # print(agent.calculateQValue(test2, Actions.UP))
    # print(agent.calculateQValue(test2, Actions.DOWN))
    # print(agent.calculateQValue(test2, Actions.LEFT))
    # print(agent.calculateQValue(test2, Actions.RIGHT))
    # print()