from WorldUnknown.domains.world1 import getWorld1Continuous, getWorld1Discrete
from WorldUnknown.domains.world2 import getWorld2Continuous, getWorld2Discrete
from WorldUnknown.domains.world3 import getWorld3Continuous, getWorld3Discrete

from WorldUnknown.QAgent import learnType
from WorldUnknown.functionQAgent import functionQAgent

from MDP.hashStates import hashState
# from MDP.gridworld_c import Actions
from MDP.gridworld import Actions, GridState
import random
import numpy as np

def testUnknownWorldsQLearnContinuous():
    world = getWorld2Discrete()
    
    maxExpectedReward = 1
    maxNumTries = 5
    discount = 0.1
    epsilon = 0.2
    radius = 1

    typeAgent = learnType.QLearn
    agent = functionQAgent(world, typeAgent, discount, maxExpectedReward, maxNumTries, epsilon, radius)

    convergeCount = 0
    for i in range(0, 50):
        state = world.initial_state
        action = random.choice(list(world.actions))
        totalR = 0
        totalRuns = 0

        while (not world.is_terminal(state) and totalRuns < 100):
            state, r = world.act(state, action)
            action = agent.learn(state, r)
            totalR += r
            totalRuns += 1

        if (convergeCount >= 10):
            break

        if (totalR == -1.0000000000000007):
            convergeCount += 1
        else:
            convergeCount = 0
        
        print(totalR)
    
    print('#####################')
    # print(agent.calculateQValue(np.array([1, 0]), Actions.UP))
    # print(agent.calculateQValue(np.array([2, 1]), Actions.UP))
    # print(agent.calculateQValue(np.array([3, 2]), Actions.UP))
    # print(agent.calculateQValue(np.array([4, 3]), Actions.UP))
    # print(agent.calculateQValue(np.array([5, 4]), Actions.UP))
    # print(agent.calculateQValue(np.array([6, 5]), Actions.UP))
    # print(agent.calculateQValue(np.array([7, 6]), Actions.UP))
    # print(agent.calculateQValue(np.array([8, 7]), Actions.UP))
    # print(agent.calculateQValue(np.array([9, 8]), Actions.UP))
    test1 = GridState(1, 0, 10, 10)
    test2 = GridState(5, 4, 10, 10)
    test3 = GridState(10, 9, 10, 10)
    print(agent.calculateQValue(test1, Actions.UP))
    print(agent.calculateQValue(test2, Actions.UP))
    print(agent.calculateQValue(test3, Actions.UP))


    print()