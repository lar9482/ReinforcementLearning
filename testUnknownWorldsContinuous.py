from WorldUnknown.domains.world1 import getWorld1Continuous
from WorldUnknown.domains.world2 import getWorld2Continuous
from WorldUnknown.domains.world3 import getWorld3Continuous

from WorldUnknown.QAgent import learnType
from WorldUnknown.functionQAgent import functionQAgent

from MDP.hashStates import hashState

import random

def testUnknownWorldsQLearnContinuous():
    world = getWorld2Continuous()
    
    maxExpectedReward = 1
    maxNumTries = 20
    discount = 0.001
    epsilon = 0.5
    radius = 1

    typeAgent = learnType.QLearn
    agent = functionQAgent(world, typeAgent, discount, maxExpectedReward, maxNumTries, epsilon, radius)

    state = world.initial_state
    action = random.choice(list(world.actions))
    for i in range(0, 10000):
        state, r = world.act(state, action)
        action = agent.learn(state, r)
        print()
    
    print()