from WorldUnknown.domains.world1 import getWorld1Discrete
from WorldUnknown.domains.world2 import getWorld2Discrete
from WorldUnknown.domains.world3 import getWorld3Discrete
from WorldUnknown.QAgent import learnType
from WorldUnknown.tableQAgent import tableQAgent

from MDP.hashStates import hashState

import random

def testUnknownWorldsQLearnDiscrete():
    world = getWorld2Discrete()
    
    maxExpectedReward = 1
    maxNumTries = 20
    discount = 0.001

    typeAgent = learnType.QLearn
    agent = tableQAgent(world, typeAgent, discount, maxExpectedReward, maxNumTries)

    state = world.initial_state
    action = random.choice(list(world.actions))
    for i in range(0, 10000):
        state, r = world.act(state, action)
        action = agent.learn(state, r)

        value = agent.Q[(hashState(state), action)]
        # value = agent.Q[max(agent.Q, key=agent.Q.get)]
        print(value)
    
    bestState = max(agent.Q, key=agent.Q.get)
    bestValue = agent.Q[bestState]
    print()

def testUnknownWorldsSARSADiscrete():
    world = getWorld2Discrete()
    
    maxExpectedReward = 1
    maxNumTries = 10
    discount = 0.1
    epsilon = 0.5

    typeAgent = learnType.SASRA
    agent = tableQAgent(world, typeAgent, discount, maxExpectedReward, maxNumTries, epsilon)
    state = world.initial_state
    action = random.choice(list(world.actions))

    for i in range(0, 5000):
        state, r = world.act(state, action)
        action = agent.learn(state, r)

        value = agent.Q[(hashState(state), action)]
        # value = agent.Q[max(agent.Q, key=agent.Q.get)]
        print(value)
    
    bestState = max(agent.Q, key=agent.Q.get)
    bestValue = agent.Q[bestState]
    print()