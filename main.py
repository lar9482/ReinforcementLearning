from WorldKnown.domains.navigationProblem import getNavigationProblem
from WorldKnown.domains.wumpusWorld import getWumpusWorld
from WorldKnown.valueIteration import valueIteration
from WorldKnown.policyIteration import policyIteration

from WorldUnknown.domains.world1 import getWorld1Continuous, getWorld1Discrete
from WorldUnknown.domains.world2 import getWorld2Continuous, getWorld2Discrete
from WorldUnknown.domains.world3 import getWorld3Continuous, getWorld3Discrete
from WorldUnknown.tableQAgent import tableQAgent, agentType

from MDP.hashStates import hashState

import random
def testKnownWorlds():

    gridWorld = getNavigationProblem()
    wumpusWorld = getWumpusWorld()

    discount = 0.1
    error = 0.1
    k = 50

    GU = valueIteration(gridWorld, discount, error)
    WU = valueIteration(wumpusWorld, discount, error)

    Gpolicy = policyIteration(gridWorld, discount, k)
    Wpolicy = policyIteration(wumpusWorld, discount, k)
    print()

def testUnknownWorldsQLearn():
    world = getWorld1Discrete()
    
    maxExpectedReward = 1
    maxNumTries = 10
    discount = 0.1

    typeAgent = agentType.QLearn
    agent = tableQAgent(world, typeAgent, discount, maxExpectedReward, maxNumTries)

    state = world.initial_state
    action = random.choice(list(world.actions))
    for i in range(0, 1000):
        state, r = world.act(state, action)
        action = agent.learn(state, r)

        value = sum(agent.Q.values()) / len(agent.Q)
        print(value)
    
    bestState = max(agent.Q, key=agent.Q.get)
    bestValue = agent.Q[bestState]
    print()

def testUnknownWorldsSARSA():
    world = getWorld2Discrete()
    
    maxExpectedReward = 1
    maxNumTries = 10
    discount = 0.1
    epsilon = 0.5

    typeAgent = agentType.SASRA
    agent = tableQAgent(world, typeAgent, discount, maxExpectedReward, maxNumTries, epsilon)
    state = world.initial_state
    action = random.choice(list(world.actions))

    for i in range(0, 5000):
        state, r = world.act(state, action)
        action = agent.learn(state, r)

        value = sum(agent.Q.values()) / len(agent.Q)
        print(value)
    
    bestState = max(agent.Q, key=agent.Q.get)
    bestValue = agent.Q[bestState]
    print()

def main():
    # testKnownWorlds()
    # testUnknownWorldsQLearn()
    testUnknownWorldsSARSA()
    
if __name__ == '__main__':
    main()