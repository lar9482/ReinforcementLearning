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

def testUnknownWorlds():
    world1 = getWorld1Discrete()
    world2 = getWorld2Continuous()
    world3 = getWorld3Continuous()
    
    maxExpectedReward = 1
    maxNumTries = 10
    discount = 0.1

    typeAgent = agentType.QLearn
    agent = tableQAgent(world1, typeAgent, discount, maxExpectedReward, maxNumTries)

    state = world1.initial_state
    action = random.choice(list(world1.actions))
    for i in range(0, 1000):
        state, r = world1.act(state, action)
        action = agent.learn(state, r)

        value = sum(agent.Q.values()) / len(agent.Q)
        
        print(value)
        # print(action)
    
    bestState = max(agent.Q, key=agent.Q.get)
    bestValue = agent.Q[bestState]
    print()

def main():
    # testKnownWorlds()
    testUnknownWorlds()

if __name__ == '__main__':
    main()