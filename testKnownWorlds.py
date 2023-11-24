from WorldKnown.domains.navigationProblem import getNavigationProblem
from WorldKnown.domains.wumpusWorld import getWumpusWorld
from WorldKnown.valueIteration import valueIteration
from WorldKnown.policyIteration import policyIteration

class testParameter_ValueIteration:
    def __init__(self, world, discount, error):
        self.world = world
        self.discount = discount
        self.error = error

class testParameter_PolicyIteration:
    def __init__(self, world, discount, error, k):
        self.world = world
        self.discount = discount
        self.error = error
        self.k = k
        
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