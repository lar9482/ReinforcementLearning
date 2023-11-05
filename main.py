from WorldKnown.domains.navigationProblem import getNavigationProblem
from WorldKnown.domains.wumpusWorld import getWumpusWorld

from WorldKnown.valueIteration import valueIteration
from WorldKnown.policyIteration import policyIteration

def main():
    gridWorld = getNavigationProblem()
    wumpusWorld = getWumpusWorld()

    discount = 0.25
    error = 0.1
    k = 10
    gridWorld.display()
    Gpolicy = policyIteration(gridWorld, discount, k)
    Wpolicy = policyIteration(wumpusWorld, discount, k)

    # GU = valueIteration(gridWorld, discount, error)
    # WU = valueIteration(wumpusWorld, discount, error)
    print()

if __name__ == '__main__':
    main()
