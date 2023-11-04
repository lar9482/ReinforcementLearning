from WorldKnown.domains.navigationProblem import getNavigationProblem
from WorldKnown.domains.wumpusWorld import getWumpusWorld

from WorldKnown.valueIteration import valueIteration

def main():
    gridWorld = getNavigationProblem()
    wumpusWorld = getWumpusWorld()

    discount = 0.1
    error = 0.01
    GU = valueIteration(gridWorld, discount, error)
    WU = valueIteration(wumpusWorld, discount, error)
    print()

if __name__ == '__main__':
    main()
