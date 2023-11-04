from WorldKnown.domains.navigationProblem import getNavigationProblem
from WorldKnown.domains.wumpusWorld import getWumpusWorld
def main():
    gridWorld = getNavigationProblem()
    gridWorld.display()

    wumpusWorld = getWumpusWorld()
    wumpusWorld.display()

if __name__ == '__main__':
    main()
