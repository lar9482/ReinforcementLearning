import copy
import sys

from WorldKnown.QValue import QValue
from MDP.hashStates import hashState

def valueIteration(mdp, discount, error):
    U = {hashState(state): 0 for state in list(mdp.states)}
    UPrime = {hashState(state): 0 for state in list(mdp.states)}

    while True:
        deltaU = 0
        U = copy.deepcopy(UPrime)

        for state in list(mdp.states):
            hashedState = hashState(state)
            
            UPrime[hashedState] = maxQValue(mdp, discount, state, U)
            if (abs(UPrime[hashedState] - U[hashedState]) > deltaU):
                deltaU = abs(UPrime[hashedState] - U[hashedState])

        if (deltaU <= (error*(1-discount)) / (discount)):
            break

    return U

def maxQValue(mdp, discount, currState, currU):
    actionsAtCurrState = mdp.actions_at(currState)
    maxU = -sys.maxsize

    for currAction in actionsAtCurrState:
        maxU = max(maxU, QValue(mdp, discount, currState, currAction, currU))
    
    return maxU