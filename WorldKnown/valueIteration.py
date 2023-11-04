import copy
import sys

from WorldKnown.QValue import QValue
from MDP.hashStates import hashState

"""
    Calculates the utility function for the MDP
    @param mdp: MDP
        The MDP simulator
    
    @param discount: Float
        The discount factor
    
    @param error: Float
        The maximum error allowed
    
    @returns {hashState(state): float}
        The actual utility function itself
        NOTE: 
        The key is a hashed version of 'state' using 'hashState'
"""
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


"""
    Returns the maximum QValue from the current state given all possible actions
    one can take from the state.

    @param mdp: MDP
        The MDP simulator
    
    @param discount: Float
        The discount factor
    
    @param currState: MDPState
        The current state
    
    @param currU: {hashState(state): float}
        A dictionary representing the utility function
        NOTE: 
        The key is a hashed version of 'state' using 'hashState'
    
    @returns float:
        The maximum Q value itself
"""
def maxQValue(mdp, discount, currState, currU):
    actionsAtCurrState = mdp.actions_at(currState)
    maxU = -sys.maxsize

    for currAction in actionsAtCurrState:
        maxU = max(maxU, QValue(mdp, discount, currState, currAction, currU))
    
    return maxU