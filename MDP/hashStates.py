from MDP.gridworld import GridState
from MDP.wumpus import WumpusState
import numpy as np
"""
    This method will hash 'state' by hand, in order to prevent any
    unhashable situations with the state of the MDP.

    WARNING: This function should only be used when referencing a dictionary.
    I.E
        U[hashState(state)] = 0
        is the correct way of using this function

        It should NOT BE USED for you're calling methods of the MDP itself.

        A call such as
        mdp.r(hashState(s1), hashState(s2))
        will throw an exception.
"""

def hashState(state):
    if isinstance(state, GridState):
        return (state.x, state.y)
    if isinstance(state, WumpusState):
        return (state.x, state.y, state.has_gold, state.has_immunity)
    if isinstance(state, np.ndarray) and state.shape == (2,):
        return (state[0], state[1])
    raise ValueError("Unable to hash the state")