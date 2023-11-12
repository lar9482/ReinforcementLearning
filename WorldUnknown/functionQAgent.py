from MDP.hashStates import hashState
from WorldUnknown.QAgent import QAgent, learnType

import sys
import copy
import random

class functionQAgent(QAgent):
    def __init__(
        self, 
        mdpSimulator, 
        learnType, 
        discount = 0.1, 
        maxExpectedReward = 1, 
        maxNumTries = 10, 
        epsilon = 0.5
    ):
        pass