from MultiArm.bandit_sim import Bandit_Sim

import math
import sys

class UCB:
    def __init__(self, numArms, payoutSTD):
        self.banditSim = Bandit_Sim(numArms, payoutSTD)
        self.numArms = numArms
        self.rewardMeansPerArm = {arm: 0 for arm in range(0, self.numArms)}
        self.numPullsPerArm = {arm: 0 for arm in range(0, self.numArms)}

    def runAlgorithm(self, numSamples):
        V = 0

        # Trying all arms before the repeat loop.
        for arm in range(0, self.numArms):
            reward = self.banditSim.pull_arm(arm)

            V += reward
            self.rewardMeansPerArm[arm] = (
                (self.numPullsPerArm[arm] * self.rewardMeansPerArm[arm] + reward) /
                (self.numPullsPerArm[arm] + 1)
            )
            self.numPullsPerArm[arm] += 1
        
        for n in range(1, numSamples+1):
            selectedArm = self.argMaxArm(n)
            reward = self.banditSim.pull_arm(selectedArm)

            V += reward
            self.rewardMeansPerArm[selectedArm] = (
                (self.numPullsPerArm[selectedArm] * self.rewardMeansPerArm[selectedArm] + reward) /
                (self.numPullsPerArm[selectedArm] + 1)
            )
            self.numPullsPerArm[selectedArm] += 1
            
        return V

    def argMaxArm(self, n):
        maxRewardMean = -sys.maxsize
        argMaxArm = 0

        for arm in range(0, self.numArms):
            currRewardMean = (
                self.rewardMeansPerArm[arm] + 
                math.sqrt(
                    (2 * math.log(n)) /
                    (self.numPullsPerArm[arm]) 
                )
            )
            if (currRewardMean > maxRewardMean):
                maxRewardMean = currRewardMean
                argMaxArm = arm
        
        return argMaxArm