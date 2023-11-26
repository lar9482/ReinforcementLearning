from MultiArm.BanditArmAlgo import BanditArmAlgo

import math
import sys

class UCB(BanditArmAlgo):
    def __init__(self, numArms, payoutSTD):
        super().__init__(numArms, payoutSTD) 

    def runAlgorithm(self, numSamples):
        cumulativeReward = self.pullAllArms()
        
        for n in range(1, numSamples):
            selectedArm = self.argMaxArm(n)
            reward = self.banditSim.pull_arm(selectedArm)

            cumulativeReward += reward
            self.rewardMeansPerArm[selectedArm] = (
                (self.numPullsPerArm[selectedArm] * self.rewardMeansPerArm[selectedArm] + reward) /
                (self.numPullsPerArm[selectedArm] + 1)
            )
            self.numPullsPerArm[selectedArm] += 1
            
        return cumulativeReward

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