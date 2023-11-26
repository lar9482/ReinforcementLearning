from MultiArm.BanditArmAlgo import BanditArmAlgo

import random
import sys

class EGreedy(BanditArmAlgo):
    def __init__(self, numArms, payoutSTD, epsilon = 0.25):
        super().__init__(numArms, payoutSTD)
        self.epsilon = epsilon
        
    def runAlgorithm(self, numSamples):
        cumulativeReward = self.pullAllArms()

        for n in range(1, numSamples):
            chance = random.uniform(0, 1)
            selectedArm = -1
            if (chance < self.epsilon):
                selectedArm = random.choice(list(range(self.numArms)))
            else:
                selectedArm = self.argMaxArm()
            
            reward = self.banditSim.pull_arm(selectedArm)

            cumulativeReward += reward
            self.actualRewardMeansPerArm[selectedArm] = (
                (self.numPullsPerArm[selectedArm] * self.actualRewardMeansPerArm[selectedArm] + reward) /
                (self.numPullsPerArm[selectedArm] + 1)
            )
            self.numPullsPerArm[selectedArm] += 1

        return cumulativeReward
    
    def argMaxArm(self):
        maxRewardMean = -sys.maxsize
        argMaxArm = -1

        for arm in list(range(self.numArms)):
            if self.actualRewardMeansPerArm[arm] > maxRewardMean:
                maxRewardMean = self.actualRewardMeansPerArm[arm]
                argMaxArm = arm
        
        return argMaxArm