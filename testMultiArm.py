from MultiArm.UCB import UCB
from MultiArm.EGreedy import EGreedy

from MultiArm.testUCBAccuracy import getDecisionAccuracyDataset
from MultiArm.testUCBAccuracy import testUCBAccuracy

from MultiArm.testRegret import getRegretDataset
from MultiArm.testRegret import testRegret

def testAllUCBAccuracy():
    dataset = getDecisionAccuracyDataset()
    for parameter in dataset:
        testUCBAccuracy(parameter)

def testAllRegret():
    dataset = getRegretDataset()
    for parameter in dataset:
        testRegret(parameter)

def testMultiArm():
    # numArms = 10
    # numSamples = 10
    # payoutSTD = 0.25

    # UCBInstance = UCB(numArms, payoutSTD)
    # cumRewardUCB = UCBInstance.runAlgorithm(numSamples)

    # epsilon = 0.25
    # EGreedyInstance = EGreedy(numArms, payoutSTD, epsilon)
    # cumRewardEGreedy = EGreedyInstance.runAlgorithm(numSamples)

    # print(cumRewardUCB)
    # print(cumRewardEGreedy)

    # testAllUCBAccuracy()
    testAllRegret()