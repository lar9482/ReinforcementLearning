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

    testAllUCBAccuracy()
    testAllRegret()