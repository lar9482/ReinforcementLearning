from MultiArm.UCB import UCB
from MultiArm.EGreedy import EGreedy
import matplotlib.pyplot as plt
from openpyxl import Workbook
import openpyxl
import os

class RegretAndNumPullsParameter:
    def __init__(self, numActions, payoutSTD):
        self.numActions = numActions
        self.payoutSTD = payoutSTD

def getRegretDataset():
    numActionsOptions = [2, 3, 4, 5]
    payoutSTDOptions = [0.1, 0.25, 0.5]
    dataset = []
    for numActions in numActionsOptions:
        for payoutSTD in payoutSTDOptions:
            parameter = RegretAndNumPullsParameter(
                numActions,
                payoutSTD
            )

            dataset.append(parameter)
    
    return dataset

def testRegret(RegretAndNumPullsParameter):
    numActions = RegretAndNumPullsParameter.numActions
    payoutSTD = RegretAndNumPullsParameter.payoutSTD
    numSamples = 100
    numTrials = 100

    regretPerTrial_UCB = {arm: [] for arm in range(numActions)}
    regretPerTrial_EGreedy_10Percent = {arm: [] for arm in range(numActions)}
    regretPerTrial_EGreedy_20Percent = {arm: [] for arm in range(numActions)}
    regretPerTrial_EGreedy_30Percent = {arm: [] for arm in range(numActions)}
    for trial in range(0, numTrials):

        UCBInstance = UCB(numActions, payoutSTD)
        EGreedy_10Percent = EGreedy(numActions, payoutSTD, 0.1)
        EGreedy_20Percent = EGreedy(numActions, payoutSTD, 0.2)
        EGreedy_30Percent = EGreedy(numActions, payoutSTD, 0.3)

        UCBInstance.runAlgorithm(numSamples)
        EGreedy_10Percent.runAlgorithm(numSamples)
        EGreedy_20Percent.runAlgorithm(numSamples)
        EGreedy_30Percent.runAlgorithm(numSamples)

        for arm in range(numActions):
            regretPerTrial_UCB[arm].append(UCBInstance.getActualRegret(arm))
            regretPerTrial_EGreedy_10Percent[arm].append(EGreedy_10Percent.getActualRegret(arm))
            regretPerTrial_EGreedy_20Percent[arm].append(EGreedy_20Percent.getActualRegret(arm))
            regretPerTrial_EGreedy_30Percent[arm].append(EGreedy_30Percent.getActualRegret(arm))

    plotAvgRegret(
        RegretAndNumPullsParameter,
        regretPerTrial_UCB,
        regretPerTrial_EGreedy_10Percent,
        regretPerTrial_EGreedy_20Percent,
        regretPerTrial_EGreedy_30Percent
    )

def plotAvgRegret(
    RegretAndNumPullsParameter,
    regretPerTrial_UCB,
    regretPerTrial_EGreedy_10Percent,
    regretPerTrial_EGreedy_20Percent,
    regretPerTrial_EGreedy_30Percent

):
    plt.figure(figsize=(10, 5))

    for arm in range(RegretAndNumPullsParameter.numActions):
        plt.plot(
           [x for x in range(0, len(regretPerTrial_UCB[arm]))], 
           regretPerTrial_UCB[arm], 
           label="UCBRegret_Arm{0}".format(str(arm+1))
        ) 

        plt.plot(
           [x for x in range(0, len(regretPerTrial_EGreedy_10Percent[arm]))], 
           regretPerTrial_EGreedy_10Percent[arm], 
           label="EGreedyRegret0.1_Arm{0}".format(str(arm+1))
        ) 

        plt.plot(
           [x for x in range(0, len(regretPerTrial_EGreedy_20Percent[arm]))], 
           regretPerTrial_EGreedy_20Percent[arm], 
           label="EGreedyRegret0.2_Arm{0}".format(str(arm+1))
        )

        plt.plot(
           [x for x in range(0, len(regretPerTrial_EGreedy_30Percent[arm]))], 
           regretPerTrial_EGreedy_30Percent[arm], 
           label="EGreedyRegret0.3_Arm{0}".format(str(arm+1))
        )

    plt.title("Arm regret per trial with 100 samples: payoutSTD_{0} numArms_{1}".format(
            str(RegretAndNumPullsParameter.payoutSTD),
            str(RegretAndNumPullsParameter.numActions)
        )
    )
    plt.xlabel('Trial')
    plt.ylabel('Arm Regret')
    plt.legend(loc=(0.8, 0))

    fileName = './results/MultiArm/Regret/Regret_actions-{0}_STD-{1}.png'.format(
        str(RegretAndNumPullsParameter.numActions),
        str(RegretAndNumPullsParameter.payoutSTD)
    )

    plt.savefig(fileName)
    plt.clf()