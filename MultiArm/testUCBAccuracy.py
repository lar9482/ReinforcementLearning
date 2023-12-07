from MultiArm.UCB import UCB

import matplotlib.pyplot as plt
from openpyxl import Workbook
import openpyxl
import os

class DecisionAccuracyParameter:
    def __init__(self, numActions, payoutSTD):
        self.numActions = numActions
        self.payoutSTD = payoutSTD

def getDecisionAccuracyDataset():
    numActionsOptions = [1, 2, 3, 4, 5, 7, 8, 9, 10]
    payoutSTDOptions = [0.01, 0.1, 0.5, 1]
    dataset = []
    for numActions in numActionsOptions:
        for payoutSTD in payoutSTDOptions:
            parameter = DecisionAccuracyParameter(
                numActions,
                payoutSTD
            )

            dataset.append(parameter)
    
    return dataset

def testUCBAccuracy(DecisionAccuracyParameter):
    numActions = DecisionAccuracyParameter.numActions
    payoutSTD = DecisionAccuracyParameter.payoutSTD
    maxMultiple = 100

    multiplesOptions = list(range(maxMultiple))

    expectedRewardsPerM = {arm: [] for arm in range(numActions)}
    actualRewardsPerM = {arm: [] for arm in range(numActions)}

    for multiple in multiplesOptions:
        numSamples = multiple * numActions

        UCBInstance = UCB(numActions, payoutSTD)
        UCBInstance.runAlgorithm(numSamples)
        for arm in range(numActions):
            expectedRewardsPerM[arm].append(
                UCBInstance.getExpectedRewardMean(arm)
            )

            actualRewardsPerM[arm].append(
                UCBInstance.getActualRewardMean(arm)
            )

    saveAvgDiffBetween_ExpectedAndActual(
        numActions,
        expectedRewardsPerM,
        actualRewardsPerM
    )
    plotExpectedAndActualRewardMeans(
        DecisionAccuracyParameter,
        expectedRewardsPerM, 
        actualRewardsPerM,
        maxMultiple
    )

def saveAvgDiffBetween_ExpectedAndActual(
    numActions,
    expectedRewardsPerM,
    actualRewardsPerM
):
    fileName = './results/MultiArm/DecisionAccuracy/DiffAccuracy_actions-{0}.xlsx'.format(str(numActions))
    if not os.path.exists(fileName):
        workbook = Workbook()

        sheet = workbook.active
        sheet.title = "Data"

        header = []
        for arm in range(numActions):
            header.append(
                'AvgExpected_{0}'.format(str(arm+1)),
            )
            header.append(
                'AvgActual_{0}'.format(str(arm+1))
            )

        sheet.append(header)
        workbook.save(fileName)
    
    dataRow = []
    for arm in range(numActions):
        avgExpected = sum(expectedRewardsPerM[arm])  / len(expectedRewardsPerM[arm])
        avgActual = sum(actualRewardsPerM[arm]) / len(actualRewardsPerM[arm])
        dataRow.append(avgExpected)
        dataRow.append(avgActual)

    workbook = openpyxl.load_workbook(fileName)
    sheet = workbook.active

    sheet.append(dataRow)
    workbook.save(fileName)
    workbook.close()

def plotExpectedAndActualRewardMeans(
    DecisionAccuracyParameter,
    expectedRewardsPerSample, 
    actualRewardsPerSample,
    maxMultiple
):
    plt.figure(figsize=(9, 5))
    for arm in range(DecisionAccuracyParameter.numActions):
        plt.plot(
           [x for x in range(0, maxMultiple)], 
           expectedRewardsPerSample[arm], 
           label="expectedArm: {0}".format(str(arm+1))
        ) 

        plt.plot(
           [x for x in range(0, maxMultiple)], 
           actualRewardsPerSample[arm], 
           label="actualArm: {0}".format(str(arm+1))
        )
    
    plt.title("Number of actions a: {0}".format(str(DecisionAccuracyParameter.numActions)))
    plt.xlabel('Multiple m, which determined h samples by h=a*m')
    plt.ylabel('Reward Mean')
    plt.legend(loc=(0.9, 0))

    fileName = './results/MultiArm/DecisionAccuracy/DecisionAccuracy_actions-{0}_STD-{1}.png'.format(
        str(DecisionAccuracyParameter.numActions),
        str(DecisionAccuracyParameter.payoutSTD)
    )
    plt.savefig(fileName)
    plt.clf()

