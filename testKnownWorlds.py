from WorldKnown.domains.navigationProblem import getNavigationProblem
from WorldKnown.domains.wumpusWorld import getWumpusWorld
from WorldKnown.valueIteration import valueIteration
from WorldKnown.policyIteration import policyIteration

import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import openpyxl

from multiprocessing import Process, Manager

class testParameter_KnownWorlds:
    def __init__(self, worldName, world, discount, error):
        self.worldName = worldName
        self.world = world
        self.discount = discount
        self.error = error

def getKnownWorldDataset():
    worldOptions = {
        'navigation': getNavigationProblem(),
        'wumpus': getWumpusWorld()
    }

    discountOptions = [0.01, 0.1, 0.5, 0.9]
    errorOptions = [0.01, 0.1, 0.5, 0.9]
    dataset = []
    for worldName in list(worldOptions.keys()):
        world = worldOptions[worldName]
        for discount in discountOptions:
            for error in errorOptions:
                parameter = testParameter_KnownWorlds(
                    worldName,
                    world,
                    discount,
                    error,
                )

                dataset.append(parameter)

    return dataset

def runKnownWorldTest(testParameter_KnownWorlds, lock):

    k10 = 10
    k100 = 100
    k1000 = 1000
    worldUtility = valueIteration(
        testParameter_KnownWorlds.world,
        testParameter_KnownWorlds.discount,
        testParameter_KnownWorlds.error
    )

    worldPolicy_K10 = policyIteration(
        testParameter_KnownWorlds.world,
        testParameter_KnownWorlds.discount,
        k10
    )

    worldPolicy_K100 = policyIteration(
        testParameter_KnownWorlds.world,
        testParameter_KnownWorlds.discount,
        k100
    )

    worldPolicy_K1000 = policyIteration(
        testParameter_KnownWorlds.world,
        testParameter_KnownWorlds.discount,
        k1000
    )

    if (testParameter_KnownWorlds.worldName == 'navigation'):
        plotUtilityAndPolicyGrid(
            testParameter_KnownWorlds.world, worldUtility, worldPolicy_K10, 
            getFileNameGrid(testParameter_KnownWorlds, k10)
        )
        plotUtilityAndPolicyGrid(
            testParameter_KnownWorlds.world, worldUtility, worldPolicy_K100, 
            getFileNameGrid(testParameter_KnownWorlds, k100)
        )
        plotUtilityAndPolicyGrid(
            testParameter_KnownWorlds.world, worldUtility, worldPolicy_K1000, 
            getFileNameGrid(testParameter_KnownWorlds, k1000)
        )
    elif (testParameter_KnownWorlds.worldName == 'wumpus'):
        plotUtilityAndPolicyWumpus(
            testParameter_KnownWorlds.world, worldUtility, worldPolicy_K10, True, True,
            getFileNameWumpus(testParameter_KnownWorlds, k10, True, True)
        )
        plotUtilityAndPolicyWumpus(
            testParameter_KnownWorlds.world, worldUtility, worldPolicy_K10, True, False,
            getFileNameWumpus(testParameter_KnownWorlds, k10, True, False)
        )
        plotUtilityAndPolicyWumpus(
            testParameter_KnownWorlds.world, worldUtility, worldPolicy_K10, False, True,
            getFileNameWumpus(testParameter_KnownWorlds, k10, False, True)
        )
        plotUtilityAndPolicyWumpus(
            testParameter_KnownWorlds.world, worldUtility, worldPolicy_K10, False, False,
            getFileNameWumpus(testParameter_KnownWorlds, k10, False, False)
        )
        ###################################################################################
        plotUtilityAndPolicyWumpus(
            testParameter_KnownWorlds.world, worldUtility, worldPolicy_K100, True, True,
            getFileNameWumpus(testParameter_KnownWorlds, k100, True, True)
        )
        plotUtilityAndPolicyWumpus(
            testParameter_KnownWorlds.world, worldUtility, worldPolicy_K100, True, False,
            getFileNameWumpus(testParameter_KnownWorlds, k100, True, False)
        )
        plotUtilityAndPolicyWumpus(
            testParameter_KnownWorlds.world, worldUtility, worldPolicy_K100, False, True,
            getFileNameWumpus(testParameter_KnownWorlds, k100, False, True)
        )
        plotUtilityAndPolicyWumpus(
            testParameter_KnownWorlds.world, worldUtility, worldPolicy_K100, False, False,
            getFileNameWumpus(testParameter_KnownWorlds, k100, False, False)
        )
        ###################################################################################
        plotUtilityAndPolicyWumpus(
            testParameter_KnownWorlds.world, worldUtility, worldPolicy_K1000, True, True,
            getFileNameWumpus(testParameter_KnownWorlds, k1000, True, True)
        )
        plotUtilityAndPolicyWumpus(
            testParameter_KnownWorlds.world, worldUtility, worldPolicy_K1000, True, False,
            getFileNameWumpus(testParameter_KnownWorlds, k1000, True, False)
        )
        plotUtilityAndPolicyWumpus(
            testParameter_KnownWorlds.world, worldUtility, worldPolicy_K1000, False, True,
            getFileNameWumpus(testParameter_KnownWorlds, k1000, False, True)
        )
        plotUtilityAndPolicyWumpus(
            testParameter_KnownWorlds.world, worldUtility, worldPolicy_K1000, False, False,
            getFileNameWumpus(testParameter_KnownWorlds, k1000, False, False)
        )
    
    lock.acquire()
    saveValueSum(testParameter_KnownWorlds, worldUtility)
    lock.release()

def plotUtilityAndPolicyGrid(world, utility, policy, fileName = ''):

    # In order for seaborn to parse the utility and policy, they need to be loaded into 
    # numpy arrays
    utilityMap = np.zeros(shape=(world.height, world.width))
    policyMap = [[' ' for _ in range(world.width)] for _ in range(world.height)]

    ActionUp = 1
    ActionDown = 2
    ActionLeft = 3
    ActionRight = 4
    for y in range(0, world.height):
        for x in range(0, world.width):
            utilityMap[y, x] = utility[(x, y)]

            # In seaborn, the map is oriented where up and down are opposite oriented
            if policy[(x, y)].value == ActionUp:
                policyMap[y][x] = '↓'
            elif policy[(x, y)].value == ActionDown:
                policyMap[y][x] = '↑'
            elif policy[(x, y)].value == ActionLeft:
                policyMap[y][x] = '←'
            elif policy[(x, y)].value == ActionRight:
                policyMap[y][x] = '→'
    
    formatted_text = (
        np.asarray(["{0}\n{1:.2f}".format( text, data) 
        for text, data in zip((
            np.array(policyMap)
        ).flatten(), utilityMap.flatten())])
    ).reshape(world.height, world.width) 

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    ax = sns.heatmap(utilityMap, annot=formatted_text, fmt="") 

    filePath = './results/WorldKnown/{0}.png'.format(
        fileName
    )
    plt.savefig(filePath)
    plt.clf()


def plotUtilityAndPolicyWumpus(
        world, utility, policy, 
        hasGold = True, hasImmunity = True, fileName = ''
    ):
    # In order for seaborn to parse the utility and policy, they need to be loaded into 
    # numpy arrays
    utilityMap = np.zeros(shape=(world.height, world.width))
    policyMap = [[' ' for _ in range(world.width)] for _ in range(world.height)]

    ActionUp = 1
    ActionDown = 2
    ActionLeft = 3
    ActionRight = 4
    ActionPickup = 5
    for y in range(0, world.height):
        for x in range(0, world.width):
            utilityMap[y, x] = utility[(x, y, hasGold, hasImmunity)]

            # In seaborn, the map is oriented where up and down are opposite oriented
            if policy[(x, y, hasGold, hasImmunity)].value == ActionUp:
                policyMap[y][x] = '↓'
            elif policy[(x, y, hasGold, hasImmunity)].value == ActionDown:
                policyMap[y][x] = '↑'
            elif policy[(x, y, hasGold, hasImmunity)].value == ActionLeft:
                policyMap[y][x] = '←'
            elif policy[(x, y, hasGold, hasImmunity)].value == ActionRight:
                policyMap[y][x] = '→'
            elif policy[(x, y, hasGold, hasImmunity)].value == ActionPickup:
                policyMap[y][x] = 'PickUp'

    formatted_text = (
        np.asarray(["{0}\n{1:.2f}".format( text, data) 
        for text, data in zip((
            np.array(policyMap)
        ).flatten(), utilityMap.flatten())])
    ).reshape(world.height, world.width) 

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    ax = sns.heatmap(utilityMap, annot=formatted_text, fmt="")
    
    filePath = './results/WorldKnown/{0}.png'.format(
        fileName
    )

    plt.savefig(filePath)
    plt.clf()

def saveValueSum(testParameter_KnownWorlds, utility):

    statWorkbook = openpyxl.load_workbook('./results/WorldKnown/ValueSum.xlsx')
    statSheet = statWorkbook.active
    sumValue = sum(list(utility.values()))
    statSheet.append([
        testParameter_KnownWorlds.worldName,
        testParameter_KnownWorlds.discount,
        testParameter_KnownWorlds.error,
        sumValue,
        str("{:.3f}".format(sumValue / len(utility)))
    ])

    statWorkbook.save('./results/WorldKnown/ValueSum.xlsx')
    statWorkbook.close()

def getFileNameGrid(testParameter_KnownWorlds, k):
    return 'world-{0}_discount-{1}_error-{2}_k-{3}'.format(
        testParameter_KnownWorlds.worldName,
        testParameter_KnownWorlds.discount,
        testParameter_KnownWorlds.error,
        k
    )

def getFileNameWumpus(testParameter_KnownWorlds, k, hasGold, hasImmunity):
    return 'world-{0}_discount-{1}_error-{2}_k-{3}_gold-{4}_imm-{5}'.format(
        testParameter_KnownWorlds.worldName,
        testParameter_KnownWorlds.discount,
        testParameter_KnownWorlds.error,
        k,
        hasGold,
        hasImmunity
    )

def testKnownWorlds():
    dataset = getKnownWorldDataset()
    for worldName in list(dataset.keys()):
        parameterList = dataset[worldName]

        with Manager() as manager:
            allProcesses = []
            lock = manager.Lock()

            for parameter in parameterList:

                process = Process(
                    target=runKnownWorldTest, 
                    args=(
                        parameter,
                        lock
                    )
                )
                allProcesses.append(process)
            
            for process in allProcesses:
                process.start()

            for process in allProcesses:
                process.join()
