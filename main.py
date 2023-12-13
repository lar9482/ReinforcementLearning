from testKnownWorlds import runKnownWorldTest, getKnownWorldDataset
from testUnknownWorldsContinuous import runUnknownWorldTest_Continuous, getContinuousDataset
from testUnknownWorldsDiscrete import runUnKnownWorldTest_Discrete, getDiscreteDataset
from testMultiArm import testMultiArm

from multiprocessing import Process, Manager

import sys

def testKnownWorlds():
    dataset = getKnownWorldDataset()
    with Manager() as manager:
        allProcesses = []
        lock = manager.Lock()

        for parameter in dataset:
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

def testUnknownWorlds_Discrete():
    dataset = getDiscreteDataset()
    for worldName in list(dataset.keys()):
        parameterList = dataset[worldName]

        with Manager() as manager:
            allProcesses = []
            lock = manager.Lock()

            for parameter in parameterList:

                process = Process(
                    target=runUnKnownWorldTest_Discrete, 
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

def testUnknownWorlds_Continuous():
    dataset = getContinuousDataset()
    numParametersInPool = 18
    parameterList = []
    for worldName in list(dataset.keys()):
        for parameter in dataset[worldName]:
            parameterList.append(parameter)
    
    # Dividing all parameters into sub-pools of size 
    # 'numParametersInPool' or less
    parameterPool = [
        parameterList[i:i+numParametersInPool] 
        for i in range(0, len(parameterList), numParametersInPool)
    ]

    for pool in parameterPool:
        with Manager() as manager:
            allProcesses = []
            lock = manager.Lock()

            for parameter in pool:
                allProcesses.append(Process(
                    target=runUnknownWorldTest_Continuous,
                    args=(parameter, lock)
                ))
            
            for process in allProcesses:
                process.start()
            
            for process in allProcesses:
                process.join()
    
def main():
    args = sys.argv
    if (args.__contains__('-known')):
        testKnownWorlds()
    
    if (args.__contains__('-unknownDis')):
        testUnknownWorlds_Discrete()
    
    if (args.__contains__('-unknownCont')):
        testUnknownWorlds_Continuous()

    if (args.__contains__('-bandit')):
        testMultiArm()

if __name__ == '__main__':
    main()