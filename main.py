from testKnownWorlds import testKnownWorlds
from testUnknownWorldsDiscrete import testUnknownWorldsQLearnDiscrete, testUnknownWorldsSARSADiscrete
from testUnknownWorldsContinuous import testUnknownWorldsQLearnContinuous, testUnknownWorldsSARSAContinuous
from testMultiArm import testMultiArm

from testUnknownWorldsContinuous import runUnknownWorldTest, getDataset

def testUnknownWorlds():
    dataset = getDataset()
    print()
    
def main():
    # testKnownWorlds()
    # testUnknownWorldsQLearnDiscrete()
    # testUnknownWorldsSARSADiscrete()
    testUnknownWorldsQLearnContinuous()
    # testUnknownWorldsSARSAContinuous()
    # testUnknownWorlds()
    # testMultiArm()
    
if __name__ == '__main__':
    main()