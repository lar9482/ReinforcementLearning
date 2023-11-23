from testKnownWorlds import testKnownWorlds
from testUnknownWorldsDiscrete import testUnknownWorldsQLearnDiscrete, testUnknownWorldsSARSADiscrete
from testUnknownWorldsContinuous import testUnknownWorldsQLearnContinuous, testUnknownWorldsSARSAContinuous
from testMultiArm import testMultiArm
def main():
    # testKnownWorlds()
    # testUnknownWorldsQLearnDiscrete()
    # testUnknownWorldsSARSADiscrete()
    # testUnknownWorldsQLearnContinuous()
    # testUnknownWorldsSARSAContinuous()
    testMultiArm()
    
if __name__ == '__main__':
    main()