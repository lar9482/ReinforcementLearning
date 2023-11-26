class DecisionAccuracyParameter:
    def __init__(self, numActions, multiples):
        self.numActions = numActions
        self.multiples = multiples
    
def getDecisionAccuracyDataset():
    numActionsOptions = [2, 5, 10, 25]
    multiplesOptions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dataset = []
    for numActions in numActionsOptions:
        for multiples in multiplesOptions:
            parameter = DecisionAccuracyParameter(
                numActions,
                multiples
            )

            dataset.append(parameter)
    
    return dataset
