from MDP.wumpus import WumpusMDP

def getWumpusWorld():
    mdp = WumpusMDP(8, 10, -0.1)

    ## add wumpus
    mdp.add_obstacle('wumpus', [6, 9], -100) # super hurtful wumpus
    mdp.add_obstacle('wumpus', [6, 8])
    mdp.add_obstacle('wumpus', [6, 7])
    mdp.add_obstacle('wumpus', [7, 5])

    ## add pits
    mdp.add_obstacle('pit', [2, 0])
    mdp.add_obstacle('pit', [2, 1])
    mdp.add_obstacle('pit', [2, 2], -0.5) # weaker pit
    mdp.add_obstacle('pit', [5, 0])
    mdp.add_obstacle('pit', [6, 1])

    ## add goal
    mdp.add_obstacle('goal', [7, 9])

    ## add objects
    mdp.add_object('gold', [0, 9])
    mdp.add_object('gold', [7, 0])
    mdp.add_object('immune', [6, 0])

    mdp.add_object('gold', [1, 1])
    mdp.add_object('immune', [1, 2])

    return mdp