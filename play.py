import random

from othello import OthelloState
from mcts import MCTS
        
def selfPlayEpisode(model: OthelloState, nnet, num_iters = 100, c_puct = 1):
    examples = []
    model = model.Clone()
    mcts = MCTS(nnet, c_puct=c_puct)

    # Until game end
    while model.GetValidMoves() != []:
        # Perform MCTS for num_iters
        mcts.uct_search(model, num_iters=num_iters)
        # Store the current state & MCTS improved policy for the state
        s = model.CloneState()
        p = mcts.pi(model)
        examples.append([s, mcts.pi(s), None]) 
        # Play according to the improved policy
        a = random.choice(len(p), p=p)
        model.DoMove(a)
    
    # Assign rewards to all examples
    reward = model.GetResult(playerjm = 1) # From perspective of player 1
    for i in range(len(examples)):
        examples[i][2] = reward
        reward = -reward # Reward alternates each step for opposite player's perspective

    return examples
 
def pit(model: OthelloState, nnet1, nnet2, num_eps=100, num_iters=100, c_puct = 1):
    """ Pits to nnet1 (player 1) and nnet2 (player 2) against each other.
        Returns reward rate from perspective of nnet1.
    """
    total_reward = 0
    for e in range(num_eps):
        model = model.Clone()
        # Hold a separate MCTS instance for each playet (each nnet)
        mcts1 = MCTS(nnet=nnet1, c_puct=c_puct) # Player 1
        mcts2 = MCTS(nnet=nnet2, c_puct=c_puct) # Player 2

        # Until game end
        while model.GetValidMoves() != []:
            # Perform MCTS for num_iters for the correct player's MCTS
            if model.GetPlayerToMove() == 1:
                mcts1.uct_search(model, num_iters=num_iters)
                p = mcts1.pi(model)
            else:
                mcts2.uct_search(model, num_iters=num_iters)
                p = mcts2.pi(model)
            # Store the current state & MCTS improved policy for the state
            # Play according to the improved policy
            a = random.choice(len(p), p=p)
            model.DoMove(a)
        total_reward += model.GetResult(playerjm=1) # Reward from player 1's perspective
    # return reward rate
    return total_reward/num_eps

def PolicyIteration(model: OthelloState, numPolicyIters=1000, numEpsSP=100, numEpsPit=50, numMctsIters=100, c_puct=1, win_thresh=0.55):
    nnet = nnet.init(sz=model.size)
    examples = []    
    for i in range(numPolicyIters):
        for e in range(numEpsSP):
            examples += selfPlayEpisode(model, nnet, num_iters=numMctsIters, c_puct=c_puct)
        new_nnet = nnet.train(examples)                  
        frac_win = pit(model, nnet1=new_nnet, nnet2=nnet, num_eps=numEpsPit, num_iters=numMctsIters, c_puct=c_puct)
        if frac_win > win_thresh: 
            nnet = new_nnet         
    return nnet