import random

from othello import OthelloState
from mcts import MCTS
from nnet import AlphaZeroSimpleCNN

def selfPlayEpisode(model: OthelloState, nnet, num_iters = 100, num_explore_steps=10, c_puct = 1):
    examples = []
    model = model.Clone()
    mcts = MCTS(nnet, c_puct=c_puct)

    num_steps = 0
    # Until game end
    while model.GetValidMoves() != []:
        # Explore with large tau for first `num_explore_steps`
        tau = 1 if num_steps < num_explore_steps else 1e-8
        # Perform MCTS for num_iters
        mcts.uct_search(model, num_iters=num_iters)
        # Store the current state & MCTS improved policy for the state
        s = model.CloneState()        
        p = mcts.pi(s, tau)
        examples.append([s, p, None]) 
        # Play according to the improved policy
        a = random.choice(len(p), p=p)
        model.DoMove(a)
        num_steps += 1
    
    # Assign rewards to all examples
    reward = model.GetResult(playerjm = 1) # From perspective of player 1
    for i in range(len(examples)):
        examples[i][2] = reward
        reward = -reward # Reward alternates each step for opposite player's perspective

    return examples
 
def pit(model: OthelloState, nnet1, nnet2, num_eps=100, num_iters=100, c_puct = 1):
    """ Pits to nnet1 (player 1) and nnet2 (player 2) against each other.
        Returns win rate from perspective of nnet1.
    """
    total_wins = 0
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
        if model.GetResult(playerjm=1) > 1: total_wins += 1 # Result from player 1's perspective
    # return reward rate
    return total_wins/num_eps

def PolicyIteration(model: OthelloState, numPolicyIters=1000, numEpsSP=100, numEpsPit=50, numMctsIters=100, c_puct=1, win_thresh=0.55, verbose=False):
    nnet = AlphaZeroSimpleCNN.init(sz=model.size, num_actions=model.GetNumMoves())
    examples = []    
    for i in range(numPolicyIters):
        for e in range(numEpsSP):
            examples += selfPlayEpisode(model, nnet, num_iters=numMctsIters, c_puct=c_puct)
        new_nnet = nnet.train(examples)                  
        win_rate = pit(model, nnet1=new_nnet, nnet2=nnet, num_eps=numEpsPit, num_iters=numMctsIters, c_puct=c_puct)
        if win_rate > win_thresh: 
            nnet = new_nnet
            if verbose:
                print(f"Network improved with win rate {win_rate}!")
    return nnet