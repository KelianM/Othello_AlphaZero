import random
from torch.utils.data import DataLoader
import torch.optim as optim

from othello import OthelloState
from dataset import GameDataset
from mcts import MCTS
from nnet import AlphaZeroSimpleCNN
from train import train

def selfPlay(model: OthelloState, nnet, num_eps = 30, num_iters = 20, num_explore_steps=5, c_puct = 1):
    examples = []
    for e in range(num_eps):
        episode_examples = []
        episode_model = model.Clone()
        mcts = MCTS(nnet, all_moves=episode_model.AllMoves, c_puct=c_puct)

        num_steps = 0
        # Until game end
        while episode_model.GetValidMoves() != []:
            # Explore with large tau for first `num_explore_steps`
            tau = 1 if num_steps < num_explore_steps else 0.25
            # Perform MCTS for num_iters
            mcts.uct_search(episode_model, num_iters=num_iters)
            # Store the current state & MCTS improved policy for the state
            s = episode_model.GetState()        
            p = mcts.pi(s, tau)
            episode_examples.append([s, p, None]) 
            # Play according to the improved policy
            a = random.choices(episode_model.AllMoves, weights=p)[0]
            episode_model.DoMove(a)
            num_steps += 1
        print(f'Self Play Episode {e}: Finished in {num_steps} steps')
        
        # Assign rewards to all examples from this episode
        reward = episode_model.GetResult(playerjm = 1) # From perspective of player 1
        for i in range(len(episode_examples)):
            episode_examples[i][2] = reward
            reward = -reward # Reward alternates each step for opposite player's perspective
        examples.extend(episode_examples)
    return examples
 
def pit(model: OthelloState, nnet1, nnet2, num_eps=100, num_iters=100, c_puct = 1):
    """ Pits to nnet1 (player 1) and nnet2 (player 2) against each other.
        Returns win rate from perspective of nnet1.
    """
    total_wins = 0
    for e in range(num_eps):
        episode_model = model.Clone()
        # Hold a separate MCTS instance for each playet (each nnet)
        mcts1 = MCTS(nnet=nnet1, all_moves=episode_model.AllMoves, c_puct=c_puct) # Player 1
        mcts2 = MCTS(nnet=nnet2, all_moves=episode_model.AllMoves, c_puct=c_puct) # Player 2

        # Until game end
        while episode_model.GetValidMoves() != []:
            # Perform MCTS for num_iters for the correct player's MCTS
            if episode_model.GetPlayerToMove() == 1:
                mcts1.uct_search(episode_model, num_iters=num_iters)
                p = mcts1.pi(episode_model)
            else:
                mcts2.uct_search(episode_model, num_iters=num_iters)
                p = mcts2.pi(episode_model)
            # Store the current state & MCTS improved policy for the state
            # Play according to the improved policy
            a = random.choice(len(p), p=p)
            episode_model.DoMove(a)
        if episode_model.GetResult(playerjm=1) > 1: total_wins += 1 # Result from player 1's perspective
    # return reward rate
    return total_wins/num_eps

def PolicyIteration(model: OthelloState, batch_size=32, epochs=100, lr=0.001, numPolicyIters=1000, numEpsSP=100, numEpsPit=50, numMctsIters=100, c_puct=1, win_thresh=0.55, verbose=False):
    nnet = AlphaZeroSimpleCNN(sz=model.size, num_actions=model.NumMoves)
    examples = []
    for i in range(numPolicyIters):
        # Generate self-play data using the current best nnet
        examples.extend(selfPlay(model, nnet, num_eps=numEpsSP, num_iters=numMctsIters, c_puct=c_puct))
        # Create the dataset and dataloader
        dataset = GameDataset(examples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        # Train a new checkpoint
        optimizer = optim.Adam(nnet.parameters(), lr=lr)
        new_nnet = train(nnet, optimizer=optimizer, epochs=epochs, dataloader=dataloader)
        # Pit the model's against each other and update the best model if it exceeds the win threshold
        win_rate = pit(model, nnet1=new_nnet, nnet2=nnet, num_eps=numEpsPit, num_iters=numMctsIters, c_puct=c_puct)
        if win_rate > win_thresh: 
            nnet = new_nnet
            if verbose:
                print(f"Network improved with win rate {win_rate}!")
    return nnet