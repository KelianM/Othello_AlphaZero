import random
import copy
from torch.utils.data import DataLoader
import torch.optim as optim

from othello import OthelloState
from dataset import GameDataset
from mcts import MCTS
from nnet import AlphaZeroSimpleCNN
from train import train

def selfPlay(model: OthelloState, nnet, args):
    examples = []
    for e in range(args.numEpsSp):
        episode_examples = []
        episode_model = model.Clone()
        mcts = MCTS(nnet, all_moves=episode_model.AllMoves, c_puct=args.cpuct)

        num_steps = 0
        # Until game end
        while episode_model.GetValidMoves() != []:
            # Explore with large tau for first `num_explore_steps`
            temp = 1 if num_steps < args.numExploreIters else 0
            # Perform MCTS for num_iters
            mcts.uct_search(episode_model, num_iters=args.numMctsIters)
            # Store the current state & MCTS improved policy for the state
            s = episode_model.GetState()        
            p = mcts.pi(s, temp)
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
 
def pit(model: OthelloState, nnet1, nnet2, args):
    """ Pits to nnet1 (player 1) and nnet2 (player 2) against each other.
        Returns win rate from perspective of nnet1.
    """
    total_wins = 0
    for e in range(args.numEpsPit):
        episode_model = model.Clone()
        # Hold a separate MCTS instance for each playet (each nnet)
        mcts1 = MCTS(nnet=nnet1, all_moves=episode_model.AllMoves, c_puct=args.cpuct) # Player 1
        mcts2 = MCTS(nnet=nnet2, all_moves=episode_model.AllMoves, c_puct=args.cpuct) # Player 2

        # Until game end
        while episode_model.GetValidMoves() != []:
            # Perform MCTS for num_iters for the correct player's MCTS
            s = episode_model.GetState()
            if episode_model.GetPlayerToMove() == 1:
                mcts1.uct_search(episode_model, num_iters=args.numMctsIters)
                p = mcts1.pi(s, temp=0)
            else:
                mcts2.uct_search(episode_model, num_iters=args.numMctsIters)
                p = mcts2.pi(s, temp=0)
            # Store the current state & MCTS improved policy for the state
            # Play according to the improved policy
            a = random.choices(episode_model.AllMoves, weights=p)[0]
            episode_model.DoMove(a)
        if episode_model.GetResult(playerjm=1) > 1: total_wins += 1 # Result from player 1's perspective
    # return reward rate
    return total_wins/args.numEpsPit

def PolicyIteration(model: OthelloState, args):
    nnet = AlphaZeroSimpleCNN(sz=model.size, num_actions=model.NumMoves)
    best_nnet = copy.deepcopy(nnet)
    examples_history = []
    for i in range(args.numPolicyIters):
        # Generate self-play data using the current best nnet
        examples_history.append(selfPlay(model, nnet, args=args))
        # When we have more than `numItersForTrainHist` chop off the oldest set of examples
        if len(examples_history) > args.numItersForTrainHist:
            examples_history = examples_history[1:]
        train_examples = []
        for e in examples_history:
            train_examples.extend(e)
        # Create the dataset and dataloader
        dataset = GameDataset(train_examples)
        dataloader = DataLoader(dataset, batch_size=args.batchSize, shuffle=True, pin_memory=True)
        # Train a new checkpoint
        optimizer = optim.Adam(nnet.parameters(), lr=args.learningRate)
        train(nnet, optimizer=optimizer, epochs=args.epochs, dataloader=dataloader)
        # Pit the model's against each other and update the best model if it exceeds the win threshold
        win_rate = pit(model, nnet1=nnet, nnet2=best_nnet, args=args)
        if win_rate > args.winThresh: 
            best_nnet = copy.deepcopy(nnet)
            if args.verbose:
                print(f"Network improved with win rate {win_rate}!")
    return nnet