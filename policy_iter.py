import random
import logging
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim

from othello import OthelloState
from dataset import GameDataset
from mcts import MCTS
from nnet import AlphaZeroSimpleCNN
from train import train

log = logging.getLogger(__name__)

def selfPlay(model: OthelloState, nnet, args):
    examples = []
    for e in range(args.numEpsSp):
        episode_examples = []
        episode_model = model.Clone()
        mcts = MCTS(nnet, all_moves=episode_model.AllMoves, c_puct=args.cpuct)

        num_steps = 0
        # Until game end
        while episode_model.GetValidMoves() != []:
            # Explore with large temp for first `numExploreSteps`
            temp = 1 if num_steps < args.numExploreSteps else 0
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
        log.info(f'Self Play Episode {e}: Finished in {num_steps} steps')
        
        # Assign rewards to all examples from this episode
        reward = episode_model.GetResult(playerjm = 1) # From perspective of player 1
        for i in range(len(episode_examples)):
            episode_examples[i][2] = reward
            reward = -reward # Reward alternates each step for opposite player's perspective
        examples.extend(episode_examples)
    return examples
 
def pit(model: OthelloState, nnet1, nnet2, args):
    """ Pits to nnet1 (player 1) and nnet2 (player 2) against each other.
        Returns number of wins, draws and losses from the perspective of nnet1.
    """
    results = []
    for e in range(args.numEpsPit):
        # 50% chance of `nnet1` going first (being player 1)
        nnet1_player = 1 if random.random() > 0.5 else 2
        log.debug(f"Pit episode {e} gameplay for nnet1 = Player {nnet1_player}")

        episode_model = model.Clone()
        # Hold a separate MCTS instance for each player (each nnet)
        mcts1 = MCTS(nnet=nnet1, all_moves=episode_model.AllMoves, c_puct=args.cpuct)
        mcts2 = MCTS(nnet=nnet2, all_moves=episode_model.AllMoves, c_puct=args.cpuct)

        num_steps = 0
        # Until game end
        while episode_model.GetValidMoves() != []:
            # Perform MCTS for num_iters for the correct player's MCTS
            s = episode_model.GetState()
            # Temperature is high only on first steps to encourage different games
            temp = 1 if num_steps == 0 else 0
            if episode_model.GetPlayerToMove() == nnet1_player:
                mcts1.uct_search(episode_model, num_iters=args.numMctsIters)
                p = mcts1.pi(s, temp=temp)
            else:
                mcts2.uct_search(episode_model, num_iters=args.numMctsIters)
                p = mcts2.pi(s, temp=temp)
            # Store the current state & MCTS improved policy for the state
            # Play according to the improved policy
            a = random.choices(episode_model.AllMoves, weights=p)[0]
            episode_model.DoMove(a)
            num_steps += 1
            log.debug('\n' + episode_model.__repr__()) # Board representation for debugging
            
        results.append(episode_model.GetResult(playerjm=nnet1_player))
        log.debug(f'Game result: {results[-1]}')
        log.info(f'Pit Episode {e}: Finished in {num_steps} steps')
    # Return total wins (1), draws (0), and losses (-1) for player `playerjm`
    return sum(x == 1 for x in results), sum(x == 0 for x in results), sum(x == -1 for x in results)

def PolicyIteration(model: OthelloState, args):
    nnet = AlphaZeroSimpleCNN(sz=model.size, num_actions=model.NumMoves)
    best_nnet = AlphaZeroSimpleCNN(sz=model.size, num_actions=model.NumMoves)
    
    nnet.save_checkpoint(folder=args.checkpointDir, filename='temp.pth.tar')
    best_nnet.load_checkpoint(folder=args.checkpointDir, filename='temp.pth.tar')
    examples_history = []
    for i in range(args.numPolicyIters):
        log.info(f"Policy iteration step {i}")
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
        # Results are from the perspective of the new candidate model we are evaluating `nnet`
        num_wins, num_draws, num_losses = pit(model, nnet1=nnet, nnet2=best_nnet, args=args)
        log.info('New/Previous model wins : %d / %d ; Draws : %d' % (num_wins, num_losses, num_draws))
        win_rate = num_wins/args.numEpsPit
        nnet.save_checkpoint(folder=args.checkpointDir, filename='temp.pth.tar')
        if win_rate > args.winThresh:
            log.info(f"Accepting new model with win-rate {win_rate}.")
            nnet.save_checkpoint(folder=args.checkpointDir, filename='best.pth.tar')
            best_nnet.load_checkpoint(folder=args.checkpointDir, filename='best.pth.tar')
        else:
            log.info(f"Rejecting new model with win-rate {win_rate}.")
    return nnet