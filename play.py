import random

from othello import OthelloState
from mcts import MCTS

def getUserMove(valid_moves):
    prompt = "Choose a move:\n"
    for index, move in enumerate(valid_moves, start=1):
        prompt += f"{index}: {move}\n"

    try:
        choice = int(input(prompt))  # Show the full prompt including the list of moves
        if 1 <= choice <= len(valid_moves):  # Ensure the choice is within valid range
            return valid_moves[choice - 1]
        else:
            print("Invalid choice. Please choose a valid move number.")
    except ValueError:
        print("That's not an integer. Please try again.")

def Play(model: OthelloState, nnet, args):
    try:
        human_player = int(input("Choose which player to play (1 or 2): "))
        if human_player not in [1, 2]:
            print("Invalid player. Please choose either 1 or 2.")
    except ValueError:
        print("Invalid input. Please enter a number.")

    episode_model = model.Clone()
    mcts = MCTS(nnet, all_moves=episode_model.AllMoves, c_puct=args.cpuct)

    while episode_model.GetValidMoves() != []:
        if episode_model.GetPlayerToMove() == human_player:
            valid_moves = episode_model.GetValidMoves()
            a = getUserMove(valid_moves)
        else:
            # Perform MCTS for num_iters
            mcts.uct_search(episode_model, num_iters=args.numMctsIters)
            # Store the current state & MCTS improved policy for the state
            s = episode_model.GetState()        
            p = mcts.pi(s, temp=0)
            # Play according to the improved policy
            a = random.choices(episode_model.AllMoves, weights=p)[0]
            print(f"AI played {a}")
        episode_model.DoMove(a)
        print('\n' + episode_model.__repr__())

    result = episode_model.GetResult(human_player)
    print("You Win!" if result == 1 else ("Nobody Wins!" if result == 0 else "AI Wins!"))