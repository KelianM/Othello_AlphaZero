from othello import OthelloState
from policy_iter import PolicyIteration

if __name__ == '__main__':
    board_size = 6
    model = OthelloState(board_size)
    nnet = PolicyIteration(model)