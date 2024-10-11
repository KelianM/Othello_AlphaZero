from othello import OthelloState
from policy_iter import PolicyIteration

# Define constants
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_POLICY_ITERS = 10
NUM_EPS_SP = 1
NUM_EPS_PIT = 10
NUM_MCTS_ITERS = 10
C_PUCT = 1
WIN_THRESH = 0.55
VERBOSE = False

if __name__ == '__main__':
    board_size = 6
    model = OthelloState(board_size)
    nnet = PolicyIteration(
        model,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        numPolicyIters=NUM_POLICY_ITERS,
        numEpsSP=NUM_EPS_SP,
        numEpsPit=NUM_EPS_PIT,
        numMctsIters=NUM_MCTS_ITERS,
        c_puct=C_PUCT,
        win_thresh=WIN_THRESH,
        verbose=VERBOSE
    )