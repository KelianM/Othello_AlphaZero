import logging

import coloredlogs

from othello import OthelloState
from policy_iter import PolicyIteration
log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

args = dotdict({
    'batchSize': 32,
    'epochs': 30,
    'learningRate': 0.001,
    'numPolicyIters': 10,
    'numEpsSp': 10,
    'numEpsPit': 10,
    'numExploreSteps': 3,
    'numMctsIters': 30,
    'cpuct': 1.414,
    'numItersForTrainHist': 20,
    'winThresh': 0.55,
    'checkpointDir': 'checkpoints'
})

if __name__ == '__main__':
    board_size = 4
    log.info(f'Playing Othello with board size {board_size}')
    model = OthelloState(board_size)
    nnet = PolicyIteration(
        model,
        args
    )