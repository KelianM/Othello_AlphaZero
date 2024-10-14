import logging

import coloredlogs

from othello import OthelloState
from policy_iter import PolicyIteration
from nnet import AlphaZeroSimpleCNN
from play import Play

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

args = dotdict({
    'batchSize': 32,
    'epochs': 75,
    'learningRate': 0.001,
    'numPolicyIters': 1,
    'numEpsSp': 50,
    'numEpsPit': 30,
    'numExploreSteps': 3,
    'numMctsIters': 30,
    'cpuct': 1.414,
    'numItersForTrainHist': 20,
    'winThresh': 0.55,
    'checkpointDir': 'checkpoints',
    'loadNnet': False
})

if __name__ == '__main__':
    board_size = 4
    log.info(f'Playing Othello with board size {board_size}')
    model = OthelloState(board_size)
    if args.loadNnet:
        nnet = AlphaZeroSimpleCNN(board_size, num_actions=board_size**2)
        nnet.load_checkpoint(folder=args.checkpointDir, filename='best.pth.tar')
    else:
        nnet = PolicyIteration(
            model,
            args
        )
    Play(model, nnet, args)