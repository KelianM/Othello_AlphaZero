from othello import OthelloState
from policy_iter import PolicyIteration

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

args = dotdict({
    'batchSize': 32,
    'epochs': 50,
    'learningRate': 0.001,
    'numPolicyIters': 10,
    'numEpsSp': 100,
    'numEpsPit': 30,
    'numExploreIters': 10,
    'numMctsIters': 30,
    'cpuct': 1.414,
    'numItersForTrainHist': 20,
    'winThresh': 0.55,
    'verbose': True
})

if __name__ == '__main__':
    board_size = 6
    model = OthelloState(board_size)
    nnet = PolicyIteration(
        model,
        args
    )