import numpy as np
from collections import defaultdict

from othello import OthelloState

class MCTS:
    def __init__(self, nnet, c_puct = 1):
        self.nnet = nnet # Neural Net used for policy prior & value prediction
        self.c_puct = c_puct # Exploration parameters

        self.visited = {} # Dict of visited (expanded) states
        self.P = {} # Dict of state policies
        self.W = defaultdict(lambda: defaultdict(int)) # Dict of total state-action values
        self.Q = defaultdict(lambda: defaultdict(int)) # Dict of mean state-action values
        self.N = defaultdict(lambda: defaultdict(int)) # Dict of state-action visit counts

    def get_move_score(self, s, a):
        """ UCT move score with prior as in AlphaZero
        """
        return self.Q[s][a] + self.c_puct*self.P[s][a]*np.sqrt(sum(self.N[s]))/(1+self.N[s][a])
                                                              
    def uct_search(self, root_model: OthelloState, num_iters=100):
        for i in range(num_iters):
            self.uct_search_iter(model=root_model.Clone())
    
    
    def uct_search_iter(self, model: OthelloState, nnet):
        """ Recursive UCT search implementation.
            Uses the NeuralNet (nnet) to predict for value and a prior policy (pi)"""
        s = model.CloneState()
        if model.GetValidMoves() == []:
            # We don't have to negate this value since `playerJustMoved` is already the previous player (no move was played)
            return model.GetResult(model.playerJustMoved)

        if s not in self.visited:
            self.visited.add(s)
            self.P[s], v = self.nnet.predict(s) # Set this state's policy to a predicted prior
            # Mask out invalid moves
            valid_moves = model.GetValidMoves()
            valid_mask = [move in valid_moves for move in model.GetAllMoves()]
            self.P[s] *= valid_mask
            # If the prior is empty, set it to uniform distribution for valid moves
            if sum(self.P[s]) == 0:
                self.P[s] = np.ones_like(self.P[s]) * valid_mask / len(valid_moves)
            return -v # Negate to give value according to previous player
    
        a = max(model.GetAllMoves(), key = lambda a: self.get_move_score(s, a))        
        model.DoMove(a)
        v = self.uct_search()

        self.W[s][a] += v
        self.N[s][a] += 1
        self.Q[s][a] = self.W[s][a] / self.N[s][a]
        return -v # Negate to give value according to previous player
    
    def pi(self, model: OthelloState):
        """ Return the policy for a model in a given state.
        """
        s = model.GetState()
        total_visits = sum(self.N[s].values())  # Total visits to the state s
        # Policy for s is the number of visits to each edge (s, a) normalised by total visits to s
        return [self.N[s][a] / total_visits for a in model.GetValidMoves()]