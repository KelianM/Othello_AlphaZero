import numpy as np
from collections import defaultdict

from othello import OthelloState

class MCTS:
    def __init__(self, nnet, c_puct = 1):
        self.nnet = nnet # Neural-net that predicts a state's policy & values
        self.c_puct = c_puct # Exploration parameters

        self.visited = {} # Dict of visited (expanded) states
        self.P = {} # Dict of state policies
        self.Q = defaultdict(lambda: defaultdict(int)) # Dict of state-action values
        self.N = defaultdict(lambda: defaultdict(int)) # Dict of state-action visit counts

    def get_move_score(self, s, a):
        """ UCT move score with prior as in AlphaZero
        """
        return self.Q[s][a] + self.c_puct*self.P[s][a]*np.sqrt(sum(self.N[s]))/(1+self.N[s][a])
                                                              
    def uct_search(self, root_model: OthelloState, num_iters=100):
        for i in range(num_iters):
            self.uct_search_iter(root_model.Clone())
    
    def uct_search_iter(self, model: OthelloState):
        """ Recursive UCT search implementation.
        """
        s = model.CloneState()
        if model.GetMoves() == []:
            # We don't have to negate this value since `playerJustMoved` is the PREVIOUS player (no move was played)
            return model.GetResult(model.playerJustMoved)

        if s not in self.visited:
            self.visited.add(s)
            self.P[s], v = self.nnet.predict(s)
            return -v # Negate to give value according to previous player
    
        a = max(model.GetMoves(), key = lambda a: self.get_move_score(s, a))        
        model.DoMove(a)
        v = self.uct_search()

        self.Q[s][a] = (self.N[s][a]*self.Q[s][a] + v)/(self.N[s][a]+1)
        self.N[s][a] += 1
        return -v # Negate to give value according to previous player
    
    def pi(self, model: OthelloState):
        """ Return the policy for a model in a given state.
        """
        s = model.GetState()
        total_visits = sum(self.N[s].values())  # Total visits to the state s
        # Policy for s is the number of visits to each edge (s, a) normalised by total visits to s
        return [self.N[s][a] / total_visits for a in model.GetMoves()]