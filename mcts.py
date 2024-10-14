import torch
import numpy as np
from collections import defaultdict

from othello import OthelloState

class MCTS:
    def __init__(self, nnet, all_moves, c_puct = 1):
        self.nnet = nnet # Neural Net used for policy prior & value prediction
        self.all_moves = all_moves # All possible actions
        self.c_puct = c_puct # Exploration parameters

        self.visited = [] # Dict of visited (expanded) states
        self.P = {} # Dict of state policies
        self.W = defaultdict(lambda: defaultdict(int)) # Dict of total state-action values
        self.Q = defaultdict(lambda: defaultdict(int)) # Dict of mean state-action values
        self.N = defaultdict(lambda: defaultdict(int)) # Dict of state-action visit counts
                                                              
    def uct_search(self, root_model: OthelloState, num_iters=100):
        for i in range(num_iters):
            self._uct_search_recursive(model=root_model.Clone())
    
    def _uct_search_recursive(self, model: OthelloState):
        """ Recursive UCT search implementation.
            Uses the NeuralNet (nnet) to predict for value v and a prior policy pi"""
        s = model.GetState()
        valid_moves = model.GetValidMoves()
        if valid_moves == []:
            # We don't have to negate this value since `playerJustMoved` is already the previous player (no move was played)
            return model.GetResult(model.playerJustMoved)

        if s not in self.visited:
            self.visited.append(s)
            with torch.no_grad():
                # expand state for prediction with the nnet
                expanded_state = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                # Set this state's policy to a predicted prior
                self.P[s], v = self.nnet(expanded_state)
            # Mask out invalid moves
            valid_mask = torch.tensor([move in valid_moves for move in self.all_moves])
            self.P[s] *= valid_mask
            # If the prior is empty, set it to uniform distribution for valid moves
            if torch.sum(self.P[s]) == 0:
                self.P[s] = torch.ones_like(self.P[s]) * valid_mask
            # Re-normalize after removing invalids
            self.P[s] /= torch.sum(self.P[s])
            return -v # Negate to give value according to previous player
    
        node_visits = sum(self.N[s].values())

        def get_move_score(a):
            if a not in valid_moves:
                return float('-inf')
            a_i = self.all_moves.index(a)
            U_sa = self.c_puct*self.P[s][a_i].item()*np.sqrt(node_visits)/(1+self.N[s][a])
            return (self.Q[s][a] + U_sa)
        
        # If unexpanded, directly use the prior policy
        if node_visits == 0:
            a = self.all_moves[torch.argmax(self.P[s])]
        else:
            a = max(self.all_moves, key = lambda a: get_move_score(a))
                 
        model.DoMove(a)
        # recursively traverse the tree
        v = self._uct_search_recursive(model)

        self.W[s][a] += v
        self.N[s][a] += 1
        self.Q[s][a] = self.W[s][a] / self.N[s][a]
        return -v # Negate to give value according to previous player
    
    def pi(self, s, temp = 1):
        """ Return the policy according to the search for a given state using number of edge visits.
            Arguments:
                s - State we are computing the policy for.
                temp - Exploration parameter.
        """
        if temp == 0:
            # Temp == 0 is a special case where we return equal likelihhod for all the most-visited edges
            visits = torch.tensor([self.N[s][a] for a in self.all_moves])
            max_visits = torch.where(visits == torch.max(visits), torch.ones_like(visits), torch.zeros_like(visits))
            return max_visits/torch.sum(max_visits)
        else:
            exp_visits = torch.tensor([self.N[s][a]**(1/temp) for a in self.all_moves])
            return exp_visits/torch.sum(exp_visits)