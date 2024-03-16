import torch
from torch import nn
from utils import make_network
import numpy as np

class QNetwork(nn.Module):
    def __init__(self,
                 gamma,
                 state_dim,
                 action_dim,
                 hidden_sizes=[10, 10]):
        super().__init__()
        self.gamma = gamma
        
        # neural net architecture
        self.network = make_network(state_dim, action_dim, hidden_sizes)
    
    def forward(self, states):
        '''Returns the Q values for each action at each state.'''
        qs = self.network(states)
        return qs

    def get_max_q(self, states):
        # TODO: Get the maximum Q values of all states s.
        pass
    
    def get_action(self, state, eps):
        # TODO: Get the action at a given state according to an epsilon greedy method.
        pass
    
    @torch.no_grad()
    def get_targets(self, rewards, next_states, dones):
        # TODO: Get the next Q function targets, as given by the Bellman optimality equation for Q functions.
        pass