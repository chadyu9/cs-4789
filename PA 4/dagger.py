from common import *
import torch
from torch import optim
from pathlib import Path
import numpy as np
from dataset import *
from torch import distributions as pyd
import torch.nn as nn
import os
from stable_baselines3 import DQN
import gym
import numpy as np
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt

class DAGGER:
    def __init__(self, state_dim, action_dim, args):
        state_to_remove = args.state_to_remove
        self.policy = DiscretePolicy(state_dim-1 if state_to_remove != None else state_dim, action_dim)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        
        theta_path = args.expert_save_path
        
        self.expert_theta = np.load(theta_path) # load expert theta
        
        self.loss = nn.CrossEntropyLoss()

        states = np.arange(state_dim)
        if state_to_remove != None:
            states = states[states!=state_to_remove]

        self.states = states


    def expert_policy(self, state):
        phis = utils.extract_features(state, 2)
        action_dist = utils.compute_action_distribution(self.expert_theta, phis)
        action = np.random.choice([0,1], p=action_dist.flatten())
        return action

    def rollout(self, env, num_steps):
        states = []
        expert_actions = []
        state = torch.from_numpy(env.reset()).float()
        
        for _ in range(num_steps):
            logits = self.policy(state[self.states])

            # TODO: Get action and expert action from current state
            action = None
            expert_action = None
            
            states.append(state[self.states])
            expert_actions.append(torch.tensor(expert_action))
            
            next_state, _, done, _ = env.step(action)
            if done:
                state = torch.from_numpy(env.reset()).float()
            else:
                state = torch.from_numpy(next_state).float()
        return ExpertData(torch.stack(states, dim=0), torch.stack(expert_actions, dim=0))
    
    def get_logits(self, states):
        return self.policy(states)

    def sample_from_logits(self, logits):
        # TODO Given logits from our neural network, sample an action from the distribution defined by said logits.
        pass

    def learn(self, expert_states, expert_actions):
        # TODO Do gradient descent here.
        pass
    
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

def experiment(args):
    # expert dataset loading
    save_path = os.path.join(args.data_dir, args.env+'_dataset.pt')
    
    expert_dataset = ExpertDataset(ExpertData(torch.tensor([]), torch.tensor([], dtype=int)))
    
    # Create env
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # policy initialization
    learner = DAGGER(state_dim, action_dim, args)
    epoch_losses = []
    
    for _ in tqdm(range(1, args.dagger_epochs+1)):
        # TODO Rollout the data
        new_data = None
        
        expert_dataset.add_data(new_data)

        dataloader = get_dataloader(expert_dataset, args)

        # Supervised learning step
        supervision_loss = []
        for _ in tqdm(range(1, args.dagger_supervision_steps+1)):
            loss = 0.0
            # TODO Gradient descent
            for batch in dataloader:
                pass
            
            supervision_loss.append(loss)
        epoch_losses.append(np.mean(supervision_loss))

    # plotting
    # epochs = np.arange(1, args.dagger_epochs + 1)
    
    # plot_losses(epochs, epoch_losses, args.env)

    # saving policy
    dagger_path = os.path.join(args.policy_save_dir, 'dagger')
    os.makedirs(dagger_path, exist_ok=True)
    
    policy_save_path = os.path.join(dagger_path, f'{args.env}.pt')

    learner.save(policy_save_path)


if __name__ == '__main__':
    args = utils.get_args()
    experiment(args)
