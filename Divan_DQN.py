import random

import numpy as np
from minigrid.wrappers import *

import os

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################
### Experience Replay ###
#########################

from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('currentState', 'action', 'nextState', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
#####################################
### Policies and Value Esitmators ###
#####################################

### Neural network model definition
class DQN(nn.Module):

    def __init__(self, alpha, inputSize, numActions, hiddenLayerSize=(512, 256)):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputSize, hiddenLayerSize[0])
        self.fc2 = nn.Linear(hiddenLayerSize[0], hiddenLayerSize[1])
        self.fc3 = nn.Linear(hiddenLayerSize[1], numActions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()


    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class QNetwork:
    def __init__(self, version = 'DQN_Default', numActions = 4, state_size = 50, lr=0.001, gamma = 0.90, memSize = 50000, logging = False, verbose = True, target_update = 5000, start_epsilon=1, stop_epsilon=0.0001, decay_rate=100000, hiddenLayerSize = (512,256)):
        self.state_size = state_size
        self.numActions = numActions
        self.lr = lr        
        self.start_epsilon = start_epsilon
        self.stop_epsilon = stop_epsilon
        self.decay_rate = decay_rate
        self.memory = ReplayMemory(memSize)
        self.models_dir = f"models/DQN/{version}/" 
        self.writer = SummaryWriter(f"logs/V02/{version}")
        self.hiddenLayerSize = hiddenLayerSize
        self.gamma = gamma
        self.steps_done = 0
        self.target_update = target_update
        self.verbose = verbose
        self.logging = logging
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        self._create_model()


    def _create_model(self):
        self.policy_net = DQN(self.lr, self.state_size, self.numActions, self.hiddenLayerSize)
        self.target_net = DQN(self.lr, self.state_size, self.numActions, self.hiddenLayerSize)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def flatten(self, observation):
        return torch.from_numpy(np.array(observation)).float().unsqueeze(0)

    def preprocess(self, observation):
        return self.flatten(observation)

    def select_action(self, state):
        """Returns the best of multiple states unless it has decided to explore which returns a random one."""
        self.eps_threshold = self.stop_epsilon+(self.start_epsilon-self.stop_epsilon)*math.exp(-1. * self.steps_done / self.decay_rate)
        sample = random.random()
        
        if sample > self.eps_threshold:
            self.model_action(state)
        else:
            return torch.tensor([[random.randrange(self.numActions)]], device=device, dtype=torch.long)
        
    def model_action(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].unsqueeze(0)

    def optimize_model(self):
        batch_size = 256

        if len(self.memory) < batch_size:
            return
        
        experience = self.memory.sample(batch_size)
        batch = Transition(*zip(*experience))

        state_batch = torch.cat(batch.currentState)
        action_batch = torch.cat(batch.action)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.nextState
                                                        if s is not None])

        next_state_values = torch.zeros(batch_size, device=device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.nextState)), device=device, dtype=torch.bool)

        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        next_state_values = torch.reshape(next_state_values, (batch_size, 1))
        TDtargets = (next_state_values * self.gamma) + reward_batch

        loss = self.policy_net.criterion(state_action_values, TDtargets)
            
        self.policy_net.optimizer.zero_grad()

        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_net.optimizer.step()

    def log(self, totalReward, game_lenght):
        if self.verbose:
            self.writer.add_scalar("rollout/ep_rew_mean", totalReward, self.steps_done)
            self.writer.add_scalar("rollout/ep_len_mean", game_lenght, self.steps_done)
            self.writer.add_scalar("rollout/exploration_rate", self.eps_threshold, self.steps_done)

    def train(self, env, episodes=1):
        """Trains the Neural Network for x episodes and returns the amount of steps, rewards and scores.

        An episode is the same as one game of tetris from start to game over

        A step is the same as a frame in the game.
        Every step it decides on the action to use and saves the result as experience.

        After every episode it trains the model with the 20000 most recent experiences.

        :rtype tuple of (steps, rewards, scoores). steps is an integer, rewards and scores are an integer list
        """

        for e in range(episodes):
            currentObs = env.reset()
            currentState = self.preprocess(currentObs)

            done = False
            game_lenght = 0
            totalReward = 0.0
            
            while not done:
                action = self.select_action(currentState) #
                a = action.item()

                obs, reward, done, info = env.step(a)

                game_lenght += 1
                self.steps_done += 1
                totalReward += reward

                if (done):
                    nextState = None
                    self.log(totalReward, game_lenght)
                    break
                else:
                    nextState = self.preprocess(obs)

                rew_tensor = torch.tensor([[reward]], device = device)
                self.memory.push(currentState, action, nextState, rew_tensor)

            
                currentState = nextState

                if (self.steps_done % self.target_update == 0):
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    self.save()
                    flag = True

            if self.verbose & flag:
                print(f"\nEpisodes: {e}\n Game:\n\t Reward: {totalReward}\n\t Lenght: {game_lenght}\n\t Epsilon: {self.eps_threshold}")
        


    def load(self):
        """Load the weights."""
        filename = self.models_dir + f'{self.steps_done}.pth'
        self.policy_net = torch.load(filename) 
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print('Loaded')

    def save(self):
        """Save the weights."""
        filename = self.models_dir + f'{self.steps_done}.pth'
        torch.save(self.policy_net, filename)

    def evaluate(self, env, evalEpisodes = 1):

        for e in range(evalEpisodes):
            currentObs = env.reset()
            currentState = self.preprocess(currentObs)
            done = False

            totalReward = 0.0
            game_lenght = 0

            while not done:
                #action = self.model_action(currentState) 
                #a = action.item()
                a = env.action_space.sample()
                obs, reward, done, info = env.step(a)

                if (done):
                    nextState = None
                else:
                    nextState = self.preprocess(obs)

                totalReward += reward
                game_lenght += 1

                currentState = nextState
                time.sleep(0.01)

            if self.verbose:
                print(f"\nEpisodes: {e + 1}\n Game:\n\t Reward: {totalReward}\n\t Lenght: {game_lenght}")
