import random

import numpy as np
from minigrid.wrappers import *

import os

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
class CNN_DQN(nn.Module):

    def __init__(self, height, width, numActions, hiddenLayerSize=(512,)): 
        super(CNN_DQN, self).__init__()     
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=1)
        self.bn2 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 4, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(width))
        convh = conv2d_size_out(conv2d_size_out(height))
        linear_input_size = convw * convh * 32
        
        self.head = nn.Linear(linear_input_size, hiddenLayerSize[0])
        self.fc1 = nn.Linear(hiddenLayerSize[0], numActions)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = F.relu(self.head(x.view(x.size(0), -1)))
        x = self.fc1(x)
        return x


class QNetwork:
    def __init__(self, version = 'DQN_Default', numActions = 4, state_size = 50, lr=0.001, 
                 gamma = 0.90, memSize = 50000, logging = False, verbose = True, 
                 target_update = 5000, start_epsilon=1, stop_epsilon=0.1, decay_rate=300000, 
                 hiddenLayerSize = (512,256)):
        super(QNetwork,self).__init__()
        self.state_size = state_size
        self.numActions = numActions
        self.lr = lr        
        self.start_epsilon = start_epsilon
        self.stop_epsilon = stop_epsilon
        self.decay_rate = decay_rate
        self.memory = ReplayMemory(memSize)
        self.models_dir = f"models/DQN/{version}/" 
        self.writer = SummaryWriter(f"logs/Results/{version}")
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
        # Instantiate the policy network and the target network
        hiddenLayerSize = (128,)
        self.policy_net = CNN_DQN(20, 10, 4, hiddenLayerSize)
        self.target_net = CNN_DQN(20, 10, 4, hiddenLayerSize)

        # Copy the weights of the policy network to the target network
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # We don't want to update the parameters of the target network so we set it to evaluation mode
        self.target_net.eval()

    ''' Preprocessing steps: Flatten and ToTensor

        Converts the observation into a flattened array of tensors for the model to analyse
    
    '''

    def flatten(self, observation):
        return torch.from_numpy(np.array(observation)).float().unsqueeze(0)

    def preprocess(self, observation):
        return self.flatten(observation)
    
    ''' Epsilon Greedy Policy:

        Compares a random number in range [0, 1] to a threshold value
        Returns the best of multiple states unless it has decided to explore which returns a random one

    '''

    def select_action(self, state):
        
        self.eps_threshold = self.stop_epsilon+(self.start_epsilon-self.stop_epsilon)*math.exp(-1. * self.steps_done / self.decay_rate)
        sample = random.random()
        
        if sample > self.eps_threshold:
            return self.model_action(state)
        else:
            return torch.tensor([[random.randrange(self.numActions)]], device=device, dtype=torch.long)
        
    ''' Model Action:

        The action generated by the model based on the input state

    '''
        
    def model_action(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].unsqueeze(0)
        
    ''' Optimization of the Policy Network:

        Mean Squared Error between the state_action_values and the TD Targets

        state_action_value: This is the expected return of the actions that were taken by the Policy Network

        TD Targets: The observations of the rewards and discounted rewards for the next state

    '''

    def optimize_model(self):
        batch_size = 256

        if len(self.memory) < batch_size:
            return
        
        experience = self.memory.sample(batch_size)
        batch = Transition(*zip(*experience))

        state_batch = torch.cat(batch.currentState)
        action_batch = torch.cat(batch.action)
        state_batch = torch.reshape(state_batch, [batch_size, 1, 20, 10])
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.nextState
                                                        if s is not None])

        next_state_values = torch.zeros(batch_size, device=device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.nextState)), device=device, dtype=torch.bool)

        non_final_next_states = torch.reshape(non_final_next_states, [non_final_next_states.shape[0], 1, 20, 10])
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

    def train(self, env, episodes=1, preprocess=False, totalsteps = 1_000_000):
        """Trains the Neural Network for x episodes and returns the amount of steps, rewards and scores.

        An episode is the same as one game of tetris from start to game over

        A step is the same as a frame in the game.
        Every step it decides on the action to use and saves the result as experience.

        After every episode it trains the model with the 20000 most recent experiences.

        :rtype tuple of (steps, rewards, scoores). steps is an integer, rewards and scores are an integer list
        """

        if episodes == 0:
            episodes = 500_000  

        for e in range(episodes):
            currentObs = env.reset()
            if preprocess:
                currentState = self.preprocess(currentObs)
            else:
                currentState = torch.tensor([currentObs[0]], device = device)

            done = False
            game_lenght = 0
            totalReward = 0.0
            flag = False
            
            while not done:
                currentState = torch.reshape(currentState, [1, 1, 20, 10])
                action = self.select_action(currentState) 
                a = action.item()

                obs, reward, done, _, _ = env.step(a)

                game_lenght += 1
                self.steps_done += 1
                totalReward += reward

                if (done):
                    nextState = None
                else:
                    if preprocess:
                        nextState = self.preprocess(obs)
                    else:
                        nextState = torch.tensor([obs], device = device)

                rew_tensor = torch.tensor([[reward]], device = device)
                self.memory.push(currentState, action, nextState, rew_tensor)

                if done:
                    self.optimize_model()
                    self.log(totalReward, game_lenght)
                    break
            
                currentState = nextState

                if (self.steps_done % self.target_update == 0):
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    self.save()
                    flag = True

            if self.verbose & flag:
                print(f"\nEpisodes: {e}\n Game:\n\t Reward: {totalReward}\n\t Lenght: {game_lenght}\n\t Epsilon: {self.eps_threshold}")
            
            if totalsteps != 0:
                if (self.steps_done >= totalsteps):
                    break


    def load(self, steps):
        """Load the weights."""
        filename = self.models_dir + f'{steps}.pth'
        self.policy_net = torch.load(filename) 
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps_done = steps
        print('Loaded')

    def save(self):
        """Save the weights."""
        filename = self.models_dir + f'{self.steps_done}.pth'
        torch.save(self.policy_net, filename)

    def evaluate(self, env, evalEpisodes = 1, test = False, preprocess = False):

        for e in range(evalEpisodes):
            currentObs = env.reset()
            if preprocess:
                currentState = self.preprocess(currentObs)
            else:
                currentState = torch.tensor([currentObs[0]], device = device)
            done = False

            totalReward = 0.0
            game_lenght = 0

            while not done:
                if test:
                    action = self.model_action(currentState) 
                    a = action.item()
                else:
                    a = env.action_space.sample()
                obs, reward, done, info,_ = env.step(a)

                if (done):
                    nextState = None
                else:
                    if preprocess:
                        nextState = self.preprocess(obs)
                    else:
                        nextState = torch.tensor([obs], device = device)

                totalReward += reward
                game_lenght += 1

                currentState = nextState

            if self.verbose:
                print(f"\nEpisodes: {e + 1}\n Game:\n\t Reward: {totalReward}\n\t Lenght: {game_lenght}")
