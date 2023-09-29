import gymnasium as gym

import random
import math

import numpy as np

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from collections import namedtuple, deque
from Melax_2 import BlocksEnv


#################
### Functions ###
#################

def flatten(observation):
    return torch.from_numpy(np.array(observation).flatten()).float().unsqueeze(0)

def preprocess(observation):
    return flatten(observation)

def select_action(state, eps, info):

    sample = random.random()
    
    if sample > eps:
        with torch.no_grad():
            return policy_net(state).max(1)[1].unsqueeze(0)
    else:
        sple = random.random()
        if sple > 1:
            return torch.tensor([[random.randrange(numActions)]], device=device, dtype=torch.long)
        else:
            return torch.tensor([[info]], device=device, dtype=torch.long)



########################
## Experience Replay ###
########################

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




########################
### Model Optimizers ###
########################

def optimize_model():
    batch_size = 256

    if len(memory) < batch_size:
        return
    
    experience = memory.sample(batch_size)
    batch = Transition(*zip(*experience))

    state_batch = torch.cat(batch.currentState)
    action_batch = torch.cat(batch.action)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    reward_batch = torch.cat(batch.reward)
    non_final_next_states = torch.cat([s for s in batch.nextState
                                                    if s is not None])

    next_state_values = torch.zeros(batch_size, device=device)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.nextState)), device=device, dtype=torch.bool)

    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    next_state_values = torch.reshape(next_state_values, (batch_size, 1))
    TDtargets = (next_state_values * gamma) + reward_batch

    loss = policy_net.criterion(state_action_values, TDtargets)
        
    policy_net.optimizer.zero_grad()

    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    policy_net.optimizer.step()



###########
## Main ###
###########

################################
#           Controls           #
version             = "DQN_RB" #  DQN_LR0.001 -- DQN_Melax
train               = False      #      3000        17000
load                = True     #
log                 = False     #
steps_done          = 470061   # 13781356 -- 19895205
render              = True    #
test                = True     #
save_files          = True
BlocksEnv.obsFlatten= False    #
################################

####################################
#       MODEL HYPERPARAMETERS       #
numActions              = 4         #
inputSize               = 50        #

alpha                   = 0.001     # Annealing LR???  
episodes                = 5000      #   Best Hyperparameters?
batch_size              = 256       #       alpha   0.002
target_update           = 5000      #

gamma                   = 0.90      #       gamma = 0.9

start_epsilon           = 1.0       # 
stop_epsilon            = 0.1       #
decay_rate              = 30000    #
learn_loss              = 0         #
                                    
pretrain_length         = batch_size#
memorySize              = 100000     #
                                    
evalEpisodes            = 500_000     #
total_steps             = 1_000_000  #
#####################################


## Init AV function with random weights \theta ###
## Init Target function with \theta^- = \theta ###

hiddenLayerSize = (512,256)
policy_net = DQN(alpha, inputSize, numActions, hiddenLayerSize)
target_net = DQN(alpha, inputSize, numActions, hiddenLayerSize)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

models_dir = f"models/DQN/{version}/" 
if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        

if load:
    filename = models_dir + f'{steps_done}.pth'
    policy_net = torch.load(filename) 
    target_net.load_state_dict(policy_net.state_dict())
    steps_done = steps_done
    print('Loaded')
else:
    steps_done = 0

#Init Replay Memory
memory = ReplayMemory(memorySize)

env = BlocksEnv()

env.rendering = render
env.test2_s = test
if save_files:
    env.file_path = version 
    env.save_files = True

obs = env.reset()
# obs, reward, done, info = env.step(0)
# for _ in range(50):
#     obs, reward, done, info = env.step(info)

if log:
    writer = SummaryWriter(f"logs/Results/{version}")

# Main RL Loop
finishCounter = 0.0
totalSteps = 0.0
totalReward = 0.0
 
print('Training...')

for e in range(evalEpisodes):
    # Init seq S_1 = {x_1} and \phi_1 = \phi{S_1} 
    currentObs = env.reset()
    currentState = preprocess(currentObs)
    info = 0
    
    # the main RL loop
    done = False
    game_lenght = 0

    verbose = False

    while not done:
        # Select and perform an action with probability \epsilon
        if train:
            eps_threshold = stop_epsilon+(start_epsilon-stop_epsilon)*math.exp(-1. * steps_done / decay_rate)
        else:
            eps_threshold = 0

        action = select_action(currentState, eps_threshold, info)
        a = action.item()

        obs, reward, done, info = env.step(a)

        game_lenght += 1
        steps_done += 1
            
        if (done):
            nextState = None
        else:
            nextState = preprocess(obs)

        rew_tensor = torch.tensor([[reward]], device = device)
        memory.push(currentState, action, nextState, rew_tensor)

        totalReward += reward

        if (done):
            totalSteps = steps_done
            finishCounter += 1

            if train:
                optimize_model()

            if log:
                writer.add_scalar("rollout/ep_rew_mean", totalReward, totalSteps)
                writer.add_scalar("rollout/ep_len_mean", game_lenght, totalSteps)
                writer.add_scalar("rollout/exploration_rate", eps_threshold, totalSteps)

            break
            
        # Move to the next state
        currentState = nextState

        # Update the target network, copying all weights and biases from the policy network
        if (steps_done % target_update == 0):
            verbose = True
            target_net.load_state_dict(policy_net.state_dict())
        
    if verbose:
        print(f"\nEpisodes: {e}\n Game:\n\t Reward: {totalReward}\n\t Lenght: {game_lenght}\n\t Epsilon: {eps_threshold}")
        if log:
            filename = models_dir + f'{steps_done}.pth'
            torch.save(policy_net, filename)

    totalReward = 0
    if steps_done >= total_steps:
        break