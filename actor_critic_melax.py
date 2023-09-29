import sys
import os
import torch as T
import gymnasium as gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
from A2C.Melax_Tetris_Gym import BlocksEnv
from torch.utils.tensorboard import SummaryWriter

BlocksEnv.rendering = False
BlocksEnv.obsFlatten = True

# hyperparameters
hidden_size = [512, 256]
learning_rate = 0.0002

# Summary Writer
version = 'A2C_Mel4'
writer = SummaryWriter(f"logs/A2C/{version}")

models_dir = f"models/A2C/{version}/"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Constants
GAMMA = 0.99
num_steps = 1000
max_episodes = 50_000

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.critic_linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.critic_linear3 = nn.Linear(hidden_size[1], 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.actor_linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.actor_linear3 = nn.Linear(hidden_size[1], self.num_actions)
    
    def forward(self, state):
        state = Variable(T.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = F.relu(self.critic_linear2(value))
        value = self.critic_linear3(value)
        
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.relu(self.actor_linear2(policy_dist))
        policy_dist = F.softmax(self.actor_linear3(policy_dist), dim=1)


        return value, policy_dist
    
    def save_checkpoint(self):
        T.save(self.state_dict(), models_dir)

    def load_checkpoint(self):
        self.load_state_dict(T.load(models_dir))
    
def a2c(env, totalSteps):
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    
    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    if totalSteps != 0:
        actor_critic.load_checkpoint()

    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        for steps in range(num_steps):
            totalSteps += 1
            value, policy_dist = actor_critic.forward(state)
            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().numpy() 

            action = np.random.choice(num_outputs, p=np.squeeze(dist))
            log_prob = T.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state

            if ((totalSteps % 1000 == 0)):
                try:
                    print('... saving models ...')
                    actor_critic.save_checkpoint()
                except:
                    print('!!! save unsuccesfull !!!')
            
            if done:
                Qval, _ = actor_critic.forward(new_state)
                Qval = Qval.detach().numpy()[0,0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:                    
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                writer.add_scalar("rollout/ep_rew_mean", np.sum(rewards), totalSteps)
                writer.add_scalar("rollout/ep_len_mean", steps, totalSteps)
                break
        
        # Compute the Advantage Function
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval
  
        #update actor critic
        values = T.FloatTensor(values)
        Qvals = T.FloatTensor(Qvals)
        log_probs = T.stack(log_probs)
        
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        # Added gradient clipping for stability
        nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
        ac_optimizer.step()

    return totalSteps


if __name__ == "__main__":
    env = BlocksEnv()
    a2c(env, 0)
    