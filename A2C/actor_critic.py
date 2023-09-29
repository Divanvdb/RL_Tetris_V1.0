import sys
import torch  
import gymnasium as gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
from Melax_Tetris_Gym import BlocksEnv
from torch.utils.tensorboard import SummaryWriter

# hyperparameters
hidden_size = [512, 256]
learning_rate = 0.0001

# Summary Writer
version = 'A2C_CP7'
writer = SummaryWriter(f"logs/A2C/{version}")

# Constants
GAMMA = 0.99
num_steps = 300
max_episodes = 1000

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.critic_linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.critic_linear3 = nn.Linear(hidden_size[1], 1)


        self.actor_linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.actor_linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.actor_linear3 = nn.Linear(hidden_size[1], self.num_actions)
    
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = F.relu(self.critic_linear2(value))
        value = self.critic_linear3(value)
        
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.relu(self.actor_linear2(policy_dist))
        policy_dist = F.softmax(self.actor_linear3(policy_dist), dim=1)


        return value, policy_dist
    
def a2c(env):
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    
    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0
    totalSteps = 0

    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        state = state[0] # For Cartpole Problem Only
        for steps in range(num_steps):
            totalSteps += 1
            value, policy_dist = actor_critic.forward(state)
            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().numpy() 

            action = np.random.choice(num_outputs, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, _, _ = env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state
            
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
                state = env.reset()
        
        # Compute the Advantage Function
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval
  
        #update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)
        
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        # Added gradient clipping for stability
        nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
        ac_optimizer.step()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    a2c(env)