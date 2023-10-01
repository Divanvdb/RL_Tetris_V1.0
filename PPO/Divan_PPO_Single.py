import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, vers, n_actions, input_dims, alpha,
            hidden_size = [512,256], chkpt_dir=f'models/PPO/'):
        super(ActorNetwork, self).__init__()


        self.checkpoint_file = os.path.join(chkpt_dir, f'{vers}/actor_torch_ppo')
        self.actor = nn.Sequential(
                nn.Linear(input_dims, hidden_size[0]),
                nn.ReLU(),
                nn.Linear(hidden_size[0], hidden_size[1]),
                nn.ReLU(),
                nn.Linear(hidden_size[1], n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, vers, input_dims, alpha, hidden_size = [512,256],
            chkpt_dir='models/PPO'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, f'{vers}/critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(input_dims, hidden_size[0]),
                nn.ReLU(),
                nn.Linear(hidden_size[0], hidden_size[1]),
                nn.ReLU(),
                nn.Linear(hidden_size[1], 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, version = 'PPO_Default', n_actions = 4, input_dims = 50, gamma=0.99, alpha=0.0001, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10, max_grad_norm = 0.5, hidden_size = [512,256]):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.max_grad_norm=max_grad_norm
        self.hidden_size = hidden_size
        self.writer = SummaryWriter(f"logs/PPO/{version}")
        models_dir = f"models/PPO/{version}/"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        self.actor = ActorNetwork(vers=version, n_actions=n_actions, input_dims=input_dims, alpha=alpha, hidden_size=self.hidden_size)
        self.critic = CriticNetwork(vers= version, input_dims=input_dims, alpha=alpha, hidden_size=self.hidden_size)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def episodic_log(self, total_reward, episode_lenght, total_steps):
        self.writer.add_scalar("rollout/ep_rew_mean", total_reward, total_steps)
        self.writer.add_scalar("rollout/ep_len_mean", episode_lenght, total_steps)

    def learn(self, steps_done, logg_):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()
            
            values = vals_arr
            returns = np.zeros(len(reward_arr), dtype=np.float32)

            for t in reversed(range(len(reward_arr)-1)):
                nextnonterminal = 1.0 - dones_arr[t + 1]
                next_return = returns[t + 1]
                returns[t] = reward_arr[t] + self.gamma * nextnonterminal * next_return
            advantage = returns - values

            
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                entropy = dist.entropy()
                
                prob_ratio = new_probs.exp() / old_probs.exp()

                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = 0.5 * (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                entropy_loss = entropy.mean()

                total_loss = actor_loss + 0.5*critic_loss + 0.1* entropy_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        if logg_:
            self.writer.add_scalar("train/value_loss", critic_loss, steps_done)
            self.writer.add_scalar("train/policy_gradient_loss", actor_loss, steps_done)
            self.writer.add_scalar("train/loss", total_loss, steps_done)
            self.writer.add_scalar("train/entropy_loss", entropy_loss, steps_done)

        self.memory.clear_memory()               