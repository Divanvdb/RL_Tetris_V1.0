import math
import random
import os

import gym
import numpy as np

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

use_cuda = T.cuda.is_available()
device = T.device("cuda" if use_cuda else "cpu")


class ActorNetwork(nn.Module):
    def __init__(self, n_actions = 4, input_dims = 50, alpha = 0.001,
            fc1_dims=512, fc2_dims=256, chkpt_dir='models/PPO'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
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
    def __init__(self, input_dims = 50, alpha = 0.001, fc1_dims=512, fc2_dims=256,
            chkpt_dir='models/PPO'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
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


class PPO:
    def __init__(
        self,
        env,
        version="PPO_Default",
        numActions=4,
        state_size=50,
        lr=0.001,
        gamma=0.90,
        hidden_size=[512, 256],
        batch_size=64,
        ppo_epochs=10,
        gae_lambda=0.95,
        update_epochs=20,
        num_minibatches=5,
        clip_coef=0.2,
        gae=True,
        norm_adv=False,
        clip_vloss=True,
        ent_coef=0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,
        logging=False,
        verbose=True
    ):
        super(PPO, self).__init__()
        self.version=version
        self.env = env
        if logging:
            self.writer = SummaryWriter(f"logs/ASDD/{version}")
        self.gamma=gamma
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.ppo_epochs=ppo_epochs
        self.gae_lambda=gae_lambda
        self.update_epochs=update_epochs
        self.num_minibatches=num_minibatches
        self.clip_coef=clip_coef
        self.gae=gae
        self.norm_adv=norm_adv
        self.clip_vloss=clip_vloss
        self.ent_coef=ent_coef
        self.vf_coef=vf_coef
        self.max_grad_norm=max_grad_norm
        self.target_kl=target_kl
        self.logging=logging
        self.verbose=verbose
        self.steps_done = 0
        self.actor = ActorNetwork(n_actions=numActions, input_dims=state_size, alpha=lr).to(device)
        self.critic = CriticNetwork(input_dims=state_size, alpha=lr)
        

    def make_env(self, env_):
        def thunk():
            #env = env_
            env = env_
            # env = gym.wrappers.FrameStack(env, 4)
            return env

        return thunk

    def choose_action(self, state, action = None):
        logits = self.actor(state)
        probs = logits
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(state)
    
    def log(self, totalReward, game_lenght):
        if self.logging:
            self.writer.add_scalar("rollout/ep_rew_mean", totalReward, self.steps_done)
            self.writer.add_scalar("rollout/ep_len_mean", game_lenght, self.steps_done)
            self.writer.add_scalar("rollout/exploration_rate", self.eps_threshold, self.steps_done)
    
    def load(self, steps):
        """Load the weights."""
        self.actor.load_checkpoint()
        self.critic.load_checkpoint() 
        self.steps_done = steps
        print('Loaded')

    def save(self):
        """Save the weights."""
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def memory(self, obs_shape, act_shape, num_steps_ = 100, num_envs_ = 4):
        self.obs = T.zeros((num_steps_, num_envs_) + obs_shape).to(device)
        self.actions = T.zeros((num_steps_, num_envs_) + act_shape).to(device)
        self.logprobs = T.zeros((num_steps_, num_envs_)).to(device)
        self.rewards = T.zeros((num_steps_, num_envs_)).to(device)
        self.dones = T.zeros((num_steps_, num_envs_)).to(device)
        self.values = T.zeros((num_steps_, num_envs_)).to(device)        

    def train(self, episodes = 500_000, num_envs = 4, num_steps = 100):
        envs = gym.vector.SyncVectorEnv([self.make_env(self.env) for i in range(num_envs)])

        self.memory(obs_shape= envs.single_observation_space.shape, act_shape= envs.single_action_space.shape
                    , num_envs_=num_envs, num_steps_=num_steps)

        next_obs = T.Tensor(envs.reset()).to(device)
        next_done = T.zeros(num_envs).to(device)
        minibatch_size = int(self.batch_size // self.num_minibatches)

        for e in range(episodes):
            for step in range(num_steps):
                self.steps_done += 1 * num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done
                with T.no_grad():
                    action, logprob, _, value = self.choose_action(next_obs)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                self.rewards[step] = T.tensor(reward).to(device).view(-1)
                next_obs, next_done = T.Tensor(next_obs).to(device), T.Tensor(done).to(device)

                for item in info:
                    if self.logging:
                        if item['d'] == True:
                            print(f"Game: \n\t Step= {self.steps_done}\n\t Return= {item['r']}\n\t Lenght= {item['l']}")
                            self.writer.add_scalar("rollout/ep_rew_mean", item["r"], self.steps_done)
                            self.writer.add_scalar("rollout/ep_len_mean", item["l"], self.steps_done)
                            break

                if self.steps_done % 2500 * num_envs == 0:
                    self.save()

            with T.no_grad():
                next_value = self.critic(next_obs).reshape(1, -1)
                returns = T.zeros_like(self.rewards).to(device)
                for t in reversed(range(num_steps - 1)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = self.rewards[t] + self.gamma * nextnonterminal * next_return
                advantages = returns - self.values

            self.b_obs = self.obs.reshape((-1,) + envs.single_observation_space.shape)
            self.b_logprobs = self.logprobs.reshape(-1)
            self.b_actions = self.actions.reshape((-1,) + envs.single_action_space.shape)
            self.b_advantages = advantages.reshape(-1)
            self.b_returns = returns.reshape(-1)
            self.b_values = self.values.reshape(-1)   

            # Optimizing the policy and value network
            print(f'Update {e}')
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.choose_action(self.b_obs[mb_inds], self.b_actions.long()[mb_inds])
                    logratio = newlogprob - self.b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with T.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = self.b_advantages[mb_inds] # Nan for some reason
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - self.b_advantages.mean()) / (self.b_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * T.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = T.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - self.b_returns[mb_inds]) ** 2
                        v_clipped = self.b_values[mb_inds] + T.clamp(
                            newvalue - self.b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - self.b_returns[mb_inds]) ** 2
                        v_loss_max = T.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - self.b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.actor.optimizer.zero_grad()
                    self.critic.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.actor.optimizer.step()
                    self.critic.optimizer.step()

                    if self.logging:
                        self.writer.add_scalar("train/value_loss", v_loss, self.steps_done)
                        self.writer.add_scalar("train/policy_gradient_loss", pg_loss, self.steps_done)
                        self.writer.add_scalar("train/loss", loss, self.steps_done)
                        self.writer.add_scalar("train/entropy_loss", pg_loss, self.steps_done)
                        self.writer.add_scalar("train/approx_kl", approx_kl, self.steps_done)
                        #writer.add_scalar("train/clip_fraction", clipfracs.item(), steps_done)

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

        print('Done') 


    def evaluate(self, evalEpisodes = 1, test = False, preprocess = False):
        for e in range(evalEpisodes):
            currentState = self.env.reset()
            if not preprocess:
                currentState = T.tensor([currentState[0]], device = device)
            done = False

            totalReward = 0.0
            game_lenght = 0

            while not done:
                if test:
                    action = self.model_action(currentState) 
                    a = action.item()
                else:
                    a = self.env.action_space.sample()
                obs, reward, done, info,_ = self.env.step(a)

                if (done):
                    nextState = None
                else:
                    nextState = obs
                    if not preprocess:
                        nextState = T.tensor([obs[0]], device = device)

                totalReward += reward
                game_lenght += 1

                currentState = nextState

            if self.verbose:
                print(f"\nEpisodes: {e + 1}\n Game:\n\t Reward: {totalReward}\n\t Lenght: {game_lenght}")

