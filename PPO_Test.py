from PPO.Divan_PPO_Single import *
from PPO.Melax_Actions import BlocksEnv
import gymnasium as gym

#####################################
#            Controls               #
version_ = "PPO_OBS"                #
evaluate = False                    #
logging = not evaluate
episodes = 500_000                      #
time_steps = 300
steps_done_ = 5085855               #
#####################################

BlocksEnv.rendering = evaluate
BlocksEnv.obsFlatten = True
BlocksEnv.save_files = False

env = BlocksEnv()

# env = gym.make("CartPole-v1")
# env = gym.wrappers.FrameStack(env, 4)
# if evaluate:
#     env = gym.make('CartPole-v1', render_mode = 'human')

agent = Agent(version=version_, input_dims=11, n_actions=4, alpha=0.0003, hidden_size=[512,256], logg_= logging)

total_reward = 0
game_lenght = 0

i = 0

if evaluate:
    agent.load_models()

for e in range(episodes):
    obs = env.reset()
    for _ in range(time_steps):
        i += 1
        action, probs, values = agent.choose_action(obs)

        obs_, rew, done, info= env.step(action)

        total_reward += rew
        game_lenght += 1

        agent.memory.store_memory(obs, action, probs, values, rew, done)

        if done:
            intObs = env.reset()
            obs = intObs[0]
            print(f"Game:\n\tReward: {total_reward}\n\tLenght: {game_lenght}")
            if logging:
                agent.episodic_log(
                    total_reward=total_reward/10,
                    episode_lenght=game_lenght,
                    total_steps=i,
                )
            total_reward = 0
            game_lenght = 0

        obs = obs_

        if ((i % 1000 == 0) & (not evaluate)):
            try:
                agent.save_models()
            except:
                print('Save Unsuccesful')

    
    if not evaluate:
        print(f"Update {e}")
        agent.learn(steps_done=i, logg_=logging)

    agent.memory.clear_memory()


raise

import gymnasium as gym
from PPO.Melax_Tetris_Gym import BlocksEnv

from stable_baselines3 import PPO

#####################################
#            Controls               #
version_ = "PPO_M1"                #
evaluate = False                    #
load = True
logging = not evaluate
episodes = 500_000                      #
TIMESTEPS = 5_000_000
steps_done_ = 5085855               #
#####################################

BlocksEnv.rendering = evaluate
BlocksEnv.obsFlatten = True

# Parallel environments
vec_env = BlocksEnv()
if evaluate:
    model = PPO.load("ppo-melax")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)

        if done:
            obs = vec_env.reset()

else:
    if load:
        model = PPO.load("ppo-melax")
    else:
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="logs/PPO/MLP_Mel")
    model.learn(total_timesteps=TIMESTEPS)
    model.save("ppo-melax")