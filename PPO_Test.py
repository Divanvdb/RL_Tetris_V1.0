from PPO.Divan_PPO_Single import *
from PPO.Melax_Tetris_Gym import BlocksEnv
import gymnasium as gym

#####################################
#            Controls               #
version_ = "PPO_Mel3"                #
evaluate = True                    #
logging = not evaluate
episodes = 500_000                      #
time_steps = 2024
steps_done_ = 5085855               #
#####################################

BlocksEnv.rendering = evaluate
BlocksEnv.obsFlatten = True

env = BlocksEnv()
# env = gym.make("CartPole-v1")
# env = gym.wrappers.FrameStack(env, 4)
# if evaluate:
#     env = gym.make('CartPole-v1', render_mode = 'human')

agent = Agent(version=version_, input_dims=50, n_actions=4, alpha=0.0001, norm_adv=False, target_kl=None)

total_reward = 0
game_lenght = 0

i = 2100000

agent.load_models()

if evaluate:
    agent.load_models()

for e in range(episodes):
    obs = env.reset()
    for _ in range(time_steps):
        i += 1
        action, probs, values = agent.choose_action(obs)

        obs_, rew, done, info = env.step(action)

        total_reward += rew
        game_lenght += 1

        agent.memory.store_memory(obs, action, probs, values, rew, done)

        if done:
            intObs = env.reset()
            obs = intObs[0]
            print(f"Game:\n\tReward: {total_reward}\n\tLenght: {game_lenght}")
            if logging:
                agent.episodic_log(
                    total_reward=total_reward,
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
                print('Save Unsuccesfull')

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

raise










from PPO.Melax_Tetris_Gym import BlocksEnv
from PPO.Divan_PPO import *
import gymnasium as gym

BlocksEnv.rendering = False
BlocksEnv.test2_s = False
BlocksEnv.save_files = False
BlocksEnv.obsFlatten = True

#####################################
#            Controls               #
version_ = "PPO_CP"  # DQN_Eps0.1
evaluate = False  #
steps_done_ = 5085855  # 12165000
#####################################

env_ = BlocksEnv()
env_ = gym.make("CartPole-v1")
preprocess_ = True

model = PPO(version=version_, env=env_, numActions=2, state_size=4, logging=True)

if evaluate:
    # env_ = gym.make("Acrobot-v1", render_mode = "human")
    # model.load(steps=steps_done_)
    model.evaluate(evalEpisodes=5, test=False, preprocess=preprocess_)
else:
    model.train(episodes=10, num_steps=150)
