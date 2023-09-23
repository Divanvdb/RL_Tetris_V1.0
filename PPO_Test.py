from PPO.Divan_PPO_Single import *
from PPO.Melax_Tetris_Gym import BlocksEnv
import gymnasium as gym

#####################################
#            Controls               #
version_ = "PPO_Melax1"                #
evaluate = True                    #
logging = not evaluate
episodes = 100                      #
time_steps = 1000
steps_done_ = 5085855               #
#####################################

BlocksEnv.rendering = evaluate
BlocksEnv.obsFlatten = True

env = BlocksEnv()
# env = gym.make("CartPole-v1")
# env = gym.wrappers.FrameStack(env, 4)
# if evaluate:
#     env = gym.make('CartPole-v1', render_mode = 'human')

agent = Agent(version=version_, input_dims=50, n_actions=4, norm_adv=True)

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
            agent.save_models()

    print(f"Update {e}")
    agent.learn(steps_done=i, logg_=logging)

    agent.memory.clear_memory()


raise

import gymnasium as gym

from stable_baselines3 import PPO

# Parallel environments
vec_env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="logs/PPO/Run1")
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")


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
