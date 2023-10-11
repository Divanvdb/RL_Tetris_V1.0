import gymnasium as gym
from PPO.Melax_Actions import BlocksEnv

from stable_baselines3 import PPO

#####################################
#            Controls               #
version_ = "PPO_M2"                #
evaluate = False                    #
load = False
logging = not evaluate
episodes = 500_000                      #
TIMESTEPS = 5_000_000
steps_done_ = 5085855               #
#####################################

BlocksEnv.rendering = evaluate
BlocksEnv.obsFlatten = False

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
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="logs/V03")
    model.learn(total_timesteps=TIMESTEPS)
    model.save("ppo-melax")