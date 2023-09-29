import gymnasium as gym

from stable_baselines3 import DQN

from DQN.Melax_Tetris_Gym import BlocksEnv
import os

############
### Main ###
############

#################################
#            Controls           #
version             = "DQN_SB3"#
training            = True     #
new                 = True     #
steps               = 140000     #
BlocksEnv.rendering = False      #
BlocksEnv.obsFlatten= True       #
#################################

models_dir = f"models/DQN_SB3/{version}/"
logdir = "logs/Results"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = BlocksEnv()
env.reset()

env.file_path = "score_" + version + ".txt"

if new:
    steps = 0
    model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=logdir, exploration_initial_eps=1.0, exploration_final_eps=0.1, exploration_fraction=0.1)
else:
    model_path = f"{models_dir}/{steps}.zip"
    model = DQN.load(model_path, env=env)

TIMESTEPS = 1_000_000
obs = model.env.reset()

i = 0
for i in range(1):
    i += 1
    if training:
        model.learn(total_timesteps=TIMESTEPS,  reset_num_timesteps=False, tb_log_name=version)
        model.save(f"{models_dir}/{TIMESTEPS*i + steps}")
    else:
        action, _ = model.predict(obs)
        obs, _, _ ,_ = model.env.step(action)
            