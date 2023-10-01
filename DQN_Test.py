from DQN.Melax_Tetris_Gym import BlocksEnv
from DQN.Divan_DQN import *
import gymnasium as gym

BlocksEnv.rendering = True         
BlocksEnv.test2_s   = True          
BlocksEnv.save_files= False         
BlocksEnv.obsFlatten= True 

#####################################
#            Controls               #
version_            = "DQN_Test1"   # DQN_Eps0.1
evaluate            = True          #
steps_done_         = 1000000      # 12165000
#####################################

env_ = BlocksEnv()
preprocess_ = True

DQN_ = QNetwork(version=version_, logging=True, decay_rate=30000, start_epsilon=1.0, stop_epsilon=0.1)

if evaluate:
    # env_ = gym.make("Acrobot-v1", render_mode = "human")
    DQN_.load(steps=steps_done_)
    DQN_.evaluate(env=env_, evalEpisodes=5, test=True, preprocess=preprocess_)
else:
    # env_ = gym.make("Acrobot-v1")
    DQN_.train(env=env_, episodes=500_000, preprocess=preprocess_, totalsteps=0)

