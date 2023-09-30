from DQN.Tetris_Full_Gym import BlocksEnv
from DQN.Divan_DQN_CNN import *
import gymnasium as gym

BlocksEnv.rendering = False         
BlocksEnv.test2_s   = True          
BlocksEnv.save_files= False         
BlocksEnv.obsFlatten= False 

#####################################
#            Controls               #
version_            = "DQN_CNN"   # 
evaluate            = False          #
steps_done_         = "model"      # 
#####################################

env_ = BlocksEnv()
preprocess_ = True

DQN_ = QNetwork(version=version_, state_size=200, logging=True, decay_rate=50_000)

if evaluate:
    # env_ = gym.make("Acrobot-v1", render_mode = "human")
    DQN_.load(steps=steps_done_)
    DQN_.evaluate(env=env_, evalEpisodes=5, test=True, preprocess=preprocess_)
else:
    # env_ = gym.make("Acrobot-v1")
    DQN_.train(env=env_, episodes=500_000, preprocess=preprocess_, totalsteps=0)

