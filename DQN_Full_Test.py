from DQN.Full_Tetris_Actions import BlocksEnv
from DQN.Divan_DQN import *
import gymnasium as gym

    
BlocksEnv.test2_s   = True          
BlocksEnv.save_files= True         
BlocksEnv.obsFlatten= True 
BlocksEnv.test_obs = True

#####################################
#            Controls               #
version_            = "DQN_Obs_LR0.001"    # DQN_FT_15
evaluate            = True         #
steps_done_         = "model"       # model_V1
space               = 17
#####################################

env_ = BlocksEnv()
BlocksEnv.rendering = False # evaluate
BlocksEnv.obs_space = space
preprocess_ = True
logg_ = not evaluate

DQN_ = QNetwork(version=version_, numActions=40, state_size=space, logging=logg_, 
                lr=0.001, decay_rate=100_000, stop_epsilon=0.15, hiddenLayerSize=[512,256])

if evaluate:
    # env_ = gym.make("Acrobot-v1", render_mode = "human")
    DQN_.load(steps=steps_done_)
    DQN_.evaluate(env=env_, evalEpisodes=500, test=True, preprocess=preprocess_, continuous=True, time_=0.001)
else:
    # env_ = gym.make("Acrobot-v1")
    # DQN_.load(steps=steps_done_)
    DQN_.train(env=env_, episodes=500_000, preprocess=preprocess_, totalsteps=0, epochs=1)

