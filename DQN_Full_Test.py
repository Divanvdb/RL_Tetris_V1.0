'''
DQN main() for full Tetris

Imports:
    BlocksEnv : Custom OpenAI Gym Tetris environment
    Divan_DQN : Custom DQN model based on the work of Prof. Herman Engelbrecht

Controls:
    version_    : Run version name
    evaluate    : Toggles the evaluation function of the DQN model for 
                    visualization of the agent gameplay
    space       : Observation space size, used for switching between 
                    original and simplified observation space
'''

from DQN.Full_Tetris_Actions import BlocksEnv
from DQN.Divan_DQN import *
import gymnasium as gym

    
BlocksEnv.test2_s   = True          
BlocksEnv.save_files= True         
BlocksEnv.obsFlatten= True 
BlocksEnv.test_obs = True

#########################################   
#            Controls                   
version_            = "DQN_Obs_LR0.001" 
evaluate            = True              
space               = 17                
#########################################

env_ = BlocksEnv()
BlocksEnv.rendering = evaluate
BlocksEnv.obs_space = space
preprocess_ = True
logg_ = not evaluate

DQN_ = QNetwork(version=version_, numActions=40, state_size=space, logging=logg_, 
                lr=0.001, decay_rate=100_000, stop_epsilon=0.15, hiddenLayerSize=[512,256])

if evaluate:
    DQN_.load(steps="model")
    DQN_.evaluate(env=env_, evalEpisodes=500, test=True, preprocess=preprocess_, continuous=True, time_=1)
else:
    DQN_.train(env=env_, episodes=500_000, preprocess=preprocess_, totalsteps=0, epochs=1)

