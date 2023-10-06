from DQN.Full_Tetris_Actions import BlocksEnv
from DQN.Divan_DQN import *
import gymnasium as gym

    
BlocksEnv.test2_s   = True          
BlocksEnv.save_files= False         
BlocksEnv.obsFlatten= True 
BlocksEnv.test_obs = False

#####################################
#            Controls               #
version_            = "DQN_FT_17"   # DQN_FT_15
evaluate            = False          #
steps_done_         = "model"      # model_V1
#####################################

env_ = BlocksEnv()
BlocksEnv.rendering = evaluate
preprocess_ = True
logg_ = not evaluate

DQN_ = QNetwork(version=version_, numActions=40, state_size=200, logging=logg_, 
                lr=0.0007, decay_rate=50_000, stop_epsilon=0.15, hiddenLayerSize=[2048,1024])

if evaluate:
    # env_ = gym.make("Acrobot-v1", render_mode = "human")
    DQN_.load(steps=steps_done_)
    DQN_.evaluate(env=env_, evalEpisodes=500, test=True, preprocess=preprocess_, continuous=True, time_=0.001)
else:
    # env_ = gym.make("Acrobot-v1")
    # DQN_.load(steps=steps_done_)
    DQN_.train(env=env_, episodes=500_000, preprocess=preprocess_, totalsteps=0, epochs=1)

