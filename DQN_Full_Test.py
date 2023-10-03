from DQN.Full_Tetris_Actions import BlocksEnv
from DQN.Divan_DQN import *
import gymnasium as gym

    
BlocksEnv.test2_s   = True          
BlocksEnv.save_files= False         
BlocksEnv.obsFlatten= True 
BlocksEnv.test_obs = False

#####################################
#            Controls               #
version_            = "DQN_FT_7"   # 
evaluate            = True          #
steps_done_         = "model"      # 
#####################################

# Add a penalty for invalid actions
# Check the model again, seems to be skipping pieces

env_ = BlocksEnv()
BlocksEnv.rendering = evaluate
preprocess_ = True
logg_ = not evaluate

DQN_ = QNetwork(version=version_, numActions=40, state_size=200, logging=logg_, 
                lr=0.0001, decay_rate=50_000, stop_epsilon=0.1, hiddenLayerSize=[512,256])

if evaluate:
    # env_ = gym.make("Acrobot-v1", render_mode = "human")
    DQN_.load(steps=steps_done_)
    DQN_.evaluate(env=env_, evalEpisodes=500, test=True, preprocess=preprocess_)
else:
    # env_ = gym.make("Acrobot-v1")
    DQN_.train(env=env_, episodes=500_000, preprocess=preprocess_, totalsteps=0, epochs=1)

