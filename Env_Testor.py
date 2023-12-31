from DQN.Full_Tetris_Actions import BlocksEnv
from DQN.Divan_DQN import *
import gymnasium as gym

BlocksEnv.rendering = True         
BlocksEnv.test2_s   = True          
BlocksEnv.save_files= False         
BlocksEnv.obsFlatten= True 

#####################################
#            Controls               #
version_            = "DQN_Melax"  # DQN_Eps0.1
evaluate            = True          #
steps_done_         = 805000      # 12165000
#####################################

env_ = BlocksEnv()
preprocess_ = True

DQN_ = QNetwork(version=version_, numActions=40,state_size=100, logging=True, decay_rate=30_000, start_epsilon=1.0, stop_epsilon=0.1)

if evaluate:
    # env_ = gym.make("Acrobot-v1", render_mode = "human")
    # DQN_.load(steps=steps_done_)
    DQN_.evaluate(env=env_, evalEpisodes=500_000, test=False, preprocess=preprocess_)
else:
    #DQN_.load(steps=steps_done_)
    # env_ = gym.make("Acrobot-v1")
    DQN_.train(env=env_, episodes=500_000, preprocess=preprocess_, totalsteps=0, epochs=1)

