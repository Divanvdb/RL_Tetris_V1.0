from DQN.Melax_Actions import BlocksEnv
from DQN.Divan_DQN import *
import gymnasium as gym

        
BlocksEnv.test2_s   = True          
BlocksEnv.save_files= True         
BlocksEnv.obsFlatten= True 

#####################################
#            Controls               #
version_            = "DQN_Direct1"       # DQN_Eps0.1
evaluate            = True          #
steps_done_         = 'model'    # 12165000
#####################################

env_ = BlocksEnv()
BlocksEnv.rendering = False# evaluate 
preprocess_ = True
logg_ = not evaluate

DQN_ = QNetwork(version=version_, numActions=20, logging= logg_, decay_rate=30_000, 
                start_epsilon=1.0, stop_epsilon=0.1)

if evaluate:
    # env_ = gym.make("Acrobot-v1", render_mode = "human")
    DQN_.load(steps=steps_done_)
    DQN_.evaluate(env=env_, evalEpisodes=500_000, test=True, preprocess=preprocess_, time_=0.001)
else:
    #DQN_.load(steps=steps_done_)
    # env_ = gym.make("Acrobot-v1")
    DQN_.train(env=env_, episodes=500_000, preprocess=preprocess_, totalsteps=0, epochs=1)

