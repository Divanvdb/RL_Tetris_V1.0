from DQN.Full_Tetris_Actions import BlocksEnv
from DQN.Divan_DQN_CNN import *
import gymnasium as gym

        
BlocksEnv.test2_s   = False          
BlocksEnv.save_files= False         
BlocksEnv.obsFlatten= False 

#####################################
#            Controls               #
version_            = "DQN_CNN_ASD1"# Change the newfigure to 5
evaluate            = True         #
steps_done_         = 'model'       # 
#####################################

env_ = BlocksEnv()
env_.reset()
logg_ = not evaluate
BlocksEnv.rendering = evaluate 
preprocess_ = True

DQN_ = QNetwork(version=version_, logging=logg_, numActions=40,  lr=0.0007, hiddenLayerSize=(256,), decay_rate=30_000, start_epsilon=1.0, stop_epsilon=0.1)

if evaluate:
    DQN_.load(steps=steps_done_)
    DQN_.evaluate(env=env_, evalEpisodes=500_000, test=True)
else:
    #DQN_.load(steps=steps_done_)
    # env_ = gym.make("Acrobot-v1")
    DQN_.train(env=env_, episodes=500_000, preprocess=preprocess_, totalsteps=0)

