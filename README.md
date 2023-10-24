# RL_Tetris_V1.0
 Solving Tetris using Reinforcement Learning

## Models and Controls

The RL models and the corresponding environments are saved within the DQN and PPO folders.

### Deep Q-learning (DQN)

The DQN model is saved under the name Divan_DQN and will be the model used for Melax's Tetris 
and full tetris.

The Divan_DQN_CNN model uses a CNN and frame stacking with the agent neural network structure

The different environments are:
    Melax_Actions       :   Melax's Tetris environment that uses a direct action space (chooses between 20 actions)
    Melax_Tetris_Gym    :   Melax's Tetris environment that uses a arcade action space (chooses between four actions)
    Tetris_Full_Gym     :   Full Tetris using arcade action space
    Full_Tetris_Actions :   Full Tetris using direct action space

Within the environments the observation space can be toggled to be a matrix, flattened, or simplified (see individual environments for this controls)

The DQN model is controlled by the following files:
    DQN_Test        :   Melax's Tetris RL model contorls
    DQN_Full_Test   :   Full Tetris RL model controls
    DQN_CNN_Test    :   Full Tetris (using CNN) model controls
    DQN_SB3         :   Stable Baselines3 DQN model controls

### Proximal policy optimization

This model has 2 different options:

    Divan_PPO_Single    :   This model is not vectorized and uses a more complex setup of the model
    Divan_PPO           :   A simplified version of the PPO model

The different environments are:
    Melax_Actions       :   Melax's Tetris environment that uses a direct action space (chooses between 20 actions)
    Melax_Tetris_Gym    :   Melax's Tetris environment that uses a arcade action space (chooses between four actions)

The PPO model is controlled by the following files:
    PPO_Test    :   Melax's Tetris RL model contorls
    PPO_SB3     :   Stable Baselines3 PPO model controls

## Env Testor

For new environments, this script will test the functionality of the environment by running the control loop for
the environment and taking random actions.

This can be used to validate the custom environments.

