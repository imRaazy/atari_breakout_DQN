# Atari Breakout with Deep Reinforcement Learning
The purpose of this project is to implement an autonomous agent capable of playing Atari Breakout through Deep Reinforcement Learning. The algorithms chosen to build the agent are Deep Q-Learning and its improvements, in particular Double DQN, Duelling DQN and the classic DQN but with prioritized experience replay buffer.

## Installation ##
To install this project you need to clone the repository and install the requiremens written in requirements.txt

## Usage ##
Each DQN version is implemented in a different notebook:
-[./BreakoutDQN.ipynb]
-[./BreakoutDoubleDQN.ipynb]
-[./BreakoutDuellingDQN.ipynb]
-[./BreakoutPRBDQN.ipynb]

In order to try one of the algorithms is enough to open the desired one, optionally define a custom name and directory where to save the model, and run the notebook.
Example:
```
save_models_and_data("name", "folder_name", DQN_model, DQN_model_target, tdm.get_arpe(), tdm.get_aav())
```
will save the trained models as following: models/"folder_name"/model_"name" and models/"folder_name"/model_target_"name" 
