# Atari Breakout with Deep Reinforcement Learning
The purpose of this project is to implement an autonomous agent capable of playing Atari Breakout through Deep Reinforcement Learning. The algorithms chosen to build the agent are Deep Q-Learning and its improvements, in particular Double DQN, Duelling DQN and the classic DQN but with prioritized experience replay buffer.

## Installation ##
To install this project you need to clone the repository and install the requiremens written in requirements.txt

## Usage ##
Each DQN version is implemented in a different notebook:
- [BreakoutDQN.ipynb](BreakoutDQN.ipynb) -> Vanilla DQN
- [BreakoutDoubleDQN.ipynb](BreakoutDoubleDQN.ipynb) -> Double DQN
- [BreakoutDuellingDQN.ipynb](BreakoutDuellingDQN.ipynb) -> Duelling DQN
- [BreakoutPRBDQN.ipynb](BreakoutPRBDQN.ipynb) -> Vanilla DQN with prioritized replay buffer

In order to try one of the algorithms is enough to open the desired one, optionally define a custom name and directory where to save the model, and run the notebook.
Example:
```
save_models_and_data("name", "folder_name", DQN_model, DQN_model_target, tdm.get_arpe(), tdm.get_aav())
```
will save the trained models as following: models/"folder_name"/model_"name" and models/"folder_name"/model_target_"name".

## Evaluation ##
In [Evalutaion.ipynb](Evalutaion.ipynb) is possible to see a comparison between the different approaches. The first plot shows the collected data during training in terms of Average Reward per Episode and Average Action Value for each model. The second one shows the Average Reward per Episode obtained by all the different models over 30 games.
