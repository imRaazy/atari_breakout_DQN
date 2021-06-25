import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

class TrainingDataManager:
    def __init__(self, env, n_aav_states):
        self.env = env
        self.aav_states = n_aav_states
        self.n_episodes = 0
        self.arpe = 0
        self.arpe_in_epoch = np.array([])
        self.aav_in_epoch = np.array([])
        self.sampled_states = self._initialize_sampled_state()
    
    def _initialize_sampled_state(self):
        steps = 5000
        states = deque(maxlen = steps)
        state = self.env.reset()
        for i in range(steps):
          action = self.env.action_space.sample()
          state, _, done, _ = self.env.step(action)
          states.append(state)
          if done:
            self.env.reset()
        indices = np.random.randint(len(states), size=self.aav_states)
        return [states[index] for index in indices]
        
    def update_aav(self, model):
        action_values = np.array([])
        for s in self.sampled_states:
            state_tensor = tf.expand_dims(tf.convert_to_tensor(s), 0)
            action_value = np.max(model(state_tensor)[0])
            action_values = np.append(action_values, action_value)
        self.aav_in_epoch = np.append(self.aav_in_epoch, np.mean(action_values))
    
    def update_arpe(self):
        self.arpe_in_epoch = np.append(self.arpe_in_epoch, self.arpe)
    
    def end_episode_update(self, episode_reward):
        self.n_episodes += 1
        self.arpe = self.arpe + (episode_reward - self.arpe) / self.n_episodes
        
    def get_aav(self):
        return self.aav_in_epoch
        
    def get_arpe(self):
        return self.arpe_in_epoch
    
    def get_episodes(self):
        return self.n_episodes
        
    def print_results(self):
        f, (ax1, ax2) = plt.subplots(2, sharex=True)
        f.tight_layout()
        ax1.plot(self.arpe_in_epoch)
        ax1.set_title('Average Reward Per Episode')
        ax2.plot(self.aav_in_epoch)
        ax2.set_title('Average Action Value (Q)')
        ax2.set_xlabel('Epochs')
        plt.show()

def save_models_and_data(name, folder, model, model_target, arpe, aav):
  path = "./models/" + folder + "/"
  model.save(path + "model_" + name, save_format="tf")
  model_target.save(path + "model_target_" + name, save_format="tf")
  arpe_df = pd.DataFrame(arpe, columns=["arpe"])
  aav_df = pd.DataFrame(aav, columns=["aav"])
  arpe_df.to_csv(path + "model_" + name + "/arpe.csv", index=False)
  aav_df.to_csv(path + "model_" + name + "/aav.csv", index=False)