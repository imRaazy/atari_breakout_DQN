import random
import numpy as np
from collections import deque

class ReplayBuffer():
  def __init__(self, memory_length, batch_size):
    self.memory_length = memory_length
    self.batch_size = batch_size
    self.replay_buffer = deque(maxlen = memory_length)
      
  def sample_experiences(self, indices=None):
    if indices == None:
      indices = np.random.randint(len(self.replay_buffer), size = self.batch_size)
    batch = [self.replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [np.array([experience[field_index] for experience in batch]) for field_index in range(5)]
    return states, actions, rewards, next_states, dones

  def add(self, args_list):
    self.replay_buffer.append(args_list)
      
  def usable(self):
    return len(self.replay_buffer) > self.batch_size

class PrioritizedReplayBuffer(ReplayBuffer):
  PRIO = 5
  PROB = 6
  
  def __init__(self, memory_length, batch_size):
    ReplayBuffer.__init__(self, memory_length, batch_size)
    self.epsilon = 0.01
    self.alpha = 0.6
    self.max_priority = 1.0 + self.epsilon
    self.alpha_priorities_sum = .0
    self.last_sampled_experiences = []

  def add(self, args_list):
    if len(self.replay_buffer) == self.memory_length:
      exp = self.replay_buffer.popleft()
      self.alpha_priorities_sum -= exp[self.PRIO]**self.alpha
      if exp[self.PRIO] == self.max_priority:
        self.max_priority = max(self.replay_buffer, key=(lambda x: x[self.PRIO]))[self.PRIO]
    priority = self.max_priority
    alpha_priority = priority ** self.alpha
    self.alpha_priorities_sum += alpha_priority
    probability = alpha_priority / self.alpha_priorities_sum
    new_args_list = list(args_list) + [priority, probability]
    ReplayBuffer.add(self, new_args_list)
    
  def update_sample(self):
    self.last_sampled_experiences = random.choices(range(len(self.replay_buffer)), 
                                                   weights=[el[self.PRIO] for el in self.replay_buffer], 
                                                   k=self.batch_size)
    
  def sample_experiences(self):
    if not self.last_sampled_experiences:
      self.update_sample()
    return ReplayBuffer.sample_experiences(self, self.last_sampled_experiences)

  def update_prio(self, tds):
    for k, i in zip(self.last_sampled_experiences, range(self.batch_size)):
      td = tds[i]
      old_exp = self.replay_buffer[k]
      new_prio = td + self.epsilon
      old_prio = old_exp[self.PRIO]
      self.alpha_priorities_sum += new_prio**self.alpha - old_prio**self.alpha
      old_exp[self.PRIO] = new_prio
      old_exp[self.PROB] = (new_prio**self.alpha) / self.alpha_priorities_sum
      self.replay_buffer[k] = old_exp
      if td > self.max_priority:
        self.max_priority = new_prio
      elif new_prio < self.max_priority and old_prio == self.max_priority:
        self.max_priority = max(self.replay_buffer, key=(lambda x: x[self.PRIO]))[self.PRIO]  