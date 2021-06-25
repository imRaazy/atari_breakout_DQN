import gym
import cv2
import numpy as np
from collections import deque
from gym import spaces

class EnvWrapper(gym.Wrapper):
  def __init__(self, env, frame_stack=4, frame_skip=4, seed=0):
    gym.Wrapper.__init__(self, env)
    assert "NoFrameskip" in env.spec.id
    self.frame_stack = frame_stack
    self.frame_skip = frame_skip
    self.frames = deque([], maxlen=frame_stack)
    self.max_lives = env.ale.lives()
    self.lives = 0
    self.observation_space = spaces.Box(low=0, high=1, shape=(84, 84, 4), dtype=np.dtype('float32'))
    env.seed(seed)
    
  def _preprocess(self, observation):
    observation = observation.astype("uint8")
    luminance_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    reshaped_frame = cv2.resize(luminance_frame, dsize=(84, 84), interpolation=cv2.INTER_AREA)
    scaled_frame = np.float32(reshaped_frame) / 255.
    return scaled_frame

  def _fix_reward(self, reward):
    return np.sign(reward)
  
  def _get_ob_stack(self):
    return np.stack(self.frames, axis=-1)
  
  def plot_preprocessed_frame(self):
    wrapped_state = self.reset()
    print(f"Wrapped environment:\n {self.observation_space}")
    first_frame = wrapped_state[:,:,0]
    plt.imshow(first_frame, cmap='gray', vmin=0, vmax=1)
    plt.title("Frame Exampe")
    plt.show()
    return
  
  def reset(self):
    if self.lives == 0:
      self.lives = self.max_lives
      self.frames.clear()
      self.env.reset()
      for _ in range(np.random.randint(30)):
        obs, _, _, _ = self.env.step(0)
      obs, _, _, _ = self.env.step(1)
      obs = self._preprocess(obs)
      for _ in range(self.frame_stack):
        self.frames.append(obs)
    else: self.step(1)
    return self._get_ob_stack()
  
  def step(self, action):
    frame_1 = np.zeros(shape=(self.env.observation_space.shape))
    frame_2 = np.zeros(shape=(self.env.observation_space.shape))
    total_reward = 0
    for i in range(self.frame_skip):
      obs, reward, done, info = self.env.step(action)
      if i == self.frame_skip - 2: frame_1 = obs
      if i == self.frame_skip - 1: frame_2 = obs
      total_reward += self._fix_reward(reward)
      if done:
        break
    obs = self._preprocess(np.max(np.stack([frame_1, frame_2]), axis=0))
    self.frames.append(obs)
    if self.env.ale.lives() < self.lives:
      self.lives -= 1
      done = True
    return self._get_ob_stack(), total_reward, done, info
