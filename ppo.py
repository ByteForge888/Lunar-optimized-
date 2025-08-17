from stable_baselines3 import PPO
import gym
import numpy as np

class AnalyzerEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=np.array([200, 0.5]), high=np.array([600, 1.5]), shape=(2,))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,))
        self.metrics = {"fps": 0, "conf": 0, "cpu": 0}

    def step(self, action):
        box_constant, sleep_time = action
        # Run lunar_analyzer.py, collect metrics
        reward = self.metrics["fps"] + self.metrics["conf"] - self.metrics["cpu"]
        return list(self.metrics.values()), reward, False, {}

    def reset(self):
        self.metrics = {"fps": 0, "conf": 0, "cpu": 0}
        return list(self.metrics.values())

env = AnalyzerEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)