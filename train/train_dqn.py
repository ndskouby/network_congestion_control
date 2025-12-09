from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import envs
import gymnasium as gym

env = gym.make("CongestionControl-v0")

# Monitor wrapper for logging stats
env = Monitor(env, filename="logs/dqn/dqn_monitor.csv")

model = DQN("MlpPolicy", env, verbose=1,
            learning_rate=1e-4,
            buffer_size=100000,
            batch_size=64, 
            gamma=0.9)

# Train model
model.learn(total_timesteps=500_000)

# Save trained model
model.save("models/dqn_cc")
