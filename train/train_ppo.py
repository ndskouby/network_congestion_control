from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import envs
import gymnasium as gym

env = gym.make("CongestionControl-v0")

# Monitor wrapper for logging stats
env = Monitor(env)

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1,
    learning_rate=1e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.9,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    device="cpu"
)

# Train model
model.learn(total_timesteps=500_000)

# Save trained model
model.save("models/ppo_cc")
