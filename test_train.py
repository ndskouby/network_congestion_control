import gymnasium as gym
import envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Create env
env = gym.make("CongestionControl-v0")
env = Monitor(env)

# Evaluation environment
eval_env = gym.make("CongestionControl-v0")
eval_env = Monitor(eval_env)

# Set up evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=5000,  # Evaluate every 5000 steps
    deterministic=True,
    render=False
)

# Create agent - PPO
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./tensorboard/"
)

print("Start training...")

model.learn(
    total_timesteps=1_000_000,
    callback=eval_callback,
    progress_bar=True
)


model.save("models/model-1")

print("Training complete")

# Test
print("\nTesting trained model...")
obs, _ = env.reset()
total_reward = 0
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if done or truncated:
        break

print(f"Test episode reward: {total_reward:.2f}")