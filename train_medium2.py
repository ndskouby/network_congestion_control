# train_medium_v2_FIXED.py
import gymnasium as gym
import envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
from datetime import datetime

# UNIQUE timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"medium-v2-{timestamp}"

print("="*80)
print(f"TRAINING MEDIUM MODEL: {run_name}")
print("="*80)

# Create directories
os.makedirs(f"models/{run_name}", exist_ok=True)
os.makedirs(f"logs/{run_name}", exist_ok=True)

print("\nLoading best easy model...")
#model = PPO.load("models/easy-20251208_140812/final")
model = PPO.load("models/easy-20251208_140812/best_model")

# Set up medium environment
print("\nSetting up MEDIUM environment...")
train_env = gym.make("CongestionControl-Medium-v0")
train_env = Monitor(train_env)

eval_env = gym.make("CongestionControl-Medium-v0")
eval_env = Monitor(eval_env)

model.set_env(train_env)
model.tensorboard_log = f"./tensorboard/{run_name}/"

# Eval callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"./models/{run_name}/",
    log_path=f"./logs/{run_name}/",
    eval_freq=5000,
    deterministic=True,
    verbose=1
)

print("TRAINING...")
print(f"TensorBoard: tensorboard --logdir ./tensorboard/{run_name}/")

# Train with reset_num_timesteps=False to continue from easy model
model.learn(
    total_timesteps=2_500_000,
    callback=eval_callback,
    progress_bar=True,
    reset_num_timesteps=False  # Continue from easy training
)

# Save final
model.save(f"models/{run_name}/final")

print("\nTraining complete")

# ========== PROPER TESTING ==========
print("\n" + "="*80)
print("TESTING RESULTS")
print("="*80)

# Test on ALL environments with FRESH test environments
test_results = {}

for env_name_full in ["CongestionControl-Easy-v0", "CongestionControl-Medium-v0", "CongestionControl-Hard-v0"]:
    env_short = env_name_full.split('-')[1]
    
    # Create fresh test environment
    test_env = gym.make(env_name_full)
    test_env = Monitor(test_env)
    
    rewards = []
    print(f"\nTesting on {env_short} environment:")
    for i in range(10):
        obs, _ = test_env.reset()
        total = 0
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            total += reward
            if done or truncated:
                break
        rewards.append(total)
        if i < 5:
            print(f"  Episode {i+1}: {total:.1f}")
    
    mean = np.mean(rewards)
    std = np.std(rewards)
    test_results[env_short] = (mean, std)
    print(f"  {env_short}: {mean:.0f} ± {std:.0f}")

# ========== COMPARISON ==========

# Hardcoded for testing
baselines = {
    'Easy': (1174, 95),
    'Medium': (1153, 191),
    'Hard': (975, 322)
}

for env_name in ['Easy', 'Medium', 'Hard']:
    old_mean, old_std = baselines[env_name]
    new_mean, new_std = test_results[env_name]
    change = new_mean - old_mean
    change_str = f"{change:+.0f}" if abs(change) > 50 else f"{change:+.0f} (minor)"
    
    print(f"{env_name:<12} {old_mean:.0f}±{old_std:.0f}{'':>8} {new_mean:.0f}±{new_std:.0f}{'':>8} {change_str:<10}")


print("="*80)
print(f"\nModel saved to: models/{run_name}/")
print(f"TensorBoard: tensorboard --logdir ./tensorboard/{run_name}/")
print("="*80)