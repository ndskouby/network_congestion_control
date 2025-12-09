import gymnasium as gym
import envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import os
from datetime import datetime

# UNIQUE name with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"easy-{timestamp}"

print(f"Training model: {model_name}")

os.makedirs(f"models/{model_name}", exist_ok=True)
os.makedirs(f"logs/{model_name}", exist_ok=True)

train_env = gym.make("CongestionControl-Easy-v0")
train_env = Monitor(train_env)

eval_env = gym.make("CongestionControl-Easy-v0")
eval_env = Monitor(eval_env)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"./models/{model_name}/",
    log_path=f"./logs/{model_name}/",
    eval_freq=5000,
    deterministic=True,
    verbose=1
)

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    policy_kwargs=dict(net_arch=[256, 256, 128]),
    tensorboard_log=f"./tensorboard/{model_name}/"
)

print("Starting training...")
model.learn(total_timesteps=1_000_000, callback=eval_callback, progress_bar=True)
model.save(f"models/{model_name}/final")

# ========== DEBUGGING SECTION ==========


# Get model's network parameters
policy_params = model.policy.state_dict()
print(f"Number of parameters: {sum(p.numel() for p in policy_params.values())}")

# Try a single prediction on a reasonable state
fake_obs = np.array([6.5, 0.5, 0.05, 0.01, 6.0], dtype=np.float32)
fake_action, _ = model.predict(fake_obs, deterministic=True)
print(f"Test prediction on reasonable state:")
print(f"  Input obs: {fake_obs}")
print(f"  Predicted action: {fake_action}")

print("\n" + "="*70)
print("="*70)

mean_old, std_old = evaluate_policy(
    model, 
    eval_env,
    n_eval_episodes=10,
    deterministic=True,
    return_episode_rewards=False
)

# Test with FRESH eval_env
print("\n2. Testing with FRESH eval_env:")
fresh_eval_env = gym.make("CongestionControl-Easy-v0")
fresh_eval_env = Monitor(fresh_eval_env)

mean_fresh, std_fresh = evaluate_policy(
    model, 
    fresh_eval_env,
    n_eval_episodes=10,
    deterministic=True,
    return_episode_rewards=False
)
print(f"   FRESH eval_env: {mean_fresh:.2f} ± {std_fresh:.2f}")

print(f"\n   Difference: {abs(mean_old - mean_fresh):.0f}")

# ========== MANUAL TEST WITH FRESH ENVIRONMENT ==========
print("\n" + "="*70)
print("MANUAL TEST WITH FRESH ENVIRONMENT (episode by episode)")
print("="*70)

test_env = gym.make("CongestionControl-Easy-v0")
test_env = Monitor(test_env)

rewards = []
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
    print(f"Episode {i+1}: {total:.2f}")

avg_manual = np.mean(rewards)
std_manual = np.std(rewards)

# ========== FINAL COMPARISON ==========
print("\n" + "="*70)
print("FINAL COMPARISON SUMMARY")
print("="*70)
print(f"TensorBoard (training):        ~1,200-1,300")
print(f"OLD eval_env:                  {mean_old:.0f} ± {std_old:.0f}")
print(f"FRESH eval_env (evaluate_policy): {mean_fresh:.0f} ± {std_fresh:.0f}")
print(f"FRESH test_env (manual):       {avg_manual:.0f} ± {std_manual:.0f}")
print("="*70)

print(f"\n{'='*70}")
print(f"Model saved to: models/{model_name}/")
print(f"{'='*70}")
