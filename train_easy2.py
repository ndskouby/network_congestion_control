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
print("\n" + "="*70)
print("CHECKING IF MODEL STATE IS SANE")
print("="*70)

# Get model's network parameters
policy_params = model.policy.state_dict()
print(f"Number of parameters: {sum(p.numel() for p in policy_params.values())}")

# Try a single prediction on a reasonable state
fake_obs = np.array([6.5, 0.5, 0.05, 0.01, 6.0], dtype=np.float32)
fake_action, _ = model.predict(fake_obs, deterministic=True)
print(f"Test prediction on reasonable state:")
print(f"  Input obs: {fake_obs}")
print(f"  Predicted action: {fake_action}")

# ========== CRITICAL TEST: OLD VS FRESH EVAL ENV ==========
print("\n" + "="*70)
print("COMPARING OLD eval_env vs FRESH eval_env")
print("="*70)

# Test with OLD eval_env (used during training)
print("\n1. Testing with OLD eval_env (used 1500+ times during training):")
mean_old, std_old = evaluate_policy(
    model, 
    eval_env,
    n_eval_episodes=10,
    deterministic=True,
    return_episode_rewards=False
)
print(f"   OLD eval_env: {mean_old:.2f} ± {std_old:.2f}")

# Test with FRESH eval_env
print("\n2. Testing with FRESH eval_env (never used before):")
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

# Diagnosis
if mean_old > 1000 and mean_fresh > 1000 and avg_manual > 1000:
    print("\n✅ ALL PASS! Training succeeded, model works!")
elif mean_old > 1000 and mean_fresh < 0 and avg_manual < 0:
    print("\n❌ OLD eval_env is CORRUPTED/LUCKY")
    print("   Fresh environments fail → Model didn't actually learn")
elif mean_old > 1000 and mean_fresh > 1000 and avg_manual < 0:
    print("\n❌ evaluate_policy works, but manual test fails")
    print("   → Bug in manual test code")
elif mean_old < 0 and mean_fresh < 0 and avg_manual < 0:
    print("\n❌ Training completely failed")
else:
    print("\n⚠️  Mixed results - unclear what's wrong")
    print(f"   Old: {mean_old:.0f}, Fresh eval: {mean_fresh:.0f}, Manual: {avg_manual:.0f}")

print(f"\n{'='*70}")
print(f"Model saved to: models/{model_name}/")
print(f"{'='*70}")

"""
======================================================================
CHECKING IF MODEL STATE IS SANE
======================================================================
Number of parameters: 201222
First layer weights sample: tensor([[ 0.0240, -0.0499, -0.0761],
        [-0.0780,  0.1213, -0.0650],
        [-0.0062,  0.0704, -0.0493]])
Test prediction on reasonable state:
  Input obs: [6.5  0.5  0.05 0.01 6.  ]
  Predicted action: 2
  (0=big decrease, 1=small decrease, 2=hold, 3=small increase, 4=big increase)
======================================================================
REPLICATING EVALCALLBACK'S EVALUATION
======================================================================
EvalCallback-style evaluation: 630.47 ± 485.79
(This should match TensorBoard's ~1,231)
======================================================================
TESTING WITH FRESH ENVIRONMENT
======================================================================
Episode 1: -438.54
Episode 2: -162.74
Episode 3: -249.35
Episode 4: -293.85
Episode 5: -378.77
Episode 6: -480.44
Episode 7: -286.98
Episode 8: -562.99
Episode 9: -311.25
Episode 10: -541.19
======================================================================
FRESH ENV RESULTS: -371 ± 125
======================================================================
======================================================================
COMPARISON SUMMARY
======================================================================
TensorBoard eval/mean_reward:  ~1,231 (from training)
EvalCallback replication:      630 ± 125
Fresh environment test:        -371 ± 125
======================================================================
❌ EvalCallback DOESN'T match TensorBoard (diff: 601)
⚠️  Both failed. Training itself is broken.
======================================================================
Model saved to: models/easy-20251208_024424/
======================================================================






======================================================================
CHECKING IF MODEL STATE IS SANE
======================================================================
Number of parameters: 201222
First layer weights sample: tensor([[ 0.0275, -0.0344, -0.1055],
        [ 0.1036, -0.0522, -0.1158],
        [ 0.0489,  0.0939, -0.1731]])

Test prediction on reasonable state:
  Input obs: [6.5  0.5  0.05 0.01 6.  ]
  Predicted action: 0
  (0=big decrease, 1=small decrease, 2=hold, 3=small increase, 4=big increase)

======================================================================
REPLICATING EVALCALLBACK'S EVALUATION
======================================================================
EvalCallback-style evaluation: 1271.53 ± 140.67
(This should match TensorBoard's ~1,231)

======================================================================
TESTING WITH FRESH ENVIRONMENT
======================================================================
Episode 1: -578.09
Episode 2: -546.67
Episode 3: -525.60
Episode 4: -574.12
Episode 5: -347.05
Episode 6: -461.00
Episode 7: -236.70
Episode 8: -695.32
Episode 9: -546.61
Episode 10: -476.93

======================================================================
FRESH ENV RESULTS: -499 ± 122
======================================================================

======================================================================
COMPARISON SUMMARY
======================================================================
TensorBoard eval/mean_reward:  ~1,231 (from training)
EvalCallback replication:      1272 ± 122
Fresh environment test:        -499 ± 122
======================================================================

✅ EvalCallback matches TensorBoard
❌ HUGE MISMATCH: eval_env works, fresh env fails!
   → Problem: eval_env has different state/config than fresh env

======================================================================
Model saved to: models/easy-20251208_031155/
======================================================================
"""