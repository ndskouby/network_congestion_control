import gymnasium as gym
import envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
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

# RIGHT AFTER model.save(), BEFORE creating test_env
print("\n" + "="*70)
print("CHECKING IF MODEL STATE IS SANE")
print("="*70)

# Get model's network parameters
policy_params = model.policy.state_dict()
print(f"Number of parameters: {sum(p.numel() for p in policy_params.values())}")
print(f"First layer weights sample: {list(policy_params.values())[0][:3, :3]}")

# Try a single prediction on a reasonable state
fake_obs = np.array([6.5, 0.5, 0.05, 0.01, 6.0], dtype=np.float32)
fake_action, _ = model.predict(fake_obs, deterministic=True)
print(f"\nTest prediction on reasonable state:")
print(f"  Input obs: {fake_obs}")
print(f"  Predicted action: {fake_action}")
print(f"  (0=big decrease, 1=small decrease, 2=hold, 3=small increase, 4=big increase)")


print(f"\n{'='*70}")
print("IMMEDIATE TEST (same script, same Python process)")
print(f"{'='*70}")

# CREATE FRESH TEST ENVIRONMENT (with Monitor for consistency)
test_env = gym.make("CongestionControl-Easy-v0")
test_env = Monitor(test_env)  # ← ADDED Monitor

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

avg_reward = np.mean(rewards)
std_reward = np.std(rewards)

print(f"\n{'='*70}")
print(f"RESULTS: {avg_reward:.0f} ± {std_reward:.0f}")
print(f"{'='*70}")

if avg_reward < 0:
    print("❌ TRAINING ITSELF IS BROKEN - The model never learned")
    print("   Check TensorBoard, reward function, or environment config")
elif avg_reward > 800:
    print("✅ TRAINING SUCCEEDED - Model learned correctly")
    print(f"   Model saved to: models/{model_name}/")
    print(f"   Load with: PPO.load('models/{model_name}/final')")
else:
    print("⚠️  MARGINAL PERFORMANCE - Model learned something but not great")
    print(f"   Expected >1000, got {avg_reward:.0f}")

print(f"\n{'='*70}")
print("NEXT STEP: Test by loading from file in separate script to verify")
print(f"{'='*70}")