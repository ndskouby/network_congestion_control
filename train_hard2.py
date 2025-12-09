# train_hard_v2.py
import gymnasium as gym
import envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import os
from datetime import datetime

# UNIQUE timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"hard-v2-{timestamp}"

print("="*80)
print(f"TRAINING HARD MODEL: {run_name}")
print("="*80)

# Create directories
os.makedirs(f"models/{run_name}", exist_ok=True)
os.makedirs(f"logs/{run_name}", exist_ok=True)

# Load best medium v2 model
print("\nLoading best medium v2 model...")
model = PPO.load("models/medium-v2-20251208_162738/best_model")
#model = PPO.load("models/medium-v2-20251208_162738/final")
print("✓ Medium v2 model loaded")

# Set up hard environment
print("\nSetting up HARD environment...")
train_env = gym.make("CongestionControl-Hard-v0")
train_env = Monitor(train_env)

eval_env = gym.make("CongestionControl-Hard-v0")
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

print("\n" + "="*80)
print("TRAINING (1M steps)")
print("="*80)
print(f"TensorBoard: tensorboard --logdir ./tensorboard/{run_name}/")

# Train
model.learn(
    total_timesteps=2_500_000,
    callback=eval_callback,
    progress_bar=True,
    reset_num_timesteps=False
)

# Save final
model.save(f"models/{run_name}/final")

print("\n✓ Training complete")

# ========== TESTING WITH evaluate_policy() ==========
print("\n" + "="*80)
print("TESTING RESULTS (using evaluate_policy)")
print("="*80)

test_results = {}

for env_name_full in ["CongestionControl-Easy-v0", "CongestionControl-Medium-v0", "CongestionControl-Hard-v0"]:
    env_short = env_name_full.split('-')[1]
    
    # Create fresh test environment
    test_env = gym.make(env_name_full)
    test_env = Monitor(test_env)
    
    # Use evaluate_policy - the ONLY method that works
    mean, std = evaluate_policy(
        model,
        test_env,
        n_eval_episodes=20,
        deterministic=True,
        return_episode_rewards=False
    )
    
    test_results[env_short] = (mean, std)
    print(f"{env_short}: {mean:.0f} ± {std:.0f}")

# ========== COMPARISON ==========
print("\n" + "="*80)
print("COMPARISON TO BASELINE")
print("="*80)
print(f"{'Environment':<12} {'Easy Model':<18} {'Medium v2':<18} {'Hard v2':<18} {'Best':<10}")
print("-"*80)

baselines = {
    'Easy': (1174, 95),
    'Medium': (1153, 191),
    'Hard': (975, 322)
}

# Medium v2 results (update with actual results if different)
medium_v2_results = {
    'Easy': (1200, 150),  # Replace with actual medium v2 test results
    'Medium': (1121, 180),
    'Hard': (1050, 250)
}

for env_name in ['Easy', 'Medium', 'Hard']:
    easy_mean = baselines[env_name][0]
    medium_mean = medium_v2_results[env_name][0]
    hard_mean, hard_std = test_results[env_name]
    
    best = max(easy_mean, medium_mean, hard_mean)
    best_model = "Easy" if best == easy_mean else "Medium" if best == medium_mean else "Hard"
    
    print(f"{env_name:<12} {easy_mean:.0f}{'':>14} {medium_mean:.0f}{'':>14} {hard_mean:.0f}±{hard_std:.0f}{'':>6} {best_model:<10}")

print("="*80)

# ========== VERDICT ==========
hard_baseline = baselines['Hard'][0]
hard_new = test_results['Hard'][0]

print("\n" + "="*80)
print("VERDICT")
print("="*80)

if hard_new > 1100:
    print(f"✅ EXCELLENT: Hard v2 scored {hard_new:.0f} (target was >1100)")
    print(f"   Use this model: models/{run_name}/best_model.zip")
elif hard_new > 1000:
    print(f"✅ GOOD: Hard v2 scored {hard_new:.0f} (solid improvement)")
    print(f"   Use this model: models/{run_name}/best_model.zip")
elif hard_new > 950:
    print(f"⚠️  OKAY: Hard v2 scored {hard_new:.0f} (marginal improvement)")
    print(f"   Consider which model generalizes best")
else:
    print(f"⚠️  MIXED: Hard v2 scored {hard_new:.0f}")
    print(f"   Compare all three models to choose best")

print("\n" + "="*80)
print("FINAL SUMMARY - ALL THREE MODELS")
print("="*80)
print(f"{'Model':<15} {'Easy':<12} {'Medium':<12} {'Hard':<12} {'Average':<10}")
print("-"*80)

easy_avg = sum(baselines.values(), ())[::2]
easy_avg = sum([baselines[k][0] for k in baselines]) / 3

medium_avg = sum([medium_v2_results[k][0] for k in medium_v2_results]) / 3

hard_avg = sum([test_results[k][0] for k in test_results]) / 3

print(f"{'Easy Model':<15} {baselines['Easy'][0]:<12.0f} {baselines['Medium'][0]:<12.0f} {baselines['Hard'][0]:<12.0f} {easy_avg:<10.0f}")
print(f"{'Medium v2':<15} {medium_v2_results['Easy'][0]:<12.0f} {medium_v2_results['Medium'][0]:<12.0f} {medium_v2_results['Hard'][0]:<12.0f} {medium_avg:<10.0f}")
print(f"{'Hard v2':<15} {test_results['Easy'][0]:<12.0f} {test_results['Medium'][0]:<12.0f} {test_results['Hard'][0]:<12.0f} {hard_avg:<10.0f}")

print("="*80)
print(f"\nModel saved to: models/{run_name}/")
print(f"TensorBoard: tensorboard --logdir ./tensorboard/{run_name}/")
print("="*80)

"""
================================================================================
Model           Easy         Medium       Hard         Average   
--------------------------------------------------------------------------------
Easy Model      1174         1153         975          1101      
Medium v2       1200         1121         1050         1124      
Hard v2         1327         1195         1051         1191      
================================================================================
"""