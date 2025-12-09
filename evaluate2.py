# evaluate_comprehensive.py
import gymnasium as gym
import envs
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def _test_model_on_env(model_path, env_name, n_episodes=20):
    """Test a model on an environment."""
    if not os.path.exists(model_path + ".zip"):
        print(f"   ⚠️  Model not found: {model_path}")
        return None, None, None
    
    model = PPO.load(model_path)
    test_env = gym.make(env_name)
    test_env = Monitor(test_env)
    
    rewards = []
    for i in range(n_episodes):
        obs, _ = test_env.reset()
        total_reward = 0
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            if done or truncated:
                break
        rewards.append(total_reward)
    
    return np.mean(rewards), np.std(rewards), rewards

def test_model_on_env(model_path, env_name, n_episodes=20):
    """Test a model on an environment using evaluate_policy."""
    if not os.path.exists(model_path + ".zip"):
        print(f"   ⚠️  Model not found: {model_path}")
        return None, None, None
    
    model = PPO.load(model_path)
    test_env = gym.make(env_name)
    test_env = Monitor(test_env)
    
    # USE evaluate_policy - THE ONLY METHOD THAT WORKS
    mean_reward, std_reward = evaluate_policy(
        model,
        test_env,
        n_eval_episodes=n_episodes,
        deterministic=True,
        return_episode_rewards=False
    )
    
    # Return dummy rewards list (not used for anything critical)
    rewards = [mean_reward] * n_episodes
    
    return mean_reward, std_reward, rewards

def _test_baseline_on_env(policy_fn, env_name, n_episodes=20):
    """Test a baseline policy on an environment."""
    test_env = gym.make(env_name)
    test_env = Monitor(test_env)
    
    rewards = []
    for i in range(n_episodes):
        obs, _ = test_env.reset()
        total_reward = 0
        for step in range(500):
            action = policy_fn(obs, step)
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            if done or truncated:
                break
        rewards.append(total_reward)
    
    return np.mean(rewards), np.std(rewards), rewards

def test_baseline_on_env(policy_fn, env_name, n_episodes=20):
    """Test a baseline policy on an environment."""
    test_env = gym.make(env_name)
    test_env = Monitor(test_env)
    
    rewards = []
    for i in range(n_episodes):
        obs, _ = test_env.reset()
        total_reward = 0
        for step in range(500):
            action = policy_fn(obs, step)
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            if done or truncated:
                break
        rewards.append(total_reward)
    
    return np.mean(rewards), np.std(rewards), rewards


# Define baseline policies
def always_increase(obs, step):
    return 4

def always_decrease(obs, step):
    return 0

def always_hold(obs, step):
    return 2

def aimd_like(obs, step):
    loss_rate = obs[3]
    queue_norm = obs[1]
    if loss_rate > 0.01 or queue_norm > 0.8:
        return 0
    else:
        return 3

def queue_based(obs, step):
    queue_norm = obs[1]
    if queue_norm < 0.3:
        return 3
    elif queue_norm > 0.7:
        return 1
    else:
        return 2

def gradual_increase(obs, step):
    queue_norm = obs[1]
    loss_rate = obs[3]
    if loss_rate > 0.05:
        return 0
    elif queue_norm > 0.9:
        return 1
    elif queue_norm > 0.6:
        return 2
    else:
        return 3


print("="*80)
print("COMPREHENSIVE EVALUATION")
print("="*80)

# Define models and environments
# NOTE: Easy model is optional - comment out if you don't have one
models = { # CURRENT BEST
    #'Easy Model': "models/easy-20251208_133412/final",
    # 'Easy Model': "models/easy-20251208_140812/best_model", # NOTE: BEST
    'Easy Model': "models/easy-20251208_140812/final", #NOTE: EVEN BETTER
    # "models/easy-20251208_133412/best_model"
    'Medium Model (Curriculum)': 'models/medium-curriculum/best_model',
    #'Medium Model (Curriculum)': 'models/curriculum-v2/medium/final',
    'Hard Model (Curriculum)': 'models/hard-curriculum/best_model',
}

models = {
    'Easy Model': "models/easy-20251208_140812/final", #NOTE: EVEN BETTER
    'Medium Model (Curriculum)': 'models/medium-v2-20251208_162738/final',
    'Hard Model (Curriculum)': 'models/hard-v2-20251208_175714/final',
}

environments = {
    'Easy': 'CongestionControl-Easy-v0',
    'Medium': 'CongestionControl-Medium-v0',
    'Hard': 'CongestionControl-Hard-v0',
}

baselines = {
    'Always Increase': always_increase,
    'Always Decrease': always_decrease,
    'Always Hold': always_hold,
    'AIMD-like': aimd_like,
    'Queue-based': queue_based,
    'Gradual Increase': gradual_increase,
}

# ========== TEST RL MODELS ==========
print("\n" + "="*80)
print("TESTING RL MODELS ON ALL ENVIRONMENTS")
print("="*80)

rl_results = {}

for model_name, model_path in models.items():
    rl_results[model_name] = {}
    print(f"\nTesting {model_name}...")
    
    for env_name, env_id in environments.items():
        print(f"  On {env_name} environment...", end=" ")
        mean, std, rewards = test_model_on_env(model_path, env_id, n_episodes=20)
        
        if mean is None:
            rl_results[model_name][env_name] = {
                'mean': float('nan'),
                'std': float('nan'),
                'rewards': []
            }
            print("SKIPPED (model not found)")
        else:
            rl_results[model_name][env_name] = {
                'mean': mean,
                'std': std,
                'rewards': rewards
            }
            print(f"{mean:.1f} ± {std:.1f}")

# ========== TEST BASELINES ==========
print("\n" + "="*80)
print("TESTING BASELINE POLICIES ON ALL ENVIRONMENTS")
print("="*80)

baseline_results = {}

for baseline_name, policy_fn in baselines.items():
    baseline_results[baseline_name] = {}
    print(f"\nTesting {baseline_name}...")
    
    for env_name, env_id in environments.items():
        print(f"  On {env_name} environment...", end=" ")
        mean, std, rewards = test_baseline_on_env(policy_fn, env_id, n_episodes=20)
        baseline_results[baseline_name][env_name] = {
            'mean': mean,
            'std': std,
            'rewards': rewards
        }
        print(f"{mean:.1f} ± {std:.1f}")

# ========== CREATE COMPARISON TABLES ==========
print("\n" + "="*80)
print("RL MODELS PERFORMANCE MATRIX")
print("="*80)

# Create DataFrame for RL models
rl_data = []
for model_name in models.keys():
    row = {'Model': model_name}
    for env_name in environments.keys():
        mean = rl_results[model_name][env_name]['mean']
        std = rl_results[model_name][env_name]['std']
        if np.isnan(mean):
            row[env_name] = "N/A"
        else:
            row[env_name] = f"{mean:.0f}±{std:.0f}"
    rl_data.append(row)

rl_df = pd.DataFrame(rl_data)
print(rl_df.to_string(index=False))

print("\n" + "="*80)
print("BASELINE POLICIES PERFORMANCE MATRIX")
print("="*80)

# Create DataFrame for baselines
baseline_data = []
for baseline_name in baselines.keys():
    row = {'Policy': baseline_name}
    for env_name in environments.keys():
        mean = baseline_results[baseline_name][env_name]['mean']
        std = baseline_results[baseline_name][env_name]['std']
        row[env_name] = f"{mean:.0f}±{std:.0f}"
    baseline_data.append(row)

baseline_df = pd.DataFrame(baseline_data)
print(baseline_df.to_string(index=False))

# ========== FIND BEST PERFORMERS ==========
print("\n" + "="*80)
print("BEST PERFORMER FOR EACH ENVIRONMENT")
print("="*80)

ranking_lists = {}
for env_name in environments.keys():
    
    # Collect all results for this environment
    all_results = {}
    
    # Add RL models
    # Assuming 'models' and 'rl_results' are defined, and np.isnan is available
    for model_name in models.keys():
        mean = rl_results[model_name][env_name]['mean']
        if not np.isnan(mean):
            all_results[model_name] = mean
    
    # Add baselines
    # Assuming 'baselines' and 'baseline_results' are defined
    for baseline_name in baselines.keys():
        all_results[baseline_name] = baseline_results[baseline_name][env_name]['mean']
    
    # Sort results
    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    
    # Create the list of tuples for the top 5 policies
    top_5_list = []
    for i, (name, score) in enumerate(sorted_results[:5], 1):
        # Determine the emoji for the rank
        trophy = "🏆" if i == 1 else ""
        
        # Append the tuple (Name, Score, Emoji) to the list
        top_5_list.append((name, score, trophy))
        
    # Assign the list to the main dictionary under the environment name
    ranking_lists[env_name] = top_5_list

# ========== CURRICULUM PROGRESSION ==========
print("\n" + "="*80)
print("CURRICULUM PROGRESSION ANALYSIS")
print("="*80)
print("\nHow well does each model generalize?\n")

for model_name in models.keys():
    print(f"{model_name}:")
    for env_name in ['Easy', 'Medium', 'Hard']:
        mean = rl_results[model_name][env_name]['mean']
        std = rl_results[model_name][env_name]['std']
        
        if np.isnan(mean):
            print(f"  {env_name:8s}: N/A")
            continue
        
        # Mark training environment
        marker = ""
        if 'Medium' in model_name and env_name == 'Medium':
            marker = " ← TRAINED ON THIS"
        elif 'Hard' in model_name and env_name == 'Hard':
            marker = " ← TRAINED ON THIS"
        elif 'Easy' in model_name and env_name == 'Easy':
            marker = " ← TRAINED ON THIS"
        
        print(f"  {env_name:8s}: {mean:7.1f} ± {std:5.1f}{marker}")
    print()

# ========== GENERALIZATION ANALYSIS ==========
print("="*80)
print("GENERALIZATION ANALYSIS")
print("="*80)

# Check if we have medium and hard models
if 'Medium Model (Curriculum)' in rl_results and 'Hard Model (Curriculum)' in rl_results:
    medium_on_easy = rl_results['Medium Model (Curriculum)']['Easy']['mean']
    medium_on_medium = rl_results['Medium Model (Curriculum)']['Medium']['mean']
    medium_on_hard = rl_results['Medium Model (Curriculum)']['Hard']['mean']
    
    hard_on_easy = rl_results['Hard Model (Curriculum)']['Easy']['mean']
    hard_on_medium = rl_results['Hard Model (Curriculum)']['Medium']['mean']
    hard_on_hard = rl_results['Hard Model (Curriculum)']['Hard']['mean']
    
    print(f"\nMedium Model generalizes:")
    print(f"  To Easy:   {medium_on_easy:.1f}")
    print(f"  On Medium: {medium_on_medium:.1f} ← TRAINED")
    print(f"  To Hard:   {medium_on_hard:.1f}")
    
    print(f"\nHard Model generalizes:")
    print(f"  To Easy:   {hard_on_easy:.1f}")
    print(f"  To Medium: {hard_on_medium:.1f}")
    print(f"  On Hard:   {hard_on_hard:.1f} ← TRAINED")

# ========== VISUALIZATION ==========
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Only create visualizations if we have RL models
valid_models = [m for m in models.keys() if not np.isnan(rl_results[m]['Easy']['mean'])]

if valid_models:
    # Plot 1: RL Models Heatmap
    ax = axes[0, 0]
    rl_matrix = np.array([
        [rl_results[model][env]['mean'] for env in environments.keys()]
        for model in valid_models
    ])
    im = ax.imshow(rl_matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(environments)))
    ax.set_yticks(range(len(valid_models)))
    ax.set_xticklabels(environments.keys())
    ax.set_yticklabels(valid_models)
    ax.set_title('RL Models Performance Heatmap', fontsize=14, fontweight='bold')
    for i in range(len(valid_models)):
        for j in range(len(environments)):
            text = ax.text(j, i, f'{rl_matrix[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=10)
    plt.colorbar(im, ax=ax)
    
    # Plot 2: Comparison bars for each environment
    ax = axes[0, 1]
    x = np.arange(len(environments))
    width = 0.35 if len(valid_models) == 2 else 0.25
    for i, model_name in enumerate(valid_models):
        means = [rl_results[model_name][env]['mean'] for env in environments.keys()]
        ax.bar(x + i*width, means, width, label=model_name, alpha=0.8)
    ax.set_xlabel('Environment')
    ax.set_ylabel('Average Reward')
    ax.set_title('RL Models: Performance by Environment', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(environments.keys())
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Best baseline vs best RL for each environment
    ax = axes[1, 0]
    env_names = list(environments.keys())
    best_rl = []
    best_baseline = []
    
    for env_name in env_names:
        # Best RL
        rl_scores = [rl_results[m][env_name]['mean'] for m in valid_models]
        best_rl.append(max(rl_scores))
        
        # Best baseline
        baseline_scores = [baseline_results[b][env_name]['mean'] for b in baselines.keys()]
        best_baseline.append(max(baseline_scores))
    
    x = np.arange(len(env_names))
    width = 0.35
    ax.bar(x - width/2, best_rl, width, label='Best RL Model', alpha=0.8, color='green')
    ax.bar(x + width/2, best_baseline, width, label='Best Baseline', alpha=0.8, color='orange')
    ax.set_xlabel('Environment')
    ax.set_ylabel('Average Reward')
    ax.set_title('Best RL vs Best Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(env_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Variance comparison
    ax = axes[1, 1]
    model_names_short = [m.split()[0] for m in valid_models]
    
    variances_by_env = {}
    for env_name in environments.keys():
        variances_by_env[env_name] = [rl_results[m][env_name]['std'] for m in valid_models]
    
    x = np.arange(len(model_names_short))
    width = 0.25
    for i, (env_name, variances) in enumerate(variances_by_env.items()):
        ax.bar(x + i*width - width, variances, width, label=f'{env_name} Env', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Performance Variance (Lower = More Stable)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names_short)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('comprehensive_evaluation.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved: comprehensive_evaluation.png")

#plt.show()

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
print("\nKey Findings:")
print(f"  • Tested {len(valid_models)} RL models on {len(environments)} environments")
print(f"  • Tested {len(baselines)} baseline policies")
print(f"  • Results saved to: comprehensive_evaluation.png")
print("="*80)