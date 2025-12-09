# evaluate_metrics.py
import gymnasium as gym
import envs
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def collect_episode_metrics(env, policy_fn, n_steps=500):
    """Run one episode and collect all metrics."""
    obs, _ = env.reset()
    
    metrics = {
        'throughput': [],
        'rtt': [],
        'loss': [],
        'queue': [],
        'utilization': []
    }
    
    for step in range(n_steps):
        action = policy_fn(obs, step)
        obs, reward, done, truncated, info = env.step(action)
        
        metrics['throughput'].append(info['throughput_mbps'])
        metrics['rtt'].append(info['rtt_s'] * 1000)  # Convert to ms
        metrics['loss'].append(obs[3] * 100)  # Convert to %
        metrics['queue'].append(info['queue_pkts'])
        
        # Calculate utilization
        util = info['throughput_mbps'] / env.unwrapped.link_capacity_mbps * 100
        metrics['utilization'].append(util)
        
        if done or truncated:
            break
    
    # Return episode averages
    return {
        'avg_throughput': np.mean(metrics['throughput']),
        'avg_rtt': np.mean(metrics['rtt']),
        'avg_loss': np.mean(metrics['loss']),
        'avg_queue': np.mean(metrics['queue']),
        'avg_utilization': np.mean(metrics['utilization']),
        'std_throughput': np.std(metrics['throughput']),
        'std_rtt': np.std(metrics['rtt']),
        'std_loss': np.std(metrics['loss']),
    }


def evaluate_model_metrics(model_path, env_name, n_episodes=20):
    """Evaluate RL model across multiple episodes."""
    model = PPO.load(model_path)
    env = gym.make(env_name)
    env = Monitor(env)
    
    def rl_policy(obs, step):
        action, _ = model.predict(obs, deterministic=True)
        return action
    
    episode_results = []
    for i in range(n_episodes):
        metrics = collect_episode_metrics(env, rl_policy)
        episode_results.append(metrics)
    
    # Average across episodes
    avg_metrics = {
        'throughput': np.mean([e['avg_throughput'] for e in episode_results]),
        'throughput_std': np.std([e['avg_throughput'] for e in episode_results]),
        'rtt': np.mean([e['avg_rtt'] for e in episode_results]),
        'rtt_std': np.std([e['avg_rtt'] for e in episode_results]),
        'loss': np.mean([e['avg_loss'] for e in episode_results]),
        'loss_std': np.std([e['avg_loss'] for e in episode_results]),
        'queue': np.mean([e['avg_queue'] for e in episode_results]),
        'queue_std': np.std([e['avg_queue'] for e in episode_results]),
        'utilization': np.mean([e['avg_utilization'] for e in episode_results]),
        'utilization_std': np.std([e['avg_utilization'] for e in episode_results]),
    }
    
    return avg_metrics


def evaluate_baseline_metrics(policy_fn, env_name, n_episodes=20):
    """Evaluate baseline policy across multiple episodes."""
    env = gym.make(env_name)
    env = Monitor(env)
    
    episode_results = []
    for i in range(n_episodes):
        metrics = collect_episode_metrics(env, policy_fn)
        episode_results.append(metrics)
    
    # Average across episodes
    avg_metrics = {
        'throughput': np.mean([e['avg_throughput'] for e in episode_results]),
        'throughput_std': np.std([e['avg_throughput'] for e in episode_results]),
        'rtt': np.mean([e['avg_rtt'] for e in episode_results]),
        'rtt_std': np.std([e['avg_rtt'] for e in episode_results]),
        'loss': np.mean([e['avg_loss'] for e in episode_results]),
        'loss_std': np.std([e['avg_loss'] for e in episode_results]),
        'queue': np.mean([e['avg_queue'] for e in episode_results]),
        'queue_std': np.std([e['avg_queue'] for e in episode_results]),
        'utilization': np.mean([e['avg_utilization'] for e in episode_results]),
        'utilization_std': np.std([e['avg_utilization'] for e in episode_results]),
    }
    
    return avg_metrics


# Define baseline policies
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

def aimd_like(obs, step):
    loss_rate = obs[3]
    queue_norm = obs[1]
    if loss_rate > 0.01 or queue_norm > 0.8:
        return 0
    else:
        return 3


if __name__ == "__main__":
    print("="*80)
    print("DETAILED METRICS EVALUATION")
    print("="*80)
    
    # Define models to test
    # models = {
    #     'Easy Model': 'models/easy-20251208_140812/final',
    #     'Medium v2': 'models/medium-v2-20251208_162738/best_model',
    #     'Hard v2': 'models/hard-v2-TIMESTAMP/best_model',  # UPDATE TIMESTAMP
    # }

    models = {
        'Easy Model': "models/easy-20251208_140812/final", #NOTE: EVEN BETTER
        'Medium Model (Curriculum)': 'models/medium-v2-20251208_162738/best_model',
        'Hard Model (Curriculum)': 'models/hard-v2-20251208_175714/best_model',
    }
    models = {
        'Easy Model (final)': "models/easy-20251208_133412/final",
        #'Easy Model (final)': "models/easy-20251208_140812/best_model",
        'Medium Model': 'models/medium-curriculum/best_model',
        'Hard Model': 'models/hard-curriculum/best_model',
    }
    
    # Define baselines
    baselines = {
        'Gradual Increase': gradual_increase,
        'AIMD-like': aimd_like,
    }
    
    # Test environment
    env_name = "CongestionControl-Hard-v0"
    
    print(f"\nTesting on {env_name}")
    print(f"Running 20 episodes per policy...\n")
    
    all_results = {}
    
    # Evaluate RL models
    print("\nEvaluating RL Models:")
    print("-"*80)
    for model_name, model_path in models.items():
        print(f"Testing {model_name}...", end=" ")
        try:
            metrics = evaluate_model_metrics(model_path, env_name, n_episodes=20)
            all_results[model_name] = metrics
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")
    
    # Evaluate baselines
    print("\nEvaluating Baselines:")
    print("-"*80)
    for baseline_name, policy_fn in baselines.items():
        print(f"Testing {baseline_name}...", end=" ")
        metrics = evaluate_baseline_metrics(policy_fn, env_name, n_episodes=20)
        all_results[baseline_name] = metrics
        print("✓")
    
    # Display results
    print("\n" + "="*80)
    print("METRICS SUMMARY (averaged across 20 episodes)")
    print("="*80)
    
    # Create DataFrame
    df_data = []
    for policy_name, metrics in all_results.items():
        df_data.append({
            'Policy': policy_name,
            'Utilization (%)': f"{metrics['utilization']:.1f}±{metrics['utilization_std']:.1f}",
            'Throughput (Mbps)': f"{metrics['throughput']:.2f}±{metrics['throughput_std']:.2f}",
            'RTT (ms)': f"{metrics['rtt']:.1f}±{metrics['rtt_std']:.1f}",
            'Loss (%)': f"{metrics['loss']:.2f}±{metrics['loss_std']:.2f}",
            'Queue (pkts)': f"{metrics['queue']:.0f}±{metrics['queue_std']:.0f}",
        })
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    print("="*80)
    
    # Find best performers
    print("\n" + "="*80)
    print("BEST PERFORMERS")
    print("="*80)
    
    best_util = max(all_results.items(), key=lambda x: x[1]['utilization'])
    best_loss = min(all_results.items(), key=lambda x: x[1]['loss'])
    best_rtt = min(all_results.items(), key=lambda x: x[1]['rtt'])
    
    print(f"🏆 Highest Utilization: {best_util[0]} ({best_util[1]['utilization']:.1f}%)")
    print(f"🏆 Lowest Loss:         {best_loss[0]} ({best_loss[1]['loss']:.2f}%)")
    print(f"🏆 Lowest RTT:          {best_rtt[0]} ({best_rtt[1]['rtt']:.1f} ms)")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    policies = list(all_results.keys())
    
    # Plot 1: Utilization
    ax = axes[0, 0]
    utils = [all_results[p]['utilization'] for p in policies]
    util_stds = [all_results[p]['utilization_std'] for p in policies]
    x = np.arange(len(policies))
    ax.bar(x, utils, yerr=util_stds, alpha=0.7, capsize=5)
    ax.set_ylabel('Utilization (%)')
    ax.set_title('Link Utilization', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Packet Loss
    ax = axes[0, 1]
    loss = [all_results[p]['loss'] for p in policies]
    loss_stds = [all_results[p]['loss_std'] for p in policies]
    ax.bar(x, loss, yerr=loss_stds, alpha=0.7, capsize=5, color='orange')
    ax.set_ylabel('Packet Loss (%)')
    ax.set_title('Packet Loss Rate', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: RTT
    ax = axes[1, 0]
    rtt = [all_results[p]['rtt'] for p in policies]
    rtt_stds = [all_results[p]['rtt_std'] for p in policies]
    ax.bar(x, rtt, yerr=rtt_stds, alpha=0.7, capsize=5, color='green')
    ax.set_ylabel('RTT (ms)')
    ax.set_title('Round Trip Time', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Throughput
    ax = axes[1, 1]
    tput = [all_results[p]['throughput'] for p in policies]
    tput_stds = [all_results[p]['throughput_std'] for p in policies]
    ax.bar(x, tput, yerr=tput_stds, alpha=0.7, capsize=5, color='red')
    ax.set_ylabel('Throughput (Mbps)')
    ax.set_title('Average Throughput', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Metrics comparison saved to 'metrics_comparison.png'")
    plt.show()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

