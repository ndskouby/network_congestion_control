# baseline_detailed.py
import gymnasium as gym
import envs
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

def run_policy(env, policy_fn, policy_name, n_steps=500):
    """Run a policy and collect detailed metrics."""
    obs, _ = env.reset()
    metrics = {
        'throughput': [],
        'rtt': [],
        'loss': [],
        'queue': [],
        'sending_rate': [],
        'action': [],
        'reward': []
    }
    
    total_reward = 0
    for step in range(n_steps):
        action = policy_fn(obs, step)
        obs, reward, done, truncated, info = env.step(action)
        
        metrics['throughput'].append(info['throughput_mbps'])
        metrics['rtt'].append(info['rtt_s'] * 1000)  # Convert to ms
        metrics['loss'].append(obs[3] * 100)  # Convert to percentage
        metrics['queue'].append(info['queue_pkts'])
        metrics['sending_rate'].append(obs[0])
        metrics['action'].append(action)
        metrics['reward'].append(reward)
        total_reward += reward
        
        if done or truncated:
            break
    
    return metrics, total_reward


# Define baseline policies
def always_increase(obs, step):
    return 4

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
    """The winning baseline!"""
    queue_norm = obs[1]
    loss_rate = obs[3]
    if loss_rate > 0.05:
        return 0  # Large decrease
    elif queue_norm > 0.9:
        return 1  # Small decrease
    elif queue_norm > 0.6:
        return 2  # Hold
    else:
        return 3  # Small increase


def evaluate_baselines(rl_models=None, env_name="CongestionControl-Hard-v0", save_plot=True):
    """
    Evaluate baseline policies and RL models.
    
    Args:
        rl_models: Dict of RL models {"Model Name": "path/to/model"}
        env_name: Which environment to test on
        save_plot: Whether to save the comparison plot
    """
    env = gym.make(env_name)
    env = Monitor(env)
    
    print("="*80)
    print(f"DETAILED BASELINE EVALUATION ON {env_name}")
    print("="*80)
    
    env.reset(seed=42)

    print(f"\nEnvironment parameters (sample from first reset):")
    print(f"  Bandwidth: {env.unwrapped.link_capacity_mbps:.1f} Mbps")
    print(f"  RTT: {env.unwrapped.base_rtt_s*1000:.1f} ms")
    print(f"  Queue: {env.unwrapped.queue_capacity_pkts} packets")
    
    # Define baseline policies
    baselines = {
        'AIMD-like': aimd_like,
        'Queue-based': queue_based,
        'Gradual Increase': gradual_increase,
    }
    
    # Load RL models
    rl_policies = {}
    if rl_models:
        print(f"\n{'='*80}")
        print("LOADING RL MODELS")
        print("="*80)
        for model_name, model_path in rl_models.items():
            try:
                if not os.path.exists(model_path + ".zip"):
                    print(f"⚠️  {model_name}: File not found - {model_path}")
                    continue
                    
                rl_model = PPO.load(model_path)
                
                # Create closure to capture model
                def make_rl_policy(m):
                    def policy(obs, step):
                        action, _ = m.predict(obs, deterministic=True)
                        return action
                    return policy
                
                rl_policies[model_name] = make_rl_policy(rl_model)
                print(f"✓ {model_name}: Loaded from {model_path}")
            except Exception as e:
                print(f"⚠️  {model_name}: Error loading - {e}")
    
    # Combine all policies
    all_policies = {**baselines, **rl_policies}
    
    # Run all policies
    print("\n" + "="*80)
    print("RUNNING POLICIES")
    print("="*80)
    
    results = {}
    for name, policy in all_policies.items():
        print(f"\nRunning {name}...")
        metrics, total_reward = run_policy(env, policy, name)
        results[name] = {
            'metrics': metrics,
            'total_reward': total_reward,
            'avg_throughput': np.mean(metrics['throughput']),
            'avg_rtt': np.mean(metrics['rtt']),
            'avg_loss': np.mean(metrics['loss']),
            'avg_queue': np.mean(metrics['queue']),
            'utilization': np.mean(metrics['throughput']) / env.unwrapped.link_capacity_mbps * 100
        }
        print(f"  Reward: {total_reward:.1f}")
        print(f"  Avg Throughput: {results[name]['avg_throughput']:.2f} Mbps ({results[name]['utilization']:.1f}% util)")
        print(f"  Avg RTT: {results[name]['avg_rtt']:.2f} ms")
        print(f"  Avg Loss: {results[name]['avg_loss']:.2f}%")
    
    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Policy':<35} {'Reward':>10} {'Throughput':>12} {'RTT (ms)':>10} {'Loss %':>8} {'Util %':>8}")
    print("-"*80)
    
    # Sort by reward
    sorted_results = sorted(results.items(), key=lambda x: x[1]['total_reward'], reverse=True)
    
    for name, result in sorted_results:
        print(f"{name:<35} {result['total_reward']:>10.1f} "
              f"{result['avg_throughput']:>10.2f} Mbps "
              f"{result['avg_rtt']:>10.2f} "
              f"{result['avg_loss']:>7.2f}% "
              f"{result['utilization']:>7.1f}%")
    
    print("="*80)
    
    # Find best policy
    best_policy = max(results.items(), key=lambda x: x[1]['total_reward'])
    print(f"\n🏆 Best Policy: {best_policy[0]} (Reward: {best_policy[1]['total_reward']:.1f})")
    
    # Plotting
    if save_plot:
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        
        # Plot 1: Throughput over time
        ax = axes[0, 0]
        for name, result in results.items():
            linewidth = 2.5 if name in rl_policies else 1.5
            alpha = 0.9 if name in rl_policies else 0.6
            ax.plot(result['metrics']['throughput'], label=name, alpha=alpha, linewidth=linewidth)
        ax.axhline(y=env.unwrapped.link_capacity_mbps, color='black', linestyle='--', 
                   linewidth=1.5, label='Link Capacity', alpha=0.7)
        ax.set_ylabel('Throughput (Mbps)', fontsize=11)
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_title('Throughput Over Time', fontweight='bold', fontsize=12)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: RTT over time
        ax = axes[0, 1]
        for name, result in results.items():
            linewidth = 2.5 if name in rl_policies else 1.5
            alpha = 0.9 if name in rl_policies else 0.6
            ax.plot(result['metrics']['rtt'], label=name, alpha=alpha, linewidth=linewidth)
        ax.axhline(y=env.unwrapped.base_rtt_s*1000, color='black', linestyle='--',
                   linewidth=1.5, label='Base RTT', alpha=0.7)
        ax.set_ylabel('RTT (ms)', fontsize=11)
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_title('RTT Over Time', fontweight='bold', fontsize=12)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Queue occupancy
        ax = axes[0, 2]
        for name, result in results.items():
            linewidth = 2.5 if name in rl_policies else 1.5
            alpha = 0.9 if name in rl_policies else 0.6
            ax.plot(result['metrics']['queue'], label=name, alpha=alpha, linewidth=linewidth)
        ax.axhline(y=env.unwrapped.queue_capacity_pkts, color='black', linestyle='--',
                   linewidth=1.5, label='Queue Capacity', alpha=0.7)
        ax.set_ylabel('Queue Occupancy (packets)', fontsize=11)
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_title('Queue Occupancy Over Time', fontweight='bold', fontsize=12)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Loss rate over time
        ax = axes[1, 0]
        for name, result in results.items():
            linewidth = 2.5 if name in rl_policies else 1.5
            alpha = 0.9 if name in rl_policies else 0.6
            ax.plot(result['metrics']['loss'], label=name, alpha=alpha, linewidth=linewidth)
        ax.set_ylabel('Loss Rate (%)', fontsize=11)
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_title('Packet Loss Over Time', fontweight='bold', fontsize=12)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Action distributions for all RL models
        ax = axes[1, 1]
        if rl_policies:
            action_labels = ['Big\nDec', 'Small\nDec', 'Hold', 'Small\nInc', 'Big\nInc']
            x = np.arange(len(action_labels))
            width = 0.8 / len(rl_policies) if len(rl_policies) > 1 else 0.5
            
            for i, (name, result) in enumerate([(n, results[n]) for n in rl_policies.keys()]):
                actions = result['metrics']['action']
                action_counts = [actions.count(j) for j in range(5)]
                offset = (i - len(rl_policies)/2 + 0.5) * width
                ax.bar(x + offset, action_counts, width, label=name, alpha=0.7)
            
            ax.set_ylabel('Count', fontsize=11)
            ax.set_xlabel('Action', fontsize=11)
            ax.set_title('RL Models Action Distribution', fontweight='bold', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(action_labels)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No RL Agents', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Plot 6: Bar chart comparison
        ax = axes[1, 2]
        policies = list(sorted_results[0][0] for _ in range(len(sorted_results)))
        policies = [name for name, _ in sorted_results]
        avg_throughput = [sorted_results[i][1]['avg_throughput'] for i in range(len(sorted_results))]
        
        x_pos = np.arange(len(policies))
        colors = ['green' if name in rl_policies else 'orange' if name == 'Gradual Increase' 
                  else 'steelblue' for name in policies]
        bars = ax.barh(x_pos, avg_throughput, color=colors, alpha=0.7)
        
        ax.set_xlabel('Average Throughput (Mbps)', fontsize=11)
        ax.set_title('Throughput Ranking', fontweight='bold', fontsize=12)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(policies, fontsize=8)
        ax.axvline(x=env.unwrapped.link_capacity_mbps, color='red', linestyle='--',
                   linewidth=1.5, alpha=0.7, label='Link Capacity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Create filename
        env_short = env_name.split('-')[1] if '-' in env_name else env_name
        plot_name = f"detailed_comparison_{env_short}.png"
        
        plt.savefig(plot_name, dpi=150, bbox_inches='tight')
        print(f"\n📊 Detailed comparison saved to '{plot_name}'")
        plt.show()
    
    return results


if __name__ == "__main__":
    # Define RL models to test
    # models = {
    #     #'Easy Model (final)': "models/easy-20251208_133412/final",
    #     'Easy Model (final)': "models/easy-20251208_140812/final",
    #     # 'Easy Model (best)': "models/easy-20251208_133412/best_model",  # This one is bad
    #     'Medium Model': 'models/medium-curriculum/best_model',
    #     'Hard Model': 'models/hard-curriculum/best_model',
    # }
    models = {
        'Easy Model': "models/easy-20251208_140812/final", #NOTE: EVEN BETTER
        'Medium Model (Curriculum)': 'models/medium-v2-20251208_162738/final',
        'Hard Model (Curriculum)': 'models/hard-v2-20251208_175714/final',
    }
    
    # Test on Hard environment
    print("\n" + "="*80)
    print("TESTING ON HARD ENVIRONMENT")
    print("="*80)
    evaluate_baselines(
        rl_models=models,
        env_name="CongestionControl-Hard-v0"
    )
    
