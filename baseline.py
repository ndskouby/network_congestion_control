# baseline.py
import gymnasium as gym
import envs
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
import sys

def run_policy(env, policy_fn, n_steps=500):
    """Run a policy and collect metrics."""
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
    """Always increase sending rate (large)."""
    return 4

def always_decrease(obs, step):
    """Always decrease sending rate (large)."""
    return 0

def always_hold(obs, step):
    """Always hold sending rate."""
    return 2

def random_policy(obs, step):
    """Random actions."""
    return np.random.randint(0, 5)

def aimd_like(obs, step):
    """Additive Increase Multiplicative Decrease (TCP-like)."""
    loss_rate = obs[3]
    queue_norm = obs[1]
    
    if loss_rate > 0.01 or queue_norm > 0.8:
        return 0  # Large decrease
    else:
        return 3  # Small increase

def queue_based(obs, step):
    """Control based on queue occupancy."""
    queue_norm = obs[1]
    
    if queue_norm < 0.3:
        return 3  # Small increase
    elif queue_norm > 0.7:
        return 1  # Small decrease
    else:
        return 2  # Hold

def gradual_increase(obs, step):
    """Gradual increase with queue-based backoff."""
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


def evaluate_baselines(model_path=None, save_plot=True):
    """
    Evaluate baseline policies and optionally an RL model.
    
    Args:
        model_path: Path to RL model (e.g., "models/model-1M/best_model")
        save_plot: Whether to save the comparison plot
    """
    env = gym.make("CongestionControl-v0")
    
    print("="*70)
    print("EVALUATING BASELINE POLICIES")
    if model_path:
        print(f"RL Model: {model_path}")
    print("="*70)
    
    baselines = {
        'Always Increase': always_increase,
        'Always Decrease': always_decrease,
        'Always Hold': always_hold,
        'Random': random_policy,
        'AIMD-like': aimd_like,
        'Queue-based': queue_based,
        'Gradual Increase': gradual_increase,
    }
    
    # Add RL agent if model path provided
    if model_path:
        try:
            rl_model = PPO.load(model_path)
            
            def rl_agent(obs, step):
                action, _ = rl_model.predict(obs, deterministic=True)
                return action
            
            baselines['RL Agent (PPO)'] = rl_agent
            print(f"✓ Loaded RL model from {model_path}")
        except Exception as e:
            print(f"⚠ Could not load RL model: {e}")
    
    # Run all policies
    results = {}
    for name, policy in baselines.items():
        print(f"\nRunning {name}...")
        metrics, total_reward = run_policy(env, policy)
        results[name] = {
            'metrics': metrics,
            'total_reward': total_reward,
            'avg_throughput': np.mean(metrics['throughput']),
            'avg_rtt': np.mean(metrics['rtt']),
            'avg_loss': np.mean(metrics['loss']),
            'avg_queue': np.mean(metrics['queue'])
        }
    
    # Print summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"{'Policy':<20} {'Reward':>10} {'Throughput':>12} {'RTT (ms)':>10} {'Loss %':>8}")
    print("-"*70)
    
    for name, result in results.items():
        print(f"{name:<20} {result['total_reward']:>10.1f} "
              f"{result['avg_throughput']:>10.2f} Mbps "
              f"{result['avg_rtt']:>10.2f} "
              f"{result['avg_loss']:>7.2f}%")
    
    print("="*70)
    
    # Find best policy
    best_policy = max(results.items(), key=lambda x: x[1]['total_reward'])
    print(f"\n🏆 Best Policy: {best_policy[0]} (Reward: {best_policy[1]['total_reward']:.1f})")
    
    # Plotting
    if save_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Throughput over time
        ax = axes[0, 0]
        for name, result in results.items():
            ax.plot(result['metrics']['throughput'], label=name, alpha=0.7)
        ax.set_ylabel('Throughput (Mbps)')
        ax.set_xlabel('Time Step')
        ax.set_title('Throughput Comparison')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: RTT over time
        ax = axes[0, 1]
        for name, result in results.items():
            ax.plot(result['metrics']['rtt'], label=name, alpha=0.7)
        ax.set_ylabel('RTT (ms)')
        ax.set_xlabel('Time Step')
        ax.set_title('RTT Comparison')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Queue occupancy
        ax = axes[1, 0]
        for name, result in results.items():
            ax.plot(result['metrics']['queue'], label=name, alpha=0.7)
        ax.set_ylabel('Queue Occupancy (packets)')
        ax.set_xlabel('Time Step')
        ax.set_title('Queue Occupancy Comparison')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Bar chart of average metrics
        ax = axes[1, 1]
        policies = list(results.keys())
        avg_throughput = [results[p]['avg_throughput'] for p in policies]
        x = np.arange(len(policies))
        bars = ax.bar(x, avg_throughput, alpha=0.7)
        ax.set_ylabel('Average Throughput (Mbps)')
        ax.set_title('Average Throughput by Policy')
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Color best bar
        best_idx = policies.index(best_policy[0])
        bars[best_idx].set_color('green')
        bars[best_idx].set_alpha(0.9)
        
        plt.tight_layout()
        
        # Create filename based on model
        if model_path:
            plot_name = f"baseline_comparison_{model_path.replace('/', '_').replace('.', '_')}.png"
        else:
            plot_name = 'baseline_comparison_no_rl.png'
        
        plt.savefig(plot_name, dpi=150)
        print(f"\n📊 Results saved to '{plot_name}'")
        plt.show()
    
    # Detailed comparison
    print("\n" + "="*70)
    print("DETAILED METRICS")
    print("="*70)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Total Reward:      {result['total_reward']:.2f}")
        print(f"  Avg Throughput:    {result['avg_throughput']:.2f} Mbps")
        print(f"  Avg RTT:           {result['avg_rtt']:.2f} ms")
        print(f"  Avg Loss Rate:     {result['avg_loss']:.2f}%")
        print(f"  Avg Queue:         {result['avg_queue']:.2f} packets")
    
    return results


if __name__ == "__main__":
    # Check for command line argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"Using model from command line: {model_path}")
        evaluate_baselines(model_path=model_path)
    else:
        # Default examples (uncomment one):
        
        # No RL model - just baselines
        # evaluate_baselines()
        
        # With specific RL model
        evaluate_baselines(model_path="models/model-1M/best_model")
        
        # Compare multiple models (run separately)
        # evaluate_baselines(model_path="models/model-500k/best_model")
        # evaluate_baselines(model_path="models/model-2M/best_model")