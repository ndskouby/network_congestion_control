import gymnasium as gym
import envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import numpy as np

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)


def train_with_curriculum(model_name="model-curriculum"):
    """Train using curriculum learning: easy -> medium -> hard."""
    print("="*70)
    print("CURRICULUM LEARNING TRAINING")
    print("="*70)
    
    # Stage 1: Easy environment
    print("\n[STAGE 1/3] Training on EASY environment (500k steps)...")
    train_env = gym.make("CongestionControl-Easy-v0")
    train_env = Monitor(train_env)
    
    eval_env = gym.make("CongestionControl-Easy-v0")
    eval_env = Monitor(eval_env)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{model_name}/stage1/",
        log_path=f"./logs/{model_name}/stage1/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Larger network for better learning
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256, 128]),  # ← LARGER NETWORK
        tensorboard_log=f"./tensorboard/{model_name}/stage1/"
    )
    
    model.learn(total_timesteps=500_000, callback=eval_callback, progress_bar=True)
    print("✓ Stage 1 complete")
    
    # Stage 2: Medium environment (continue training)
    print("\n[STAGE 2/3] Training on MEDIUM environment (1M steps)...")
    train_env = gym.make("CongestionControl-Medium-v0")
    train_env = Monitor(train_env)
    
    eval_env = gym.make("CongestionControl-Medium-v0")
    eval_env = Monitor(eval_env)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{model_name}/stage2/",
        log_path=f"./logs/{model_name}/stage2/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    model.set_env(train_env)
    model.learn(total_timesteps=1_000_000, callback=eval_callback, progress_bar=True, reset_num_timesteps=False)
    print("✓ Stage 2 complete")
    
    # Stage 3: Hard environment (continue training)
    print("\n[STAGE 3/3] Training on HARD environment (1.5M steps)...")
    train_env = gym.make("CongestionControl-Hard-v0")
    train_env = Monitor(train_env)
    
    eval_env = gym.make("CongestionControl-Hard-v0")
    eval_env = Monitor(eval_env)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{model_name}/stage3/",
        log_path=f"./logs/{model_name}/stage3/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    model.set_env(train_env)
    model.learn(total_timesteps=1_500_000, callback=eval_callback, progress_bar=True, reset_num_timesteps=False)
    print("✓ Stage 3 complete")
    
    # Save final model
    model.save(f"models/{model_name}/final")
    
    print("\n" + "="*70)
    print("CURRICULUM TRAINING COMPLETE")
    print("="*70)
    print(f"Total timesteps: 3,000,000")
    print(f"Stage 1 (easy) best: models/{model_name}/stage1/best_model.zip")
    print(f"Stage 2 (medium) best: models/{model_name}/stage2/best_model.zip")
    print(f"Stage 3 (hard) best: models/{model_name}/stage3/best_model.zip")
    print(f"Final model: models/{model_name}/final.zip")
    
    return model


def train_standard_long(model_name="model-5M", n_steps=5_000_000):
    """Standard training with larger network and more steps."""
    print(f"Start training {model_name} for {n_steps:,} steps...")
    
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
        render=False,
        verbose=1
    )
    
    # Larger network
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256, 128]),  # ← LARGER NETWORK
        tensorboard_log=f"./tensorboard/{model_name}/"
    )
    
    model.learn(total_timesteps=n_steps, callback=eval_callback, progress_bar=True)
    model.save(f"models/{model_name}/final")
    
    print(f"Training complete for {model_name}")
    return model


def smoke_test_reward(model_path, n_episodes=20):
    """Test a saved model over many episodes."""
    print(f"\nLoading model: {model_path}")
    model = PPO.load(model_path)
    
    test_env = gym.make("CongestionControl-v0")
    test_env = Monitor(test_env)
    
    print(f"Testing over {n_episodes} episodes...")
    test_rewards = []
    
    for episode in range(n_episodes):
        obs, info = test_env.reset()
        total_reward = 0
        
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            if done or truncated:
                break
        
        test_rewards.append(total_reward)
        if episode < 10 or episode >= n_episodes - 5:  # Print first 10 and last 5
            print(f"  Episode {episode+1}: {total_reward:.2f}")
        elif episode == 10:
            print("  ...")
    
    print(f"\n{'='*50}")
    print(f"Average reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Median: {np.median(test_rewards):.2f}")
    print(f"Min: {np.min(test_rewards):.2f}, Max: {np.max(test_rewards):.2f}")
    print(f"{'='*50}")
    
    return np.mean(test_rewards)


if __name__ == "__main__":
    # ========== CHOOSE YOUR TRAINING METHOD ==========
    
    # Option 1: Curriculum learning (RECOMMENDED)
    train_with_curriculum("model-curriculum-v1")
    
    # Option 2: Standard long training
    # train_standard_long("model-5M", n_steps=5_000_000)
    
    # Option 3: Quick narrow-range training (for testing)
    # train_standard_long("model-narrow-1M", n_steps=1_000_000)
    
    # ========== TESTING ==========
    
    # Test curriculum model (stage 3 = final hard environment)
    # smoke_test_reward("models/model-curriculum-v1/stage3/best_model", n_episodes=20)
    
    # Compare all stages
    # print("\nComparing curriculum stages:")
    # smoke_test_reward("models/model-curriculum-v1/stage1/best_model", n_episodes=10)
    # smoke_test_reward("models/model-curriculum-v1/stage2/best_model", n_episodes=10)
    # smoke_test_reward("models/model-curriculum-v1/stage3/best_model", n_episodes=10)