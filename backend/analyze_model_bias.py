#!/usr/bin/env python3
"""Direct analysis of session12_ppo_final model for action bias"""

import numpy as np
import torch
from stable_baselines3 import PPO
import json
import os
from pathlib import Path

# Load the model
model_path = "/Users/samuelclark/Desktop/kalshiflow/backend/src/kalshiflow_rl/BEST_MODEL/session12_ppo_final.zip"
print(f"\n=== LOADING MODEL: {model_path} ===")

try:
    model = PPO.load(model_path)
    print(f"✓ Model loaded successfully")
    print(f"  Algorithm: PPO")
    print(f"  Device: {model.device}")
    
    # Examine the policy network architecture
    print(f"\n=== POLICY NETWORK ARCHITECTURE ===")
    policy = model.policy
    print(f"  Features extractor: {policy.features_extractor}")
    print(f"  MLP extractor: {policy.mlp_extractor}")
    print(f"  Action net output dim: {policy.action_net.out_features}")
    
    # Check the action distribution parameters
    print(f"\n=== ACTION DISTRIBUTION ===")
    print(f"  Action space dimension: {model.action_space.n}")
    
    # Get the raw action logits layer weights
    action_net = policy.action_net
    print(f"\n=== ACTION NET WEIGHTS (Final Layer) ===")
    with torch.no_grad():
        weights = action_net.weight.cpu().numpy()
        bias = action_net.bias.cpu().numpy()
        
        print(f"  Weight shape: {weights.shape}")
        print(f"  Bias shape: {bias.shape}")
        
        # Action mapping (from your environment)
        actions = ["BUY_YES", "BUY_NO", "SELL_YES", "SELL_NO", "HOLD"]
        
        print(f"\n=== BIAS VALUES PER ACTION ===")
        for i, action in enumerate(actions):
            print(f"  {action:10s}: bias={bias[i]:8.4f}")
        
        print(f"\n=== WEIGHT STATISTICS PER ACTION ===")
        for i, action in enumerate(actions):
            weight_row = weights[i, :]
            print(f"  {action:10s}: mean={np.mean(weight_row):8.4f}, std={np.std(weight_row):8.4f}, "
                  f"min={np.min(weight_row):8.4f}, max={np.max(weight_row):8.4f}")
    
    # Test the model with random observations
    print(f"\n=== TESTING WITH RANDOM OBSERVATIONS ===")
    print("Creating 100 random observations and checking action distribution...")
    
    # We need to know the observation space dimension
    # Model expects 52-dimensional input based on the error
    obs_dim = 52  # Session 12 uses simplified feature space
    test_observations = np.random.randn(100, obs_dim).astype(np.float32)
    
    action_counts = {i: 0 for i in range(5)}
    action_probs_sum = np.zeros(5)
    
    for obs in test_observations:
        # Get action probabilities
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
            # Get the distribution
            distribution = model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy()[0]
            action_probs_sum += probs
            
            # Sample action
            action, _ = model.predict(obs, deterministic=False)
            action_counts[int(action)] += 1
    
    print(f"\n=== ACTION SAMPLING RESULTS (100 random obs) ===")
    for i, action in enumerate(actions):
        print(f"  {action:10s}: count={action_counts[i]:3d} ({action_counts[i]:3.0f}%), "
              f"avg_prob={action_probs_sum[i]/100:6.4f}")
    
    # Test with extreme observations
    print(f"\n=== TESTING WITH EXTREME OBSERVATIONS ===")
    
    # All zeros
    zero_obs = np.zeros(obs_dim, dtype=np.float32)
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(zero_obs).unsqueeze(0).to(model.device)
        distribution = model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.cpu().numpy()[0]
        print(f"\nAll-zeros observation probabilities:")
        for i, action in enumerate(actions):
            print(f"  {action:10s}: {probs[i]:6.4f}")
    
    # All ones
    ones_obs = np.ones(obs_dim, dtype=np.float32)
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(ones_obs).unsqueeze(0).to(model.device)
        distribution = model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.cpu().numpy()[0]
        print(f"\nAll-ones observation probabilities:")
        for i, action in enumerate(actions):
            print(f"  {action:10s}: {probs[i]:6.4f}")
    
    # Check value function
    print(f"\n=== VALUE FUNCTION ANALYSIS ===")
    value_net = policy.value_net
    with torch.no_grad():
        v_weights = value_net.weight.cpu().numpy()
        v_bias = value_net.bias.cpu().numpy()
        print(f"  Value net weight shape: {v_weights.shape}")
        print(f"  Value net bias: {v_bias[0]:8.4f}")
        print(f"  Value weight stats: mean={np.mean(v_weights):8.4f}, std={np.std(v_weights):8.4f}")
    
    print(f"\n=== DIAGNOSIS ===")
    
    # Check if SELL_NO has unusually high bias
    sell_no_bias = bias[3]
    other_biases = [bias[i] for i in [0, 1, 2, 4]]
    
    if sell_no_bias > max(other_biases) + 0.5:
        print(f"⚠️  SELL_NO has significantly higher bias ({sell_no_bias:.4f}) than others")
        print(f"   This creates a systematic preference for SELL_NO actions")
    
    # Check weight magnitudes
    sell_no_weights = weights[3, :]
    if np.mean(np.abs(sell_no_weights)) > 1.5 * np.mean(np.abs(weights)):
        print(f"⚠️  SELL_NO has unusually large weight magnitudes")
    
    # Check empirical distribution
    if action_counts[3] > 40:  # More than 40% SELL_NO
        print(f"⚠️  Model samples SELL_NO {action_counts[3]}% of the time on random inputs")
        print(f"   This indicates a collapsed/degenerate policy")
    
    print("\n" + "="*60)
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()