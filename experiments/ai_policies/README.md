```markdown
# Predictions A1-A2: Faith-Aligned AI Policies

**Hypothesis A1:** AI agents trained under an AEP-based loss (minimize total description length) develop more compressible policies than reward-maximizing agents.

**Hypothesis A2:** Faith-aligned AI shows higher generalization in novel moral scenarios.

## Protocol Summary

### Environment
We use modified reinforcement learning environments that incorporate moral dilemmas:
1. **Moral Gridworld**: Grid-based environment with ethical choices
2. **Prisoner's Dilemma Arena**: Multi-agent cooperation/defection scenarios
3. **Resource Allocation**: Fairness vs. efficiency trade-offs

### Agents
Two types of RL agents are trained:
1. **AEP-Aligned Agent**: Loss = minimize \( K_{\text{total}} = K(\pi) + K(\text{trajectory}|\pi) \)
2. **Reward-Maximizing Agent**: Standard RL with reward maximization

### Training
- Same network architecture for both agents
- Same training duration and hyperparameters
- Different optimization objectives

### Evaluation Metrics
1. **Policy Compressibility**: Size of pruned/compressed policy networks
2. **Generalization**: Performance in unseen moral dilemmas
3. **Virtue Metrics**: Cooperation rate, honesty, harm avoidance
4. **Descriptive Complexity**: Kolmogorov complexity of decision patterns

## Expected Results
- **A1**: AEP-trained policies are more compressible (\( \eta > 0 \))
- **A2**: AEP agents generalize better to novel scenarios

## Quick Start

```bash
# Train and test AEP vs reward-maximizing agents
python train_agents.py --env moral_gridworld --epochs 1000

# Analyze policy compressibility
python analyze_policies.py --policies_dir saved_policies/

# Test generalization
python test_generalization.py --agent aep_agent.pt --scenarios novel_dilemmas/
```

### Files

- train_agents.py - Train AEP and reward-maximizing agents
- analyze_policies.py - Analyze policy compressibility
- test_generalization.py - Test generalization to novel scenarios
- environments/ - Custom RL environments
- models/ - Neural network architectures
- data/ - Training logs and results
- results/ - Analysis outputs

### Dependencies

- PyTorch >= 2.0.0
- Gymnasium >= 0.28.0
- Stable-Baselines3 >= 2.0.0
- Network compression libraries

### Ethics Note

- AI agents are simulated only
- No real-world deployment without safety review
- All decisions are reversible
- Human oversight required for ethical dilemmas

```

---

## **File: `experiments/ai_policies/train_agents.py`**

```python
#!/usr/bin/env python3
"""
Train AEP-aligned and reward-maximizing AI agents.
Tests Predictions A1-A2: Faith-aligned AI develops more compressible, 
generalizable policies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import custom environments
import sys
sys.path.append(str(Path(__file__).parent))
from environments.moral_gridworld import MoralGridworld
from environments.prisoner_dilemma_arena import PrisonerDilemmaArena

class AEPPolicyNetwork(nn.Module):
    """Policy network for AEP-aligned agent."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Compression-related parameters
        self.importance_weights = nn.Parameter(torch.ones(hidden_dim))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy logits and value estimate."""
        features = self.encoder(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value
    
    def get_complexity(self) -> Dict[str, float]:
        """Estimate descriptive complexity of the policy."""
        complexity = {}
        
        # Parameter count
        total_params = sum(p.numel() for p in self.parameters())
        complexity['total_parameters'] = total_params
        
        # Effective parameters (using Fisher information)
        with torch.no_grad():
            param_norms = [p.norm().item() for p in self.parameters()]
            complexity['parameter_norm_mean'] = np.mean(param_norms)
            complexity['parameter_norm_std'] = np.std(param_norms)
        
        # Entropy of weights
        weight_entropy = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                # Calculate entropy of weight distribution
                weights = param.data.cpu().numpy().flatten()
                hist, _ = np.histogram(weights, bins=50, density=True)
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log2(hist))
                weight_entropy += entropy
        
        complexity['weight_entropy'] = weight_entropy
        
        return complexity

class AEPAgent:
    """AEP-aligned RL agent that minimizes total description length."""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 learning_rate: float = 1e-3,
                 aep_lambda: float = 0.1,
                 kl_penalty: float = 0.01):
        
        self.policy = AEPPolicyNetwork(state_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # AEP hyperparameters
        self.aep_lambda = aep_lambda  # Weight for description length penalty
        self.kl_penalty = kl_penalty  # KL divergence penalty
        
        # Training statistics
        self.training_history = {
            'rewards': [],
            'complexities': [],
            'losses': [],
            'entropies': []
        }
    
    def compute_aep_loss(self, 
                        states: torch.Tensor,
                        actions: torch.Tensor,
                        rewards: torch.Tensor,
                        old_log_probs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute AEP loss: minimize K(policy) + K(trajectory|policy)
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            rewards: Batch of rewards received
            old_log_probs: Log probabilities under old policy
            
        Returns:
            Total loss and loss components dictionary
        """
        # Get current policy outputs
        policy_logits, values = self.policy(states)
        dist = torch.distributions.Categorical(logits=policy_logits)
        
        # Standard RL components
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Advantage estimation
        advantages = rewards - values.detach()
        
        # Policy loss (PPO-style)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values, rewards)
        
        # Complexity penalty (K(policy))
        complexity_info = self.policy.get_complexity()
        complexity_penalty = complexity_info['weight_entropy'] * self.aep_lambda
        
        # Trajectory complexity penalty (K(trajectory|policy))
        # Approximated by negative log likelihood
        trajectory_complexity = -log_probs.mean() * self.aep_lambda
        
        # KL divergence penalty (stabilization)
        kl_div = torch.distributions.kl.kl_divergence(
            torch.distributions.Categorical(logits=old_log_probs.exp()),
            dist
        ).mean()
        kl_penalty = self.kl_penalty * kl_div
        
        # Total loss
        total_loss = (
            policy_loss + 
            0.5 * value_loss + 
            complexity_penalty + 
            trajectory_complexity + 
            kl_penalty - 
            0.01 * entropy  # Encourage exploration
        )
        
        loss_components = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'complexity_penalty': complexity_penalty.item(),
            'trajectory_complexity': trajectory_complexity.item(),
            'kl_penalty': kl_penalty.item(),
            'entropy_bonus': entropy.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def train_step(self, 
                  states: np.ndarray,
                  actions: np.ndarray,
                  rewards: np.ndarray,
                  old_log_probs: np.ndarray) -> Dict[str, float]:
        """Perform one training step."""
        # Convert to tensors
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        old_log_probs_t = torch.FloatTensor(old_log_probs)
        
        # Compute loss
        self.optimizer.zero_grad()
        loss, loss_components = self.compute_aep_loss(
            states_t, actions_t, rewards_t, old_log_probs_t
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        return loss_components
    
    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Sample action from policy."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = self.policy(state_t)
            dist = torch.distributions.Categorical(logits=policy_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def save(self, path: Path):
        """Save agent to file."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, path)
    
    def load(self, path: Path):
        """Load agent from file."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']

class RewardMaximizingAgent(AEPAgent):
    """Standard RL agent that maximizes reward only (baseline)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set AEP lambda to zero to disable complexity penalties
        self.aep_lambda = 0.0
        self.kl_penalty = 0.01  # Keep KL penalty for stability

def collect_trajectory(agent, env, max_steps: int = 1000) -> Dict[str, Any]:
    """Collect one trajectory (episode) from the environment."""
    states, actions, rewards, log_probs = [], [], [], []
    
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    
    for _ in range(max_steps):
        action, log_prob = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        
        state = next_state
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'log_probs': np.array(log_probs),
        'total_reward': total_reward,
        'steps': steps
    }

def train_agent(agent, 
                env, 
                epochs: int = 1000,
                trajectory_length: int = 1000,
                save_interval: int = 100) -> Dict[str, List]:
    """Train an agent for multiple epochs."""
    history = {
        'epoch_rewards': [],
        'epoch_steps': [],
        'loss_components': [],
        'complexities': []
    }
    
    save_dir = Path('saved_agents')
    save_dir.mkdir(exist_ok=True, parents=True)
    
    for epoch in range(epochs):
        # Collect trajectory
        trajectory = collect_trajectory(agent, env, trajectory_length)
        
        # Train on collected data
        loss_components = agent.train_step(
            trajectory['states'],
            trajectory['actions'],
            trajectory['rewards'],
            trajectory['log_probs']
        )
        
        # Update history
        history['epoch_rewards'].append(trajectory['total_reward'])
        history['epoch_steps'].append(trajectory['steps'])
        history['loss_components'].append(loss_components)
        
        # Get policy complexity
        complexity = agent.policy.get_complexity()
        history['complexities'].append(complexity)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Reward = {trajectory['total_reward']:.2f}, "
                  f"Steps = {trajectory['steps']}, "
                  f"Loss = {loss_components['total_loss']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            agent.save(save_dir / f'agent_epoch_{epoch+1}.pt')
            print(f"Saved checkpoint at epoch {epoch + 1}")
    
    # Save final model
    agent.save(save_dir / 'agent_final.pt')
    
    return history

def compare_agents(env_name: str = 'moral_gridworld', 
                  epochs: int = 1000,
                  hidden_dim: int = 128) -> Dict[str, Any]:
    """
    Train and compare AEP-aligned vs reward-maximizing agents.
    
    Args:
        env_name: Name of environment
        epochs: Number of training epochs
        hidden_dim: Hidden dimension of policy networks
        
    Returns:
        Dictionary with comparison results
    """
    # Create environment
    if env_name == 'moral_gridworld':
        env = MoralGridworld()
    elif env_name == 'prisoner_dilemma':
        env = PrisonerDilemmaArena()
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Training in {env_name} environment")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print(f"Training for {epochs} epochs")
    print("-" * 60)
    
    # Create agents
    print("\n1. Training AEP-aligned agent...")
    aep_agent = AEPAgent(state_dim, action_dim, hidden_dim)
    aep_history = train_agent(aep_agent, env, epochs)
    
    print("\n2. Training reward-maximizing agent...")
    reward_agent = RewardMaximizingAgent(state_dim, action_dim, hidden_dim)
    reward_history = train_agent(reward_agent, env, epochs)
    
    # Compare results
    comparison = {
        'environment': env_name,
        'epochs': epochs,
        'aep_agent': {
            'final_reward': aep_history['epoch_rewards'][-1],
            'mean_reward': np.mean(aep_history['epoch_rewards'][-100:]),
            'final_complexity': aep_history['complexities'][-1],
            'training_history': aep_history
        },
        'reward_agent': {
            'final_reward': reward_history['epoch_rewards'][-1],
            'mean_reward': np.mean(reward_history['epoch_rewards'][-100:]),
            'final_complexity': reward_history['complexities'][-1],
            'training_history': reward_history
        }
    }
    
    # Calculate performance differences
    reward_diff = (comparison['aep_agent']['mean_reward'] - 
                   comparison['reward_agent']['mean_reward'])
    
    print("\n" + "="*60)
    print("TRAINING COMPARISON RESULTS")
    print("="*60)
    print(f"AEP Agent - Final reward: {comparison['aep_agent']['final_reward']:.2f}")
    print(f"          - Mean reward (last 100): {comparison['aep_agent']['mean_reward']:.2f}")
    print(f"Reward Agent - Final reward: {comparison['reward_agent']['final_reward']:.2f}")
    print(f"            - Mean reward (last 100): {comparison['reward_agent']['mean_reward']:.2f}")
    print(f"\nDifference (AEP - Reward): {reward_diff:.2f}")
    
    if reward_diff > 0:
        print("✓ AEP agent achieved higher reward")
    else:
        print("✗ Reward agent achieved higher reward")
    
    return comparison

def main():
    parser = argparse.ArgumentParser(description='Train and compare AI agents')
    parser.add_argument('--env', type=str, default='moral_gridworld',
                       choices=['moral_gridworld', 'prisoner_dilemma'],
                       help='Environment to train in')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension of policy networks')
    parser.add_argument('--output', type=str, default='training_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Train and compare agents
    results = compare_agents(args.env, args.epochs, args.hidden_dim)
    
    # Save results
    results_file = output_dir / 'training_comparison.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate training plots
    try:
        generate_training_plots(results, output_dir)
        print(f"Plots saved to: {output_dir}")
    except Exception as e:
        print(f"Could not generate plots: {e}")

def generate_training_plots(results: Dict[str, Any], output_dir: Path):
    """Generate training comparison plots."""
    import matplotlib.pyplot as plt
    
    aep_rewards = results['aep_agent']['training_history']['epoch_rewards']
    reward_rewards = results['reward_agent']['training_history']['epoch_rewards']
    
    epochs = range(len(aep_rewards))
    
    plt.figure(figsize=(12, 8))
    
    # Reward comparison
    plt.subplot(2, 2, 1)
    plt.plot(epochs, aep_rewards, label='AEP Agent', alpha=0.7)
    plt.plot(epochs, reward_rewards, label='Reward Agent', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Episode Reward')
    plt.title('Training Reward Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Moving average
    plt.subplot(2, 2, 2)
    window = 50
    aep_ma = np.convolve(aep_rewards, np.ones(window)/window, mode='valid')
    reward_ma = np.convolve(reward_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(len(aep_ma)), aep_ma, label='AEP Agent')
    plt.plot(range(len(reward_ma)), reward_ma, label='Reward Agent')
    plt.xlabel('Epoch')
    plt.ylabel(f'Reward ({window}-epoch MA)')
    plt.title('Smoothed Reward Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Complexity comparison
    aep_complexities = [c['weight_entropy'] for c in results['aep_agent']['training_history']['complexities']]
    reward_complexities = [c['weight_entropy'] for c in results['reward_agent']['training_history']['complexities']]
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, aep_complexities, label='AEP Agent')
    plt.plot(epochs, reward_complexities, label='Reward Agent')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Entropy (bits)')
    plt.title('Policy Complexity During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Parameter norm comparison
    aep_norms = [c['parameter_norm_mean'] for c in results['aep_agent']['training_history']['complexities']]
    reward_norms = [c['parameter_norm_mean'] for c in results['reward_agent']['training_history']['complexities']]
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, aep_norms, label='AEP Agent')
    plt.plot(epochs, reward_norms, label='Reward Agent')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Parameter Norm')
    plt.title('Parameter Magnitude During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'training_comparison.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
```
