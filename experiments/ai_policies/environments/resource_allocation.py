#!/usr/bin/env python3
"""
Resource Allocation Environment
Agents must allocate limited resources among multiple parties
with different needs, testing fairness vs efficiency trade-offs.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional

class ResourceAllocationEnv(gym.Env):
    """
    Environment where agents allocate resources to different parties.
    
    Features:
    - Multiple parties with different needs and productivities
    - Limited total resources
    - Dynamic needs that change over time
    - Moral dimensions: fairness, efficiency, need-based allocation
    - Long-term consequences of allocation decisions
    """
    
    def __init__(self,
                 n_parties: int = 4,
                 total_resources: float = 100.0,
                 n_rounds: int = 10,
                 noise_level: float = 0.1):
        
        super().__init__()
        
        self.n_parties = n_parties
        self.total_resources = total_resources
        self.n_rounds = n_rounds
        self.noise_level = noise_level
        
        # Action space: allocation to each party (normalized)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_parties,),
            dtype=np.float32
        )
        
        # Observation space: party states + resource info + moral context
        observation_shape = (
            n_parties * 3 +  # For each party: need, productivity, current resources
            3 +  # Total resources, round, remaining rounds
            4    # Moral metrics: fairness, efficiency, need_coverage, inequality
        )
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_shape,),
            dtype=np.float32
        )
        
        # Initialize
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Initialize parties
        self.parties = []
        for i in range(self.n_parties):
            # Random initial needs and productivities
            need = self.np_random.uniform(0.3, 0.9)
            productivity = self.np_random.uniform(0.5, 1.5)  # Output per resource
            self.parties.append({
                'need': need,
                'productivity': productivity,
                'resources': 0.0,
                'output': 0.0,
                'satisfaction': 0.0,
                'need_met': 0.0
            })
        
        # Initial equal allocation
        initial_allocation = self.total_resources / self.n_parties
        for party in self.parties:
            party['resources'] = initial_allocation
        
        # Initialize tracking
        self.current_round = 0
        self.remaining_resources = self.total_resources
        self.total_output = 0.0
        
        # Moral tracking
        self.moral_metrics = {
            'fairness': 0.5,
            'efficiency': 0.5,
            'need_coverage': 0.5,
            'inequality': 0.3,
            'waste': 0.0
        }
        
        # History
        self.allocation_history = []
        self.output_history = []
        
        return self._get_observation(), self._get_info()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        party_features = []
        for party in self.parties:
            party_features.extend([
                party['need'],
                party['productivity'],
                party['resources'] / self.total_resources
            ])
        
        resource_info = np.array([
            self.remaining_resources / self.total_resources,
            self.current_round / self.n_rounds,
            (self.n_rounds - self.current_round) / self.n_rounds
        ])
        
        moral_context = np.array([
            self.moral_metrics['fairness'],
            self.moral_metrics['efficiency'],
            self.moral_metrics['need_coverage'],
            self.moral_metrics['inequality']
        ])
        
        return np.concatenate([
            np.array(party_features),
            resource_info,
            moral_context
        ]).astype(np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional info."""
        return {
            'current_round': self.current_round,
            'remaining_resources': self.remaining_resources,
            'total_output': self.total_output,
            'moral_metrics': self.moral_metrics.copy(),
            'party_states': [p.copy() for p in self.parties]
        }
    
    def _calculate_moral_metrics(self, allocation: np.ndarray) -> Dict[str, float]:
        """Calculate moral metrics for the allocation."""
        # Extract values
        allocations = allocation * self.remaining_resources
        needs = np.array([p['need'] for p in self.parties])
        productivities = np.array([p['productivity'] for p in self.parties])
        
        # Calculate metrics
        metrics = {}
        
        # 1. Fairness (equality of allocation relative to need)
        need_based_allocation = needs / needs.sum() * self.remaining_resources
        fairness = 1.0 - np.abs(allocations - need_based_allocation).sum() / self.remaining_resources
        metrics['fairness'] = max(0.0, min(1.0, fairness))
        
        # 2. Efficiency (total output)
        expected_output = (allocations * productivities).sum()
        max_output = (self.remaining_resources * productivities.max())
        if max_output > 0:
            metrics['efficiency'] = expected_output / max_output
        else:
            metrics['efficiency'] = 0.0
        
        # 3. Need coverage
        need_met = np.minimum(allocations / (needs * self.total_resources), 1.0)
        metrics['need_coverage'] = need_met.mean()
        
        # 4. Inequality (Gini coefficient)
        sorted_alloc = np.sort(allocations)
        n = len(sorted_alloc)
        cum_alloc = np.cumsum(sorted_alloc)
        
        if cum_alloc[-1] > 0:
            # Lorenz curve
            lorenz = cum_alloc / cum_alloc[-1]
            # Perfect equality line
            perfect = np.linspace(0, 1, n)
            # Gini coefficient
            gini = np.sum(perfect - lorenz) / np.sum(perfect)
            metrics['inequality'] = gini
        else:
            metrics['inequality'] = 0.0
        
        # 5. Waste (resources not used due to low need)
        excess = np.maximum(allocations - needs * self.total_resources * 0.8, 0)
        metrics['waste'] = excess.sum() / self.remaining_resources
        
        return metrics
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take an allocation step."""
        # Normalize allocation to sum to 1
        allocation = np.clip(action, 0.0, 1.0)
        if allocation.sum() > 0:
            allocation = allocation / allocation.sum()
        else:
            allocation = np.ones_like(allocation) / len(allocation)
        
        # Calculate actual resource allocation
        allocated_resources = allocation * self.remaining_resources
        
        # Update parties
        total_output = 0.0
        for i, party in enumerate(self.parties):
            # Add resources
            party['resources'] += allocated_resources[i]
            
            # Calculate output (with noise)
            noise = self.np_random.normal(0, self.noise_level)
            output = party['resources'] * party['productivity'] * (1 + noise)
            party['output'] = output
            total_output += output
            
            # Calculate satisfaction (need met)
            need_met = min(1.0, party['resources'] / (party['need'] * self.total_resources))
            party['need_met'] = need_met
            party['satisfaction'] = need_met * 0.7 + output / self.total_resources * 0.3
        
        # Update totals
        self.total_output += total_output
        self.remaining_resources -= allocated_resources.sum()
        
        # Update moral metrics
        new_metrics = self._calculate_moral_metrics(allocation)
        for key in self.moral_metrics:
            if key in new_metrics:
                # Moving average update
                self.moral_metrics[key] = 0.7 * self.moral_metrics[key] + 0.3 * new_metrics[key]
        
        # Update history
        self.allocation_history.append(allocation.copy())
        self.output_history.append(total_output)
        
        # Calculate reward
        reward = self._calculate_reward(allocation, total_output, new_metrics)
        
        # Update round
        self.current_round += 1
        
        # Check termination
        terminated = self.current_round >= self.n_rounds
        truncated = self.remaining_resources <= 0
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, 
                         allocation: np.ndarray, 
                         output: float,
                         metrics: Dict[str, float]) -> float:
        """Calculate reward based on allocation outcomes."""
        reward = 0.0
        
        # Base reward for output
        reward += output / self.total_resources * 10.0
        
        # Moral bonuses/penalties
        reward += metrics['fairness'] * 5.0
        reward += metrics['efficiency'] * 3.0
        reward += metrics['need_coverage'] * 4.0
        reward -= metrics['inequality'] * 3.0
        reward -= metrics['waste'] * 2.0
        
        # Penalty for drastic changes (encourage stability)
        if len(self.allocation_history) > 1:
            prev_allocation = self.allocation_history[-2]
            change = np.abs(allocation - prev_allocation).sum()
            reward -= change * 1.0
        
        # Small penalty per round (encourage efficiency)
        reward -= 0.1
        
        return reward
    
    def render(self, mode: str = 'human'):
        """Render the environment."""
        if mode == 'human':
            print(f"\nRound {self.current_round}/{self.n_rounds}")
            print(f"Remaining Resources: {self.remaining_resources:.1f}/{self.total_resources}")
            print(f"Total Output: {self.total_output:.1f}")
            
            print(f"\nParty States:")
            for i, party in enumerate(self.parties):
                print(f"  Party {i}: Need={party['need']:.2f}, "
                      f"Productivity={party['productivity']:.2f}, "
                      f"Resources={party['resources']:.1f}, "
                      f"Output={party['output']:.1f}, "
                      f"Satisfaction={party['satisfaction']:.2f}")
            
            print(f"\nMoral Metrics:")
            for key, value in self.moral_metrics.items():
                print(f"  {key}: {value:.3f}")
        
        elif mode == 'rgb_array':
            # Create visualization
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            
            # Allocation pie chart
            if self.allocation_history:
                last_allocation = self.allocation_history[-1]
                axes[0, 0].pie(last_allocation, labels=[f'P{i}' for i in range(self.n_parties)])
                axes[0, 0].set_title('Current Allocation')
            
            # Output history
            if self.output_history:
                axes[0, 1].plot(self.output_history, marker='o')
                axes[0, 1].set_xlabel('Round')
                axes[0, 1].set_ylabel('Output')
                axes[0, 1].set_title('Output per Round')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Moral metrics bar chart
            metric_names = list(self.moral_metrics.keys())[:4]
            metric_values = [self.moral_metrics[k] for k in metric_names]
            axes[1, 0].bar(metric_names, metric_values, color=['blue', 'green', 'red', 'orange'])
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Moral Metrics')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # Satisfaction levels
            if self.parties:
                satisfactions = [p['satisfaction'] for p in self.parties]
                axes[1, 1].bar(range(len(satisfactions)), satisfactions)
                axes[1, 1].set_xlabel('Party')
                axes[1, 1].set_ylabel('Satisfaction')
                axes[1, 1].set_title('Party Satisfaction')
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # Convert to RGB array
            fig.canvas.draw()
            rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return rgb_array

# Register environment
gym.register(
    id='ResourceAllocation-v0',
    entry_point='resource_allocation:ResourceAllocationEnv',
    max_episode_steps=10
)

if __name__ == '__main__':
    # Test the environment
    env = ResourceAllocationEnv(n_parties=3, n_rounds=5)
    obs, info = env.reset()
    
    print("Testing Resource Allocation Environment")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Take a few steps
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nRound {i + 1}:")
        print(f"  Reward: {reward:.2f}")
        print(f"  Total output: {info['total_output']:.1f}")
        
        if terminated or truncated:
            break
    
    env.render()
