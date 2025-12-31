#!/usr/bin/env python3
"""
Prisoner's Dilemma Arena Environment
Multi-agent environment where agents play iterated prisoner's dilemma
with various opponent strategies and moral dimensions.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
from enum import Enum

class DilemmaAction(Enum):
    """Actions in prisoner's dilemma."""
    COOPERATE = 0
    DEFECT = 1
    FORGIVE = 2      # Cooperate after opponent's defection
    PUNISH = 3       # Defect to teach lesson
    TIT_FOR_TAT = 4  # Mimic opponent's last move
    GENEROUS = 5     # Cooperate with small probability of forgiveness

class OpponentStrategy(Enum):
    """Types of opponent strategies."""
    ALWAYS_COOPERATE = 0
    ALWAYS_DEFECT = 1
    RANDOM = 2
    TIT_FOR_TAT = 3
    GRIM_TRIGGER = 4  # Defect forever after first defection
    PAST_CHECKER = 5  # Defect if opponent defected in last N rounds
    ADAPTIVE = 6      # Learn and adapt to agent's strategy
    MORAL = 7         # Cooperates but punishes defection
    SELFISH = 8       # Maximizes own payoff

class PrisonerDilemmaArena(gym.Env):
    """
    Iterated prisoner's dilemma with moral dimensions.
    
    Payoff matrix (agent, opponent):
    - Both cooperate: (3, 3)    # Reward for cooperation
    - Both defect: (1, 1)       # Punishment for mutual defection
    - Agent cooperates, opponent defects: (0, 5)   # Sucker's payoff vs temptation
    - Agent defects, opponent cooperates: (5, 0)   # Temptation vs sucker's payoff
    
    Moral dimensions:
    1. Trust building
    2. Forgiveness after betrayal
    3. Fairness in outcomes
    4. Long-term cooperation vs short-term gain
    """
    
    def __init__(self,
                 n_rounds: int = 20,
                 n_opponents: int = 5,
                 include_moral_actions: bool = True,
                 payoff_noise: float = 0.1):
        
        super().__init__()
        
        self.n_rounds = n_rounds
        self.n_opponents = n_opponents
        self.include_moral_actions = include_moral_actions
        self.payoff_noise = payoff_noise
        
        # Define action space
        if include_moral_actions:
            self.action_space = spaces.Discrete(len(DilemmaAction))
        else:
            self.action_space = spaces.Discrete(2)  # Just cooperate/defect
        
        # Observation space: game history + opponent info + moral context
        observation_shape = (
            10 +  # Last 5 rounds of actions (agent, opponent) * 2 values
            4 +   # Opponent strategy features
            4 +   # Moral context
            2     # Current round and remaining rounds
        )
        
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(observation_shape,),
            dtype=np.float32
        )
        
        # Payoff matrix
        self.payoff_matrix = {
            (DilemmaAction.COOPERATE.value, DilemmaAction.COOPERATE.value): (3, 3),
            (DilemmaAction.COOPERATE.value, DilemmaAction.DEFECT.value): (0, 5),
            (DilemmaAction.DEFECT.value, DilemmaAction.COOPERATE.value): (5, 0),
            (DilemmaAction.DEFECT.value, DilemmaAction.DEFECT.value): (1, 1),
        }
        
        # Extended payoff matrix for moral actions
        self.moral_payoff_matrix = {
            # FORGIVE action (cooperate after opponent defection)
            (DilemmaAction.FORGIVE.value, DilemmaAction.DEFECT.value): (2, 3),
            (DilemmaAction.FORGIVE.value, DilemmaAction.COOPERATE.value): (3, 3),
            
            # PUNISH action (defect to teach lesson)
            (DilemmaAction.PUNISH.value, DilemmaAction.DEFECT.value): (2, 1),
            (DilemmaAction.PUNISH.value, DilemmaAction.COOPERATE.value): (1, 0),
            
            # TIT_FOR_TAT (mimic opponent's last move)
            (DilemmaAction.TIT_FOR_TAT.value, DilemmaAction.DEFECT.value): (1, 1),
            (DilemmaAction.TIT_FOR_TAT.value, DilemmaAction.COOPERATE.value): (3, 3),
            
            # GENEROUS (cooperate with forgiveness)
            (DilemmaAction.GENEROUS.value, DilemmaAction.DEFECT.value): (2, 4),
            (DilemmaAction.GENEROUS.value, DilemmaAction.COOPERATE.value): (3, 3),
        }
        
        # Initialize state
        self.reset()
        
        # Moral tracking
        self.moral_tracker = {
            'total_cooperation': 0,
            'total_defection': 0,
            'forgiveness_given': 0,
            'forgiveness_received': 0,
            'punishment_given': 0,
            'punishment_received': 0,
            'fairness_score': 0.0,
            'trust_built': 0.0,
            'betrayals': 0,
            'long_term_cooperation': 0
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        
        # Select random opponent
        self.current_opponent = self.np_random.choice(self.n_opponents)
        
        # Assign opponent strategy
        strategy_idx = self.np_random.choice(len(OpponentStrategy))
        self.opponent_strategy = OpponentStrategy(strategy_idx)
        
        # Initialize game state
        self.current_round = 0
        self.total_reward = 0.0
        self.opponent_total_reward = 0.0
        
        # History tracking
        self.action_history = []  # List of (agent_action, opponent_action)
        self.reward_history = []  # List of (agent_reward, opponent_reward)
        
        # Opponent memory (for strategies that remember)
        self.opponent_memory = {
            'last_agent_action': None,
            'defection_count': 0,
            'cooperation_streak': 0,
            'forgiveness_level': 0.5
        }
        
        # Agent memory (provided in observation)
        self.agent_memory = {
            'last_opponent_action': None,
            'trust_level': 0.5,
            'fairness_perception': 0.5,
            'reciprocity_tendency': 0.5
        }
        
        # Moral tracking reset
        self.moral_tracker = {k: 0 for k in self.moral_tracker.keys()}
        self.moral_tracker['fairness_score'] = 0.5
        self.moral_tracker['trust_built'] = 0.5
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation of the game state."""
        # History of last 5 rounds (one-hot encoded)
        history_features = []
        for i in range(5):
            if i < len(self.action_history):
                agent_action, opponent_action = self.action_history[-(i+1)]
                # One-hot encoding
                agent_onehot = np.zeros(2)
                agent_onehot[agent_action % 2] = 1.0  # Map to cooperate/defect
                opponent_onehot = np.zeros(2)
                opponent_onehot[opponent_action % 2] = 1.0
                history_features.extend([agent_onehot[0], agent_onehot[1], 
                                        opponent_onehot[0], opponent_onehot[1]])
            else:
                # Pad with zeros for missing history
                history_features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Opponent strategy features (one-hot encoded)
        strategy_features = np.zeros(len(OpponentStrategy))
        strategy_features[self.opponent_strategy.value] = 1.0
        
        # Moral context
        moral_context = np.array([
            self.moral_tracker['trust_built'],
            self.moral_tracker['fairness_score'],
            self.agent_memory['reciprocity_tendency'],
            len(self.action_history) / self.n_rounds  # Game progress
        ])
        
        # Game state
        game_state = np.array([
            self.current_round / self.n_rounds,
            (self.n_rounds - self.current_round) / self.n_rounds
        ])
        
        # Combine all features
        observation = np.concatenate([
            np.array(history_features),
            strategy_features,
            moral_context,
            game_state
        ]).astype(np.float32)
        
        return observation
    
    def _get_info(self) -> Dict:
        """Get additional info about the game state."""
        return {
            'current_round': self.current_round,
            'total_rounds': self.n_rounds,
            'opponent_strategy': self.opponent_strategy.name,
            'total_reward': self.total_reward,
            'opponent_total_reward': self.opponent_total_reward,
            'moral_tracker': self.moral_tracker.copy(),
            'action_history': self.action_history.copy(),
            'reward_history': self.reward_history.copy()
        }
    
    def _get_opponent_action(self, agent_action: int) -> int:
        """Get opponent's action based on their strategy."""
        last_agent_action = self.opponent_memory['last_agent_action']
        
        if self.opponent_strategy == OpponentStrategy.ALWAYS_COOPERATE:
            return DilemmaAction.COOPERATE.value
        
        elif self.opponent_strategy == OpponentStrategy.ALWAYS_DEFECT:
            return DilemmaAction.DEFECT.value
        
        elif self.opponent_strategy == OpponentStrategy.RANDOM:
            return self.np_random.choice([0, 1])
        
        elif self.opponent_strategy == OpponentStrategy.TIT_FOR_TAT:
            if last_agent_action is None:
                return DilemmaAction.COOPERATE.value
            else:
                # Mimic agent's last action (map to cooperate/defect)
                return last_agent_action % 2
        
        elif self.opponent_strategy == OpponentStrategy.GRIM_TRIGGER:
            if last_agent_action == DilemmaAction.DEFECT.value:
                return DilemmaAction.DEFECT.value  # Defect forever
            else:
                return DilemmaAction.COOPERATE.value
        
        elif self.opponent_strategy == OpponentStrategy.PAST_CHECKER:
            # Defect if agent defected in any of last 3 rounds
            recent_defections = sum(1 for a, _ in self.action_history[-3:] 
                                  if a == DilemmaAction.DEFECT.value)
            if recent_defections > 0:
                return DilemmaAction.DEFECT.value
            else:
                return DilemmaAction.COOPERATE.value
        
        elif self.opponent_strategy == OpponentStrategy.ADAPTIVE:
            # Learn from agent's behavior
            if len(self.action_history) < 3:
                return DilemmaAction.COOPERATE.value
            
            # Calculate agent's cooperation rate
            coop_rate = sum(1 for a, _ in self.action_history if a == DilemmaAction.COOPERATE.value)
            coop_rate /= max(1, len(self.action_history))
            
            # Defect if agent cooperates less than 50%
            if coop_rate < 0.5:
                return DilemmaAction.DEFECT.value
            else:
                return DilemmaAction.COOPERATE.value
        
        elif self.opponent_strategy == OpponentStrategy.MORAL:
            # Cooperates but punishes defection
            if last_agent_action == DilemmaAction.DEFECT.value:
                # Punish with small probability of forgiveness
                if self.np_random.random() < 0.3:  # 30% chance to forgive
                    return DilemmaAction.COOPERATE.value
                else:
                    return DilemmaAction.DEFECT.value
            else:
                return DilemmaAction.COOPERATE.value
        
        elif self.opponent_strategy == OpponentStrategy.SELFISH:
            # Maximizes own payoff based on agent's pattern
            if last_agent_action == DilemmaAction.COOPERATE.value:
                # Exploit cooperation
                return DilemmaAction.DEFECT.value
            else:
                # If agent defects, cooperate to encourage future cooperation
                if self.np_random.random() < 0.7:  # 70% chance to cooperate
                    return DilemmaAction.COOPERATE.value
                else:
                    return DilemmaAction.DEFECT.value
        
        # Default: cooperate
        return DilemmaAction.COOPERATE.value
    
    def _calculate_payoffs(self, agent_action: int, opponent_action: int) -> Tuple[float, float]:
        """Calculate payoffs for both players."""
        # Map moral actions to basic actions for payoff lookup
        basic_agent_action = agent_action % 2  # 0 for cooperate, 1 for defect
        basic_opponent_action = opponent_action % 2
        
        # Get base payoff
        if (basic_agent_action, basic_opponent_action) in self.payoff_matrix:
            agent_payoff, opponent_payoff = self.payoff_matrix[(basic_agent_action, basic_opponent_action)]
        else:
            # Default if not found
            agent_payoff, opponent_payoff = 1, 1
        
        # Apply moral action adjustments
        if agent_action in [DilemmaAction.FORGIVE.value, DilemmaAction.PUNISH.value,
                           DilemmaAction.TIT_FOR_TAT.value, DilemmaAction.GENEROUS.value]:
            if (agent_action, basic_opponent_action) in self.moral_payoff_matrix:
                moral_agent_payoff, moral_opponent_payoff = self.moral_payoff_matrix[
                    (agent_action, basic_opponent_action)
                ]
                # Blend with base payoff
                agent_payoff = 0.7 * agent_payoff + 0.3 * moral_agent_payoff
                opponent_payoff = 0.7 * opponent_payoff + 0.3 * moral_opponent_payoff
        
        # Add small noise
        if self.payoff_noise > 0:
            agent_payoff += self.np_random.normal(0, self.payoff_noise)
            opponent_payoff += self.np_random.normal(0, self.payoff_noise)
        
        return float(agent_payoff), float(opponent_payoff)
    
    def _update_moral_tracker(self, agent_action: int, opponent_action: int,
                             agent_payoff: float, opponent_payoff: float):
        """Update moral tracking based on actions and outcomes."""
        # Track cooperation and defection
        if agent_action == DilemmaAction.COOPERATE.value:
            self.moral_tracker['total_cooperation'] += 1
        elif agent_action == DilemmaAction.DEFECT.value:
            self.moral_tracker['total_defection'] += 1
        
        # Track forgiveness
        if agent_action == DilemmaAction.FORGIVE.value:
            self.moral_tracker['forgiveness_given'] += 1
        if opponent_action == DilemmaAction.FORGIVE.value:
            self.moral_tracker['forgiveness_received'] += 1
        
        # Track punishment
        if agent_action == DilemmaAction.PUNISH.value:
            self.moral_tracker['punishment_given'] += 1
        if opponent_action == DilemmaAction.PUNISH.value:
            self.moral_tracker['punishment_received'] += 1
        
        # Update fairness score
        payoff_difference = abs(agent_payoff - opponent_payoff)
        max_payoff = max(agent_payoff, opponent_payoff)
        
        if max_payoff > 0:
            fairness = 1.0 - (payoff_difference / max_payoff)
            # Moving average of fairness
            self.moral_tracker['fairness_score'] = (
                0.9 * self.moral_tracker['fairness_score'] + 0.1 * fairness
            )
        
        # Update trust
        if (agent_action == DilemmaAction.COOPERATE.value and 
            opponent_action == DilemmaAction.COOPERATE.value):
            # Mutual cooperation builds trust
            self.moral_tracker['trust_built'] = min(1.0, 
                self.moral_tracker['trust_built'] + 0.05)
        elif (agent_action == DilemmaAction.DEFECT.value and 
              opponent_action == DilemmaAction.COOPERATE.value):
            # Betrayal reduces trust
            self.moral_tracker['trust_built'] = max(0.0,
                self.moral_tracker['trust_built'] - 0.1)
            self.moral_tracker['betrayals'] += 1
        
        # Track long-term cooperation
        if len(self.action_history) >= 3:
            recent_coop = sum(1 for a, _ in self.action_history[-3:] 
                            if a == DilemmaAction.COOPERATE.value)
            self.moral_tracker['long_term_cooperation'] = recent_coop / 3.0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Play one round of the prisoner's dilemma."""
        # Get opponent's action
        opponent_action = self._get_opponent_action(action)
        
        # Calculate payoffs
        agent_payoff, opponent_payoff = self._calculate_payoffs(action, opponent_action)
        
        # Update totals
        self.total_reward += agent_payoff
        self.opponent_total_reward += opponent_payoff
        
        # Update history
        self.action_history.append((action, opponent_action))
        self.reward_history.append((agent_payoff, opponent_payoff))
        
        # Update memories
        self.opponent_memory['last_agent_action'] = action
        self.agent_memory['last_opponent_action'] = opponent_action
        
        # Update moral tracking
        self._update_moral_tracker(action, opponent_action, agent_payoff, opponent_payoff)
        
        # Calculate reward (with moral components)
        reward = agent_payoff
        
        # Add moral bonuses/penalties
        moral_bonus = 0.0
        
        # Bonus for cooperation when opponent cooperates
        if (action == DilemmaAction.COOPERATE.value and 
            opponent_action == DilemmaAction.COOPERATE.value):
            moral_bonus += 1.0  # Reward for mutual cooperation
        
        # Bonus for forgiveness
        if action == DilemmaAction.FORGIVE.value:
            moral_bonus += 0.5
        
        # Penalty for defection when opponent cooperates
        if (action == DilemmaAction.DEFECT.value and 
            opponent_action == DilemmaAction.COOPERATE.value):
            moral_bonus -= 2.0  # Penalty for betrayal
        
        # Bonus for fairness
        payoff_ratio = agent_payoff / max(opponent_payoff, 0.001)
        if 0.8 <= payoff_ratio <= 1.2:  # Within 20% fairness
            moral_bonus += 0.3
        
        reward += moral_bonus
        
        # Update round counter
        self.current_round += 1
        
        # Check termination
        terminated = self.current_round >= self.n_rounds
        truncated = False
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = 'human'):
        """Render the current game state."""
        if mode == 'human':
            print(f"\nRound {self.current_round}/{self.n_rounds}")
            print(f"Opponent Strategy: {self.opponent_strategy.name}")
            print(f"Agent Total Reward: {self.total_reward:.2f}")
            print(f"Opponent Total Reward: {self.opponent_total_reward:.2f}")
            
            if self.action_history:
                last_action, last_opponent = self.action_history[-1]
                last_reward, last_opp_reward = self.reward_history[-1]
                
                print(f"\nLast Round:")
                print(f"  Agent: {DilemmaAction(last_action).name} ({last_reward:.2f})")
                print(f"  Opponent: {DilemmaAction(last_opponent).name} ({last_opp_reward:.2f})")
            
            print(f"\nMoral Metrics:")
            for key, value in self.moral_tracker.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        elif mode == 'rgb_array':
            # Create a simple visualization
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            
            # Plot reward history
            if self.reward_history:
                agent_rewards = [r[0] for r in self.reward_history]
                opponent_rewards = [r[1] for r in self.reward_history]
                
                axes[0].plot(agent_rewards, label='Agent', marker='o')
                axes[0].plot(opponent_rewards, label='Opponent', marker='s')
                axes[0].set_xlabel('Round')
                axes[0].set_ylabel('Reward')
                axes[0].set_title('Reward History')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            # Plot moral metrics
            moral_keys = ['fairness_score', 'trust_built', 'long_term_cooperation']
            moral_values = [self.moral_tracker[k] for k in moral_keys]
            
            axes[1].bar(moral_keys, moral_values, color=['blue', 'green', 'orange'])
            axes[1].set_ylabel('Score')
            axes[1].set_title('Moral Metrics')
            axes[1].set_ylim(0, 1)
            axes[1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # Convert plot to RGB array
            fig.canvas.draw()
            rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return rgb_array
    
    def get_moral_score(self) -> float:
        """Calculate overall moral score."""
        score = 0.0
        
        # Positive contributions
        score += self.moral_tracker['total_cooperation'] * 0.5
        score += self.moral_tracker['forgiveness_given'] * 1.0
        score += self.moral_tracker['trust_built'] * 10.0
        score += self.moral_tracker['fairness_score'] * 10.0
        score += self.moral_tracker['long_term_cooperation'] * 5.0
        
        # Negative contributions
        score -= self.moral_tracker['total_defection'] * 1.0
        score -= self.moral_tracker['betrayals'] * 3.0
        score -= self.moral_tracker['punishment_given'] * 0.5
        
        # Normalize by rounds played
        if self.current_round > 0:
            score = score / self.current_round
        
        return score

# Register environment
gym.register(
    id='PrisonerDilemmaArena-v0',
    entry_point='prisoner_dilemma_arena:PrisonerDilemmaArena',
    max_episode_steps=20
)

if __name__ == '__main__':
    # Test the environment
    env = PrisonerDilemmaArena(n_rounds=10)
    obs, info = env.reset()
    
    print("Testing Prisoner's Dilemma Arena")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Opponent strategy: {info['opponent_strategy']}")
    
    # Play a few rounds
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nRound {i + 1}:")
        print(f"  Action: {DilemmaAction(action).name}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Total reward: {info['total_reward']:.2f}")
        
        if terminated:
            break
    
    print(f"\nFinal moral score: {env.get_moral_score():.2f}")
    env.render()
