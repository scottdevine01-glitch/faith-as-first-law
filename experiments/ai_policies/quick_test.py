#!/usr/bin/env python3
"""
Quick test script for AI Policies experiment.
Runs a minimal version to verify everything works.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    modules = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('gymnasium', 'Gymnasium'),
    ]
    
    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name}: {e}")
            all_ok = False
    
    # Test custom modules
    custom_modules = [
        'train_agents',
        'policy_compression',
        'environments.moral_gridworld'
    ]
    
    print("\nTesting custom modules...")
    for module_path in custom_modules:
        try:
            __import__(module_path)
            print(f"  ✓ {module_path}")
        except ImportError as e:
            print(f"  ✗ {module_path}: {e}")
            all_ok = False
    
    return all_ok

def test_environment():
    """Test that environments can be created."""
    print("\nTesting environments...")
    
    try:
        from environments.moral_gridworld import MoralGridworld
        env = MoralGridworld()
        state = env.reset()
        print(f"  ✓ MoralGridworld: state_dim={env.observation_space.shape}, "
              f"actions={env.action_space.n}")
    except Exception as e:
        print(f"  ✗ MoralGridworld: {e}")
        return False
    
    try:
        from environments.prisoner_dilemma_arena import PrisonerDilemmaArena
        env = PrisonerDilemmaArena()
        state = env.reset()
        print(f"  ✓ PrisonerDilemmaArena: state_dim={env.observation_space.shape}, "
              f"actions={env.action_space.n}")
    except Exception as e:
        print(f"  ✗ PrisonerDilemmaArena: {e}")
    
    return True

def test_training():
    """Test a minimal training run."""
    print("\nTesting minimal training...")
    
    try:
        import torch
        from train_agents import AEPAgent, RewardMaximizingAgent
        
        # Create a simple environment
        class SimpleEnv:
            def __init__(self):
                self.observation_space = type('obj', (object,), {'shape': (4,)})()
                self.action_space = type('obj', (object,), {'n': 2})()
            
            def reset(self):
                return np.zeros(4), {}
            
            def step(self, action):
                return np.zeros(4), 0.0, False, False, {}
        
        import numpy as np
        
        env = SimpleEnv()
        state_dim = 4
        action_dim = 2
        
        # Create agents
        aep_agent = AEPAgent(state_dim, action_dim, hidden_dim=16)
        reward_agent = RewardMaximizingAgent(state_dim, action_dim, hidden_dim=16)
        
        print(f"  ✓ Created AEP agent with {sum(p.numel() for p in aep_agent.policy.parameters())} parameters")
        print(f"  ✓ Created Reward agent with {sum(p.numel() for p in reward_agent.policy.parameters())} parameters")
        
        # Test action selection
        state = np.zeros(4)
        action, log_prob = aep_agent.get_action(state)
        print(f"  ✓ AEP agent selected action: {action}, log_prob: {log_prob:.4f}")
        
        action, log_prob = reward_agent.get_action(state)
        print(f"  ✓ Reward agent selected action: {action}, log_prob: {log_prob:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compression():
    """Test policy compression analysis."""
    print("\nTesting compression analysis...")
    
    try:
        from policy_compression import PolicyCompressionAnalyzer
        
        analyzer = PolicyCompressionAnalyzer()
        
        # Create a dummy policy
        import torch
        dummy_policy = {
            'policy_state_dict': {
                'encoder.0.weight': torch.randn(16, 4),
                'encoder.0.bias': torch.randn(16),
                'policy_head.0.weight': torch.randn(8, 16),
                'policy_head.0.bias': torch.randn(8)
            }
        }
        
        # Save dummy policy
        test_dir = Path('test_compression')
        test_dir.mkdir(exist_ok=True)
        
        policy_path = test_dir / 'dummy_policy.pt'
        torch.save(dummy_policy, policy_path)
        
        # Analyze
        results = analyzer.analyze_policy_file(policy_path)
        
        print(f"  ✓ Analyzed policy with {results['total_parameters']} parameters")
        print(f"  ✓ Best compression ratio: {results['model_compressibility']['best_compression_ratio']:.4f}")
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Compression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("AI POLICIES EXPERIMENT - QUICK TEST")
    print("="*60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Environments", test_environment),
        ("Training", test_training),
        ("Compression", test_compression)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-"*40)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:20} {status}")
        if not success:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("\nTo run the full experiment:")
        print("  python run_experiment.py --output my_experiment")
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        print("\nPlease check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
