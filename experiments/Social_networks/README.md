```markdown
# Prediction B3: Social Network Compressibility

**Hypothesis:** High-trust social networks have lower graph complexity (more compressible adjacency matrices) than low-trust networks.

## Protocol Summary
- **Design:** Comparative study of organizational networks
- **Data Sources:** 
  - Email/messaging logs (1 month minimum)
  - Meeting attendance records
  - Project collaboration data
- **Networks:** 
  - **High-trust**: Transparent organizations with shared values
  - **Low-trust**: Organizations with high conflict, low transparency
- **Compression:** Apply graph compression algorithms to adjacency matrices
- **Metric:** Graph compression ratio G_C = (compressed size)/(original size)

## Expected Result
G_C(high-trust) < G_C(low-trust)

## Quick Start

```bash
# Run with example data
python network_compression.py --example

# Run with your network data
python network_compression.py --data_dir ./network_data/ --output results/
```

### Files

- network_compression.py – Main analysis script
- social_data_analysis.py – Statistical analysis
- data/ – Templates and example data
- results/ – Output directory

### Data Collection

1. Obtain network data from:
   - Email servers (headers only, anonymized)
   - Messaging platforms (Slack, Teams, etc.)
   - Meeting calendars
   - Project management tools
2. Construct networks:
   - Nodes: Individuals
   - Edges: Weighted by interaction frequency
   - Time windows: Weekly slices for dynamics
3. Anonymize data:
   - Remove all personal identifiers
   - Use consistent anonymization mapping
   - Store mapping separately if needed for longitudinal analysis
4. Save networks as:
   - network_high_trust_week1.edgelist
   - network_low_trust_week1.edgelist
   - Or use GEXF/GraphML formats

### Analysis

Run full network analysis:

```bash
python social_data_analysis.py --high_trust high_trust_network.edgelist --low_trust low_trust_network.edgelist --output analysis_report/
```

### Output

- network_metrics.csv – Graph theory metrics
- compression_results.csv – Compression ratios
- statistical_tests.csv – Hypothesis tests
- network_visualization.png – Visualization
- analysis_report.json – Complete results

### Dependencies

- Python 3.9+
- networkx, matplotlib, seaborn
- scipy, statsmodels
- Optional: Gephi for advanced visualization

### Ethics Note

- Full anonymization required before analysis
- Institutional approval for organizational data
- Transparency about data use
- Data minimization – collect only what's needed
- Secure storage of sensitive organizational data

```

---

## **File: `experiments/social_networks/network_compression.py`**

```python
#!/usr/bin/env python3
"""
Social Network Compression Analysis
Prediction B3: High-trust networks are more compressible than low-trust networks.
"""

import networkx as nx
import pandas as pd
import numpy as np
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import json
from datetime import datetime

class NetworkCompressor:
    """Analyze compressibility of social networks."""
    
    def __init__(self):
        self.metrics = {}
    
    def load_network(self, filepath: Path, format: str = 'edgelist') -> nx.Graph:
        """Load network from file."""
        if format == 'edgelist':
            G = nx.read_edgelist(filepath, nodetype=int, data=[('weight', float)])
        elif format == 'adjlist':
            G = nx.read_adjlist(filepath)
        elif format == 'gexf':
            G = nx.read_gexf(filepath)
        elif format == 'graphml':
            G = nx.read_graphml(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Ensure undirected for social networks
        if G.is_directed():
            G = G.to_undirected()
        
        return G
    
    def calculate_graph_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """Calculate standard graph theory metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['n_nodes'] = G.number_of_nodes()
        metrics['n_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        
        # Degree statistics
        if metrics['n_nodes'] > 0:
            degrees = [d for n, d in G.degree()]
            metrics['avg_degree'] = np.mean(degrees)
            metrics['std_degree'] = np.std(degrees)
            metrics['max_degree'] = np.max(degrees)
            metrics['min_degree'] = np.min(degrees)
        
        # Connectivity
        metrics['connected_components'] = nx.number_connected_components(G)
        if nx.is_connected(G):
            metrics['diameter'] = nx.diameter(G)
            metrics['avg_path_length'] = nx.average_shortest_path_length(G)
        else:
            metrics['diameter'] = np.nan
            metrics['avg_path_length'] = np.nan
        
        # Clustering and transitivity
        metrics['avg_clustering'] = nx.average_clustering(G)
        metrics['transitivity'] = nx.transitivity(G)
        
        # Centrality measures (sample for large graphs)
        if metrics['n_nodes'] <= 1000:
            metrics['betweenness_centrality'] = np.mean(list(nx.betweenness_centrality(G).values()))
            metrics['closeness_centrality'] = np.mean(list(nx.closeness_centrality(G).values()))
        else:
            # Approximate for large graphs
            sample_nodes = list(G.nodes())[:100]
            metrics['betweenness_centrality'] = np.mean([nx.betweenness_centrality(G, k=100)[n] for n in sample_nodes])
            metrics['closeness_centrality'] = np.mean([nx.closeness_centrality(G)[n] for n in sample_nodes])
        
        # Community structure
        try:
            from community import community_louvain
            partition = community_louvain.best_partition(G)
            metrics['modularity'] = community_louvain.modularity(partition, G)
            metrics['n_communities'] = len(set(partition.values()))
        except:
            metrics['modularity'] = np.nan
            metrics['n_communities'] = np.nan
        
        return metrics
    
    def compress_network(self, G: nx.Graph, method: str = 'adjacency') -> Dict[str, float]:
        """Compress network using different representations."""
        n = G.number_of_nodes()
        
        if method == 'adjacency':
            # Dense adjacency matrix
            adj_matrix = nx.to_numpy_array(G)
            data = adj_matrix.astype(np.float32).tobytes()
        
        elif method == 'edgelist':
            # Edge list with weights
            edges = list(G.edges(data='weight', default=1.0))
            data = pickle.dumps(edges, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif method == 'sparse':
            # Sparse matrix representation
            from scipy import sparse
            adj_matrix = nx.to_scipy_sparse_array(G)
            data = pickle.dumps(adj_matrix, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif method == 'degree_seq':
            # Degree sequence
            degrees = [d for _, d in G.degree()]
            data = pickle.dumps(degrees, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif method == 'graph6':
            # Graph6 format (for unweighted graphs)
            if nx.is_weighted(G):
                # Convert to unweighted for graph6
                G_unweighted = nx.Graph(G)
                for u, v in G_unweighted.edges():
                    G_unweighted[u][v].clear()
                data = nx.to_graph6_bytes(G_unweighted)
            else:
                data = nx.to_graph6_bytes(G)
        
        else:
            raise ValueError(f"Unknown compression method: {method}")
        
        # Compress using gzip
        compressed = gzip.compress(data, compresslevel=9)
        
        return {
            'method': method,
            'original_size': len(data),
            'compressed_size': len(compressed),
            'compression_ratio': len(compressed) / len(data) if len(data) > 0 else 1.0,
            'bits_per_edge': (len(compressed) * 8) / G.number_of_edges() if G.number_of_edges() > 0 else np.nan,
            'bits_per_node': (len(compressed) * 8) / G.number_of_nodes() if G.number_of_nodes() > 0 else np.nan
        }
    
    def analyze_network(self, G: nx.Graph, network_name: str) -> Dict[str, any]:
        """Complete analysis of a single network."""
        results = {
            'network_name': network_name,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Calculate graph metrics
        results.update(self.calculate_graph_metrics(G))
        
        # Compression analysis with different methods
        compression_results = []
        for method in ['adjacency', 'edgelist', 'sparse', 'degree_seq', 'graph6']:
            try:
                compression = self.compress_network(G, method)
                compression['method'] = method
                compression_results.append(compression)
            except Exception as e:
                print(f"Warning: {method} compression failed: {e}")
        
        results['compression_analysis'] = compression_results
        
        # Calculate average compression ratio across methods
        valid_ratios = [c['compression_ratio'] for c in compression_results if not np.isnan(c['compression_ratio'])]
        if valid_ratios:
            results['avg_compression_ratio'] = np.mean(valid_ratios)
            results['min_compression_ratio'] = np.min(valid_ratios)
            results['max_compression_ratio'] = np.max(valid_ratios)
        
        return results
    
    def compare_networks(self, network1: Dict[str, any], network2: Dict[str, any]) -> Dict[str, any]:
        """Compare two networks."""
        comparison = {
            'network1': network1['network_name'],
            'network2': network2['network_name'],
            'comparison_timestamp': datetime.now().isoformat()
        }
        
        # Compare basic metrics
        for metric in ['n_nodes', 'n_edges', 'density', 'avg_clustering', 'transitivity']:
            if metric in network1 and metric in network2:
                val1 = network1[metric]
                val2 = network2[metric]
                if not np.isnan(val1) and not np.isnan(val2):
                    comparison[f'{metric}_ratio'] = val1 / val2
                    comparison[f'{metric}_diff'] = val1 - val2
        
        # Compare compression
        if 'avg_compression_ratio' in network1 and 'avg_compression_ratio' in network2:
            cr1 = network1['avg_compression_ratio']
            cr2 = network2['avg_compression_ratio']
            comparison['compression_ratio_difference'] = cr1 - cr2
            comparison['compression_ratio_ratio'] = cr1 / cr2
        
        # Hypothesis test result
        if 'avg_compression_ratio' in network1 and 'avg_compression_ratio' in network2:
            comparison['hypothesis_supported'] = network1['avg_compression_ratio'] < network2['avg_compression_ratio']
        
        return comparison

def create_example_networks() -> Tuple[nx.Graph, nx.Graph]:
    """Create example high-trust and low-trust networks."""
    
    # High-trust network: Small-world with high clustering
    print("Generating example high-trust network (small-world)...")
    n_nodes = 100
    k = 4  # Each node connected to k nearest neighbors
    p = 0.1  # Probability of rewiring
    
    high_trust = nx.watts_strogatz_graph(n_nodes, k, p)
    
    # Add weights based on clustering (higher weights for clustered connections)
    for u, v in high_trust.edges():
        common_neighbors = len(list(nx.common_neighbors(high_trust, u, v)))
        weight = 1.0 + common_neighbors * 0.5  # Stronger weights for clustered connections
        high_trust[u][v]['weight'] = weight
    
    # Low-trust network: Random with some hub structure
    print("Generating example low-trust network (scale-free with random edges)...")
    low_trust = nx.barabasi_albert_graph(n_nodes, 2)  # Scale-free base
    
    # Add random edges to increase disorder
    n_random_edges = 50
    for _ in range(n_random_edges):
        u = np.random.randint(0, n_nodes)
        v = np.random.randint(0, n_nodes)
        if u != v and not low_trust.has_edge(u, v):
            weight = np.random.uniform(0.1, 0.5)  # Weak, random weights
            low_trust.add_edge(u, v, weight=weight)
    
    # Add some negative edges (conflict) as negative weights
    n_negative_edges = 10
    for _ in range(n_negative_edges):
        u = np.random.randint(0, n_nodes)
        v = np.random.randint(0, n_nodes)
        if u != v:
            if low_trust.has_edge(u, v):
                low_trust[u][v]['weight'] = -abs(low_trust[u][v]['weight'])  # Make existing edge negative
            else:
                low_trust.add_edge(u, v, weight=np.random.uniform(-1.0, -0.1))
    
    return high_trust, low_trust

def save_network(G: nx.Graph, filename: Path, format: str = 'edgelist'):
    """Save network to file."""
    if format == 'edgelist':
        nx.write_edgelist(G, filename, data=['weight'])
    elif format == 'gexf':
        nx.write_gexf(G, filename)
    elif format == 'graphml':
        nx.write_graphml(G, filename)
    else:
        raise ValueError(f"Unknown format: {format}")

def main():
    parser = argparse.ArgumentParser(description='Analyze social network compressibility')
    parser.add_argument('--high_trust', type=str, default=None,
                       help='Path to high-trust network file')
    parser.add_argument('--low_trust', type=str, default=None,
                       help='Path to low-trust network file')
    parser.add_argument('--example', action='store_true',
                       help='Generate and analyze example networks')
    parser.add_argument('--output', type=str, default='network_results',
                       help='Output directory for results')
    parser.add_argument('--format', type=str, default='edgelist',
                       choices=['edgelist', 'gexf', 'graphml', 'adjlist'],
                       help='Input network format')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    compressor = NetworkCompressor()
    
    if args.example:
        print("="*60)
        print("SOCIAL NETWORK COMPRESSION EXAMPLE")
        print("="*60)
        
        # Generate example networks
        high_trust, low_trust = create_example_networks()
        
        # Save example networks
        save_network(high_trust, output_dir / 'example_high_trust.edgelist')
        save_network(low_trust, output_dir / 'example_low_trust.edgelist')
        print(f"✓ Example networks saved to {output_dir}")
        
        # Analyze networks
        print("\nAnalyzing networks...")
        ht_analysis = compressor.analyze_network(high_trust, 'example_high_trust')
        lt_analysis = compressor.analyze_network(low_trust, 'example_low_trust')
        
        # Compare
        comparison = compressor.compare_networks(ht_analysis, lt_analysis)
        
        # Save results
        results = {
            'high_trust_network': ht_analysis,
            'low_trust_network': lt_analysis,
            'comparison': comparison
        }
        
        with open(output_dir / 'example_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        print(f"\nHigh-trust network:")
        print(f"  Nodes: {ht_analysis['n_nodes']}, Edges: {ht_analysis['n_edges']}")
        print(f"  Density: {ht_analysis['density']:.3f}")
        print(f"  Avg. clustering: {ht_analysis['avg_clustering']:.3f}")
        print(f"  Avg. compression ratio: {ht_analysis.get('avg_compression_ratio', 'N/A'):.3f}")
        
        print(f"\nLow-trust network:")
        print(f"  Nodes: {lt_analysis['n_nodes']}, Edges: {lt_analysis['n_edges']}")
        print(f"  Density: {lt_analysis['density']:.3f}")
        print(f"  Avg. clustering: {lt_analysis['avg_clustering']:.3f}")
        print(f"  Avg. compression ratio: {lt_analysis.get('avg_compression_ratio', 'N/A'):.3f}")
        
        print(f"\nComparison:")
        if 'compression_ratio_difference' in comparison:
            diff = comparison['compression_ratio_difference']
            print(f"  Compression ratio difference (high - low): {diff:.4f}")
            print(f"  Hypothesis (high < low): {comparison.get('hypothesis_supported', 'N/A')}")
            if comparison.get('hypothesis_supported', False):
                print("  ✓ Hypothesis supported: High-trust network is more compressible")
            else:
                print("  ✗ Hypothesis not supported")
        
        print(f"\nDetailed results saved to: {output_dir / 'example_analysis.json'}")
    
    elif args.high_trust and args.low_trust:
        print("="*60)
        print("SOCIAL NETWORK COMPRESSION ANALYSIS")
        print("="*60)
        
        # Load networks
        print(f"\nLoading networks...")
        high_trust = compressor.load_network(Path(args.high_trust), args.format)
        low_trust = compressor.load_network(Path(args.low_trust), args.format)
        
        print(f"✓ High-trust network: {high_trust.number_of_nodes()} nodes, {high_trust.number_of_edges()} edges")
        print(f"✓ Low-trust network: {low_trust.number_of_nodes()} nodes, {low_trust.number_of_edges()} edges")
        
        # Analyze networks
        print("\nAnalyzing networks...")
        ht_analysis = compressor.analyze_network(high_trust, Path(args.high_trust).stem)
        lt_analysis = compressor.analyze_network(low_trust, Path(args.low_trust).stem)
        
        # Compare
        comparison = compressor.compare_networks(ht_analysis, lt_analysis)
        
        # Save results
        results = {
            'high_trust_network': ht_analysis,
            'low_trust_network': lt_analysis,
            'comparison': comparison
        }
        
        output_file = output_dir / 'network_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create CSV summaries
        # Graph metrics
        metrics_df = pd.DataFrame([ht_analysis, lt_analysis])
        metrics_df.to_csv(output_dir / 'graph_metrics.csv', index=False)
        
        # Compression results
        compression_data = []
        for analysis in [ht_analysis, lt_analysis]:
            for comp in analysis.get('compression_analysis', []):
                comp['network'] = analysis['network_name']
                compression_data.append(comp)
        
        if compression_data:
            compression_df = pd.DataFrame(compression_data)
            compression_df.to_csv(output_dir / 'compression_results.csv', index=False)
        
        print(f"\n✓ Analysis complete!")
        print(f"  Results saved to: {output_dir}")
        print(f"  Main results: {output_file}")
        
        # Quick hypothesis check
        if 'hypothesis_supported' in comparison:
            if comparison['hypothesis_supported']:
                print(f"  ✓ Hypothesis supported: High-trust network is more compressible")
            else:
                print(f"  ✗ Hypothesis not supported")
    
    else:
        print("Please provide --high_trust and --low_trust network files, or use --example")
        parser.print_help()

if __name__ == '__main__':
    main()
```

---

File: experiments/social_networks/social_data_analysis.py

```python
#!/usr/bin/env python3
"""
Statistical analysis for social network compression experiment.
Tests whether high-trust networks are more compressible than low-trust networks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

def load_analysis_results(filepath: str) -> Dict:
    """Load network analysis results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_compression_csv(filepath: str) -> pd.DataFrame:
    """Load compression results from CSV."""
    return pd.read_csv(filepath)

def descriptive_statistics(ht_analysis: Dict, lt_analysis: Dict) -> pd.DataFrame:
    """Calculate descriptive statistics for both networks."""
    metrics = ['n_nodes', 'n_edges', 'density', 'avg_clustering', 
               'transitivity', 'avg_compression_ratio']
    
    data = []
    for network_name, analysis in [('high_trust', ht_analysis), ('low_trust', lt_analysis)]:
        row = {'network': network_name}
        for metric in metrics:
            if metric in analysis:
                row[metric] = analysis[metric]
            else:
                row[metric] = np.nan
        data.append(row)
    
    return pd.DataFrame(data)

def compression_by_method(compression_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze compression ratios by method."""
    if compression_df.empty:
        return pd.DataFrame()
    
    results = compression_df.groupby(['network', 'method']).agg({
        'compression_ratio': ['mean', 'std', 'count'],
        'original_size': 'mean',
        'compressed_size': 'mean'
    }).round(4)
    
    results.columns = ['_'.join(col).strip() for col in results.columns.values]
    return results.reset_index()

def statistical_tests(ht_analysis: Dict, lt_analysis: Dict, 
                     compression_df: pd.DataFrame) -> Dict[str, any]:
    """Perform statistical tests comparing networks."""
    results = {}
    
    # Basic metrics comparison
    basic_metrics = ['density', 'avg_clustering', 'transitivity']
    for metric in basic_metrics:
        if metric in ht_analysis and metric in lt_analysis:
            ht_val = ht_analysis[metric]
            lt_val = lt_analysis[metric]
            if not np.isnan(ht_val) and not np.isnan(lt_val):
                # Simple difference
                results[f'{metric}_diff'] = ht_val - lt_val
                results[f'{metric}_ratio'] = ht_val / lt_val if lt_val != 0 else np.nan
    
    # Compression ratio comparison
    if 'avg_compression_ratio' in ht_analysis and 'avg_compression_ratio' in lt_analysis:
        ht_cr = ht_analysis['avg_compression_ratio']
        lt_cr = lt_analysis['avg_compression_ratio']
        
        if not np.isnan(ht_cr) and not np.isnan(lt_cr):
            results['compression_ratio_diff'] = ht_cr - lt_cr
            results['compression_ratio_ratio'] = ht_cr / lt_cr if lt_cr != 0 else np.nan
            
            # One-sample t-test (testing if difference is less than 0)
            # Since we have only two networks, we can't do a proper t-test
            # Instead, we'll use a permutation test if we had multiple networks
            results['hypothesis_supported'] = ht_cr < lt_cr
    
    # Compare compression across methods
    if not compression_df.empty:
        method_results = {}
        for method in compression_df['method'].unique():
            method_data = compression_df[compression_df['method'] == method]
            ht_method = method_data[method_data['network'].str.contains('high')]['compression_ratio'].values
            lt_method = method_data[method_data['network'].str.contains('low')]['compression_ratio'].values
            
            if len(ht_method) > 0 and len(lt_method) > 0:
                method_results[method] = {
                    'ht_mean': float(np.mean(ht_method)),
                    'lt_mean': float(np.mean(lt_method)),
                    'diff': float(np.mean(ht_method) - np.mean(lt_method)),
                    'ht_lt': bool(np.mean(ht_method) < np.mean(lt_method))
                }
        
        results['method_comparisons'] = method_results
    
    return results

def create_visualizations(ht_analysis: Dict, lt_analysis: Dict, 
                         compression_df: pd.DataFrame, output_dir: Path):
    """Create publication-ready visualizations."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Compression ratio comparison
    plt.figure(figsize=(12, 5))
    
    # Bar plot of average compression ratios
    plt.subplot(1, 2, 1)
    networks = ['High Trust', 'Low Trust']
    ratios = [ht_analysis.get('avg_compression_ratio', np.nan), 
              lt_analysis.get('avg_compression_ratio', np.nan)]
    
    bars = plt.bar(networks, ratios, color=['#2ecc71', '#e74c3c'], alpha=0.7)
    plt.title('Average Compression Ratio by Network Type', fontsize=14, fontweight='bold')
    plt.ylabel('Compression Ratio', fontsize=12)
    plt.xlabel('Network Type', fontsize=12)
    
    # Add values on bars
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        if not np.isnan(ratio):
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                     f'{ratio:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add hypothesis annotation
    if ht_analysis.get('avg_compression_ratio', np.nan) < lt_analysis.get('avg_compression_ratio', np.nan):
        plt.text(0.5, 0.95, '✓ Hypothesis supported', 
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                ha='center', va='top')
    else:
        plt.text(0.5, 0.95, '✗ Hypothesis not supported', 
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
                ha='center', va='top')
    
    # 2. Compression by method
    if not compression_df.empty:
        plt.subplot(1, 2, 2)
        
        # Pivot data for grouped bar plot
        pivot_df = compression_df.pivot_table(
            index='method', 
            columns='network', 
            values='compression_ratio',
            aggfunc='mean'
        )
        
        # Rename columns for display
        pivot_df.columns = ['High Trust' if 'high' in str(col).lower() else 'Low Trust' 
                           for col in pivot_df.columns]
        
        ax = pivot_df.plot(kind='bar', width=0.8, alpha=0.7)
        plt.title('Compression Ratio by Method', fontsize=14, fontweight='bold')
        plt.ylabel('Compression Ratio', fontsize=12)
        plt.xlabel('Compression Method', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Network Type')
        
        # Add grid
        plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'compression_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'compression_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    # 3. Network metrics comparison (if available)
    metrics_to_plot = ['density', 'avg_clustering', 'transitivity']
    available_metrics = [m for m in metrics_to_plot 
                        if m in ht_analysis and m in lt_analysis]
    
    if available_metrics:
        plt.figure(figsize=(10, 6))
        
        n_metrics = len(available_metrics)
        x = np.arange(n_metrics)
        width = 0.35
        
        ht_values = [ht_analysis[m] for m in available_metrics]
        lt_values = [lt_analysis[m] for m in available_metrics]
        
        plt.bar(x - width/2, ht_values, width, label='High Trust', 
                color='#2ecc71', alpha=0.7)
        plt.bar(x + width/2, lt_values, width, label='Low Trust', 
                color='#e74c3c', alpha=0.7)
        
        plt.xlabel('Graph Metric', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Network Metrics Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, [m.replace('_', ' ').title() for m in available_metrics])
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'network_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Visualizations saved to {output_dir}")

def generate_report(analysis_file: str, output_dir: str) -> Dict[str, any]:
    """
    Generate complete analysis report.
    
    Args:
        analysis_file: Path to network_analysis.json
        output_dir: Directory to save report
        
    Returns:
        Dictionary with all analysis results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("SOCIAL NETWORK COMPRESSION ANALYSIS REPORT")
    print("="*70)
    
    # Load data
    print(f"\nLoading analysis results from: {analysis_file}")
    analysis_data = load_analysis_results(analysis_file)
    
    ht_analysis = analysis_data['high_trust_network']
    lt_analysis = analysis_data['low_trust_network']
    comparison = analysis_data.get('comparison', {})
    
    # Load compression CSV if available
    compression_csv = Path(analysis_file).parent / 'compression_results.csv'
    if compression_csv.exists():
        compression_df = load_compression_csv(compression_csv)
    else:
        compression_df = pd.DataFrame()
    
    # 1. Network information
    print(f"\n1. NETWORK INFORMATION")
    print("-"*40)
    print(f"High-trust network: {ht_analysis.get('network_name', 'N/A')}")
    print(f"  Nodes: {ht_analysis.get('n_nodes', 'N/A')}")
    print(f"  Edges: {ht_analysis.get('n_edges', 'N/A')}")
    print(f"  Density: {ht_analysis.get('density', 'N/A'):.4f}")
    
    print(f"\nLow-trust network: {lt_analysis.get('network_name', 'N/A')}")
    print(f"  Nodes: {lt_analysis.get('n_nodes', 'N/A')}")
    print(f"  Edges: {lt_analysis.get('n_edges', 'N/A')}")
    print(f"  Density: {lt_analysis.get('density', 'N/A'):.4f}")
    
    # 2. Descriptive statistics
    print(f"\n2. DESCRIPTIVE STATISTICS")
    print("-"*40)
    desc_stats = descriptive_statistics(ht_analysis, lt_analysis)
    print(desc_stats.to_string(index=False))
    
    # Save descriptive stats
    desc_stats.to_csv(output_path / 'descriptive_statistics.csv')
    
    # 3. Compression analysis
    print(f"\n3. COMPRESSION ANALYSIS")
    print("-"*40)
    
    if 'avg_compression_ratio' in ht_analysis and 'avg_compression_ratio' in lt_analysis:
        ht_cr = ht_analysis['avg_compression_ratio']
        lt_cr = lt_analysis['avg_compression_ratio']
        
        print(f"High-trust avg compression ratio: {ht_cr:.4f}")
        print(f"Low-trust avg compression ratio: {lt_cr:.4f}")
        print(f"Difference (high - low): {ht_cr - lt_cr:.4f}")
        print(f"Ratio (high/low): {ht_cr/lt_cr:.4f}" if lt_cr != 0 else "Ratio: N/A (division by zero)")
        
        # Hypothesis check
        print(f"\nHypothesis: High-trust network is more compressible (lower C_R)")
        print(f"  Condition: C_R(high) < C_R(low)")
        print(f"  Result: {ht_cr:.4f} < {lt_cr:.4f} = {ht_cr < lt_cr}")
        
        if ht_cr < lt_cr:
            print("  ✓ PRIMARY HYPOTHESIS SUPPORTED")
            print(f"    High-trust network compresses more efficiently")
        else:
            print("  ✗ PRIMARY HYPOTHESIS NOT SUPPORTED")
            print(f"    High-trust network does not compress more efficiently")
    else:
        print("Compression ratios not available in analysis data")
    
    # 4. Compression by method
    if not compression_df.empty:
        print(f"\n4. COMPRESSION BY METHOD")
        print("-"*40)
        
        method_stats = compression_by_method(compression_df)
        print(method_stats.to_string(index=False))
        
        # Save method stats
        method_stats.to_csv(output_path / 'compression_by_method.csv')
        
        # Count methods supporting hypothesis
        supporting_methods = 0
        total_methods = 0
        
        for method in compression_df['method'].unique():
            method_data = compression_df[compression_df['method'] == method]
            ht_vals = method_data[method_data['network'].str.contains('high', case=False)]['compression_ratio'].values
            lt_vals = method_data[method_data['network'].str.contains('low', case=False)]['compression_ratio'].values
            
            if len(ht_vals) > 0 and len(lt_vals) > 0:
                total_methods += 1
                if np.mean(ht_vals) < np.mean(lt_vals):
                    supporting_methods += 1
        
        print(f"\nMethods supporting hypothesis: {supporting_methods}/{total_methods}")
        if total_methods > 0:
            print(f"Support rate: {supporting_methods/total_methods*100:.1f}%")
    
    # 5. Statistical tests
    print(f"\n5. STATISTICAL COMPARISON")
    print("-"*40)
    
    stats_results = statistical_tests(ht_analysis, lt_analysis, compression_df)
    
    for key, value in stats_results.items():
        if key != 'method_comparisons':
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    # Save statistical results
    with open(output_path / 'statistical_results.json', 'w') as f:
        json.dump(stats_results, f, indent=2, default=str)
    
    # 6. Create visualizations
    print(f"\n6. VISUALIZATIONS")
    print("-"*40)
    create_visualizations(ht_analysis, lt_analysis, compression_df, output_path)
    print(f"Plots saved to: {output_path}")
    
    # 7. Save complete report
    report_data = {
        'network_info': {
            'high_trust': {
                'name': ht_analysis.get('network_name', 'N/A'),
                'n_nodes': ht_analysis.get('n_nodes', 'N/A'),
                'n_edges': ht_analysis.get('n_edges', 'N/A')
            },
            'low_trust': {
                'name': lt_analysis.get('network_name', 'N/A'),
                'n_nodes': lt_analysis.get('n_nodes', 'N/A'),
                'n_edges': lt_analysis.get('n_edges', 'N/A')
            }
        },
        'descriptive_statistics': desc_stats.to_dict('records'),
        'compression_analysis': {
            'high_trust_avg_cr': ht_analysis.get('avg_compression_ratio', np.nan),
            'low_trust_avg_cr': lt_analysis.get('avg_compression_ratio', np.nan),
            'difference': ht_analysis.get('avg_compression_ratio', np.nan) - 
                         lt_analysis.get('avg_compression_ratio', np.nan) 
                         if 'avg_compression_ratio' in ht_analysis and 'avg_compression_ratio' in lt_analysis else np.nan
        },
        'statistical_results': stats_results,
        'analysis_date': pd.Timestamp.now().isoformat()
    }
    
    with open(output_path / 'analysis_report.json', 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print(f"\nComplete report saved to: {output_path / 'analysis_report.json'}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return report_data

def main():
    parser = argparse.ArgumentParser(
        description='Analyze social network compression results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python social_data_analysis.py --analysis network_analysis.json --output analysis/
  python social_data_analysis.py --compression compression_results.csv --output analysis/
        """
    )
    
    parser.add_argument('--analysis', type=str, 
                       help='Path to network_analysis.json file')
    parser.add_argument('--compression', type=str,
                       help='Path to compression_results.csv file')
    parser.add_argument('--output', type=str, default='social_analysis_results',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    if args.analysis:
        report = generate_report(args.analysis, args.output)
        
        # Print summary conclusion
        print("\n" + "="*70)
        print("SUMMARY CONCLUSION")
        print("="*70)
        
        ht_cr = report['compression_analysis'].get('high_trust_avg_cr', np.nan)
        lt_cr = report['compression_analysis'].get('low_trust_avg_cr', np.nan)
        
        if not np.isnan(ht_cr) and not np.isnan(lt_cr):
            if ht_cr < lt_cr:
                print("✓ SUPPORT FOR PREDICTION B3")
                print("  High-trust social networks are more compressible than low-trust networks.")
                print(f"  C_R(high) = {ht_cr:.3f} < C_R(low) = {lt_cr:.3f}")
            else:
                print("✗ NO SUPPORT FOR PREDICTION B3")
                print("  High-trust social networks are not more compressible than low-trust networks.")
                print(f"  C_R(high) = {ht_cr:.3f} ≥ C_R(low) = {lt_cr:.3f}")
        else:
            print("⚠ INCONCLUSIVE: Missing compression ratio data")
        
        print("\nSee detailed results in output directory.")
    
    elif args.compression:
        print("Individual compression file analysis not yet implemented")
        print("Please provide the full analysis JSON file instead")
    
    else:
        print("Please provide --analysis or --compression file")
        parser.print_help()

if __name__ == '__main__':
    main()
```
