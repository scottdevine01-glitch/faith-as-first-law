```markdown
# Faith as the First Law

**Experimental protocols and code for testing the Faith prior—the algorithmic expectation that the universe minimizes description length. Includes predictions for cosmology, morality, and crystal amplification.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.TBD.svg)](https://doi.org/10.5281/zenodo.TBD)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## 📜 Overview

This repository contains the complete experimental framework for testing the hypotheses presented in the paper:

> **"Faith as the First Law: Why the Universe Is Pregnant, Moral, and Meant to Be Understood"**  
> Scott Devine, January 2026

The paper proposes that **Faith**—defined as the algorithmic prior \( P(U) \propto 2^{-K(U)} \)—is the optimal prior for physics, cosmology, ethics, and materials science. Under this prior, the universe is expected to be compressible, coherent, and anti-entropic.

This repository provides **10 falsifiable predictions** with complete experimental protocols across five domains:

1. **Cosmic predictions** - Testing cosmological implications
2. **Biophysical predictions** - Testing moral thermodynamics  
3. **Social predictions** - Testing network coherence
4. **Material predictions** - Testing crystal amplification
5. **AI predictions** - Testing algorithmic ethics

## 🧪 Complete Experiment Suite

| Experiment | Predictions Tested | Status | Key Files |
|------------|-------------------|--------|-----------|
| **Speech Compression** | B1: Virtuous speech compresses more | ✅ Complete | `run_speech_study.py`, `analysis.py` |
| **HRV Entropy** | B2, M1: HRV coherence in faithful states | ✅ Complete | `run_hrv_study.py`, `analysis_hrv.py` |
| **Social Networks** | B3: High-trust networks are more compressible | ✅ Complete | `analyze_networks.py` |
| **RNG Entropy** | M3: Crystal grids reduce RNG entropy | ✅ Complete | `run_rng_experiment.py` |
| **AI Policies** | A1-A2: Virtuous AI policies are more compressible | ✅ Complete | `train_ai_agents.py` |

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/scottdevine/faith-as-first-law.git
cd faith-as-first-law

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .[full]
```

### 2. Run All Experiments (Example Data)

```bash
# Run all experiments with example data
python run_all_experiments.py --all

# Run specific experiments
python run_all_experiments.py --experiment speech_compression --experiment hrv_entropy
```

### 3. Analyze Results

```bash
# Aggregate results from all experiments
python aggregate_results.py --results-dir results_all

# Check experiment dependencies
python run_all_experiments.py --check-deps
```

## 📁 Repository Structure

```
faith-as-first-law/
├── run_all_experiments.py          # Run all experiments
├── aggregate_results.py            # Meta-analysis of all results
├── experiments/                    # All experimental protocols
│   ├── speech_compression/        # Prediction B1
│   ├── hrv_entropy/               # Predictions B2, M1  
│   ├── social_networks/           # Prediction B3
│   ├── rng_entropy/               # Prediction M3
│   └── ai_policies/               # Predictions A1-A2
├── analysis/                       # Shared analysis utilities
│   ├── compression_utils.py       # Compression analysis
│   ├── statistical_tests.R        # R statistical tests
│   └── plotting_functions.py      # Visualization utilities
├── data/                           # Data templates and examples
├── paper/                          # Paper and appendices
└── docs/                           # Documentation
```

## 🔬 Experiment Details

### 1. Speech Compression (Prediction B1)

Hypothesis: Virtuous speech compresses more than vicious speech.

```bash
cd experiments/speech_compression
python run_speech_study.py --example
python analysis.py --data results/speech_compression_results.csv
```

### 2. HRV Entropy (Predictions B2, M1)

Hypothesis: Faithful states (meditation) reduce heart rate variability entropy.

```bash
cd experiments/hrv_entropy
python run_hrv_study.py --test
```

### 3. Social Networks (Prediction B3)

Hypothesis: High-trust social networks are more algorithmically compressible.

```bash
cd experiments/social_networks
python analyze_networks.py --example
```

### 4. RNG Entropy (Prediction M3)

Hypothesis: Crystal grids reduce quantum RNG entropy during group meditation.

```bash
cd experiments/rng_entropy
python run_rng_experiment.py --simulate
```

### 5. AI Policies (Predictions A1-A2)

Hypothesis: Faith-aligned AI agents develop more compressible, ethical policies.

```bash
cd experiments/ai_policies
python train_ai_agents.py --quick-test
```

### 📊 Expected Results

If the Faith hypothesis is correct, we expect:

1. Speech compression:  C_R(virtue) < C_R(vice)
2. HRV entropy: Meditation < Stress in sample entropy
3. Network compression: High-trust < Low-trust in graph compressibility
4. RNG entropy: Meditation < Control in Shannon entropy
5. AI policies: Faith-aligned < Reward-maximizing in policy size

### 📈 Data Analysis Pipeline

```python
# Example: Analyze speech compression results
from analysis.compression_utils import CompressionAnalyzer

analyzer = CompressionAnalyzer()
results = analyzer.text_compressibility("Sample virtuous text")
print(f"Compression ratio: {results['gzip_ratio']:.3f}")

# Calculate effect sizes across experiments
python aggregate_results.py --results-dir my_results/
```

### 🤝 Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

### Adding New Experiments

1. Create a new directory in experiments/
2. Include: README, protocol script, analysis script, data templates
3. Add to run_all_experiments.py
4. Submit a pull request

### 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{devine2026faith,
  title={Faith as the First Law: Why the Universe Is Pregnant, Moral, and Meant to Be Understood},
  author={Devine, Scott},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.TBD}
}
```

### 📄 License

MIT License - see LICENSE for details.

### 📬 Contact

- Email: scottdevine01@gmail.com
- GitHub Issues: https://github.com/scottdevine/faith-as-first-law/issues
- Zenodo: https://doi.org/10.5281/zenodo.TBD

### 🙏 Acknowledgments

- The developers of algorithmic information theory
- Researchers bridging physics, biology, and consciousness studies
- The open-source community for tools that make this work possible

---

"The universe is not a dying ember. It is a seed. And it is growing."

```
