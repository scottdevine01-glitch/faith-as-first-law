```markdown
# Faith as the First Law

**Experimental protocols and code for testing the Faith prior—the algorithmic expectation that the universe minimizes description length. Includes predictions for cosmology, morality, and crystal amplification.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.TBD.svg)](https://doi.org/10.5281/zenodo.TBD)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📜 Overview

This repository contains the complete experimental framework for testing the hypotheses presented in the paper:

> **"Faith as the First Law: Why the Universe Is Pregnant, Moral, and Meant to Be Understood"**  
> Scott Devine, January 2026

The paper proposes that **Faith**—defined as the algorithmic prior \( P(U) \propto 2^{-K(U)} \)—is the optimal prior for physics, cosmology, ethics, and materials science. Under this prior, the universe is expected to be compressible, coherent, and anti-entropic.

This repository provides:
- 🔬 **Experimental protocols** for 10 falsifiable predictions
- 📊 **Analysis scripts** for complexity/entropy measurements
- 💾 **Data templates** for consistent recording
- 🤖 **AI simulation code** for moral policy compression
- 📈 **Statistical analysis plans** with pre-registration templates

## 🧪 Predictions Tested

| Domain | Prediction | Status |
|--------|------------|--------|
| **Cosmic** | Cosmic compression factor \( R(z) \) decreases with time | Future (CMB-S4, LSST) |
| **Cosmic** | \( \Lambda \propto m_\nu^4 \) exactly | Ongoing (KATRIN, Project 8) |
| **Cosmic** | CMB excess power at \( l=2 \) (28-Gyr periodicity) | Ready (Planck/ACT reanalysis) |
| **Biophysical** | Virtuous speech compresses more than vicious speech | Ready (gzip analysis) |
| **Biophysical** | HRV coherence higher in faithful states | Ready (HRV monitoring) |
| **Social** | High-trust networks are more compressible | Ready (graph compression) |
| **Material** | Quartz crystal amplifies HRV coherence | Ready (double-blind study) |
| **Material** | Intention+crystal increases water \( T_2 \) relaxation | Ready (NMR protocol) |
| **Material** | Crystal grids reduce RNG entropy during meditation | Ready (quantum RNG) |
| **AI** | AEP-trained AI has more compressible policies | Ready (RL simulation) |
| **AI** | Virtuous AI generalizes better in moral dilemmas | Ready (generalization tests) |

## 📁 Repository Structure
```
faith-as-first-law/
├──experiments/              # Experimental protocols by domain
│├── speech_compression/   # Prediction B1: Virtuous speech compression
│├── hrv_entropy/          # Predictions B2, M1: HRV coherence
│├── social_networks/      # Prediction B3: Network compressibility
│├── rng_entropy/          # Prediction M3: RNG entropy reduction
│└── ai_policies/          # Predictions A1-A2: AI policy compression
├──analysis/                 # Statistical analysis tools
│├── statistical_tests.R   # R scripts for all tests
│├── compression_utils.py  # Python compression helpers
│└── plotting_functions.py # Visualization utilities
├──data/                     # Data handling
│├── templates/            # Empty CSV templates
│└── example/              # Sample datasets
├──paper/                    # Paper-related materials
│└── appendices/           # Complete appendices
├──LICENSE                   # MIT License
└──README.md                 # This file
```

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/scottdevine/faith-as-first-law.git
cd faith-as-first-law
```

### 2. Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# R dependencies (optional)
Rscript install_dependencies.R
```

### 3. Run an Example Experiment

Start with the simplest test (Prediction B1: speech compression):

```bash
cd experiments/speech_compression
python run_speech_study.py --condition virtue --participants 5
```

### 4. Analyze Results

```bash
cd ../analysis
Rscript statistical_tests.R --experiment speech --output results/
```

## 🔬 Quick Start: Test a Prediction

Example: Test Virtuous Speech Compression (Prediction B1)

```python
from compression_utils import analyze_speech_compression

# Load transcriptions
virtue_text = open("data/virtue_narrative.txt").read()
vice_text = open("data/vice_narrative.txt").read()

# Compute compression ratios
results = analyze_speech_compression([virtue_text, vice_text], 
                                     labels=['virtue', 'vice'])

print(f"Virtue compression ratio: {results['virtue']['C_R']:.3f}")
print(f"Vice compression ratio: {results['vice']['C_R']:.3f}")
print(f"Difference: {results['virtue']['C_R'] - results['vice']['C_R']:.3f}")
```

Expected Result:

```
Virtue compression ratio: 0.415
Vice compression ratio: 0.521
Difference: -0.106 (virtue compresses more)
```

### 📊 Data Collection Templates

Each experiment directory contains CSV templates for consistent data recording:

- participant_info.csv – Demographics and consent
- experimental_data.csv – Raw measurements
- analysis_results.csv – Processed outputs

### 📈 Statistical Analysis Plan

All experiments follow a pre-registered analysis plan:

1. Primary Outcome: Pre-specified complexity measure
2. Sample Size: Calculated for medium effect size (f=0.25) at 80% power
3. Analysis: Mixed-effects models for within-subjects designs
4. Correction: Bonferroni-Holm for multiple comparisons
5. Reporting: APA style with exact p-values and effect sizes

Pre-registration templates are available in paper/preregistration/.

### 🤝 Contributing

We welcome:

- 🔁 Independent replications (especially cross-cultural)
- 🔧 Code improvements and optimizations
- 📖 Documentation enhancements
- 🐛 Bug reports via GitHub Issues

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

This project is licensed under the MIT License - see the LICENSE file for details.

### 📬 Contact

For questions, collaboration requests, or replication support:

- Email: scottdevine01@gmail.com
- GitHub Issues: https://github.com/scottdevine/faith-as-first-law/issues
- Response Time: 1-2 weeks for non-urgent inquiries

### 🙏 Acknowledgments

- The developers of algorithmic information theory (Kolmogorov, Solomonoff, Chaitin)
- Researchers bridging physics, biology, and consciousness studies
- The open-source community for tools that make this work possible

---

“The universe is not a dying ember. It is a seed. And it is growing.”

```
