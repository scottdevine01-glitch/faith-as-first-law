```markdown
# Prediction B1: Virtuous Speech Compression

**Hypothesis:** Speech about virtuous acts compresses more (lower C_R) than speech about vicious acts.

## Protocol Summary
- **Design:** Within-subjects, three conditions (virtue/vice/neutral)
- **Task:** 3-minute monologue describing:
  - A personal act of kindness (virtue)
  - A personal act of deception (vice)
  - Morning routine (neutral)
- **Recording:** Audio → verbatim transcription
- **Compression:** Apply `gzip -9` to cleaned transcripts
- **Metric:** C_R = (compressed size)/(original size)

## Expected Result
C_R(virtue) < C_R(neutral) < C_R(vice)

## Quick Start

```bash
# Run with example data
python run_speech_study.py --example

# Run with your data
python run_speech_study.py --data_dir ./my_transcripts/
```

### Files

- run_speech_study.py – Main experiment script
- analysis.py – Statistical analysis
- data/ – Templates and example data
- results/ – Output directory

### Data Collection

1. Record audio of participants describing:
   - A virtuous act (kindness, honesty, courage)
   - A vicious act (deceit, selfishness, cruelty)
   - A neutral topic (daily routine)
2. Transcribe verbatim
3. Clean transcripts (remove filler words, timestamps)
4. Save as .txt files with naming convention:
   - `P001_virtue.txt`
   - `P001_vice.txt`
   - `P001_neutral.txt`

### Analysis

Run full statistical analysis:

```bash
python analysis.py --data results/speech_compression_results.csv --output analysis_report/
```

### Output

- speech_compression_results.csv – Raw results
- descriptive_stats.csv – Summary statistics
- pairwise_tests.csv – Statistical comparisons
- compression_plot.png – Visualization

### Dependencies

- Python 3.9+
- pandas, numpy, scipy, matplotlib, seaborn
- For audio recording: any standard recording device

### Ethics Note

- Obtain informed consent
- Provide debriefing after "vice" condition
- Allow participants to skip any condition
- Anonymize all data before sharing

```
