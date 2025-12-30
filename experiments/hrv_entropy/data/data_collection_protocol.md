# HRV Data Collection Protocol

## Equipment Setup
1. **HRV Monitor**: Polar H10 chest strap or equivalent FDA-cleared device
2. **Recording Software**: Kubios HRV Premium or custom Python script
3. **Crystals**: Quartz crystals (cleaned) and identical glass placebos
4. **Environment**: Quiet room, 22°C, minimal electromagnetic interference

## Session Procedure

### Phase 1: Baseline (5 minutes)
1. Participant sits comfortably
2. Attach HRV monitor
3. Record 5-minute resting baseline
4. Ensure signal quality > 0.90

### Phase 2: Intervention (10 minutes)
1. Randomly assign crystal condition:
   - **Quartz**: Hold cleansed quartz crystal
   - **Glass**: Hold identical glass placebo  
   - **None**: No object
2. Guide through condition:
   - **Compassion**: Loving-kindness meditation
   - **Resentment**: Recall past betrayal
   - **Neutral**: Listen to audiobook
3. Continuous HRV recording

### Phase 3: Recovery (5 minutes)
1. Remove object
2. Resting recording
3. Debrief participant

## Data Quality Checks
- **Signal quality**: > 0.90 Kubios quality score
- **Artifact correction**: < 5% beats corrected
- **Missing data**: < 1% of recording
- **Stationarity**: Pass Augmented Dickey-Fuller test (p < 0.05)

## File Naming Convention

HRV_P[PID][DATE][COND]_[CRYSTAL].edf
Example:HRV_P001_20260115_COMPASSION_QUARTZ.edf

```

## Ethical Considerations
- Informed consent for emotion induction
- Psychological support available
- Option to skip resentment condition
- Data anonymization before analysis
```
