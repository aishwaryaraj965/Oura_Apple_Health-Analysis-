

# Personal Health Agent (PHA) Analysis using Oura and Apple Wearable Data - Figures

## Overview
This repo contains scripts and visualizations from the Oura Ring data analysis using the Personal Health Agent framework from Google Research


Tools Used for Analysis: 
*  👩🏻‍💻 Cursor
*  ⌚️ 💍 Health Wearable Devices (Oura/Apple)


## Generated Figures

### 1. Health Coach Dashboard (`01_health_coach_dashboard.png`)
**Size:** 1.6 MB | **Resolution:** 300 DPI

A comprehensive 8-panel dashboard featuring:
- **Goal 1:** Sleep Duration Progress Tracking over time
- **Goal 2:** Sleep Consistency (7-day rolling standard deviation)
- **Goal 3:** Resting Heart Rate Optimization trend
- **Weekly Progress:** Last 8 weeks scorecard (Sleep, Readiness, Activity)
- **Bedtime Consistency:** By day of week with error bars
- **Recovery Capacity:** HRV trend with 14-day average
- **Goal Achievement Rate:** Success rates for key goals
- **12-Week Progress Dashboard:** All key metrics normalized and chronologically ordered

### 2. High-Level Trends Analysis (`02_high_level_trends.png`)
**Size:** 1.1 MB | **Resolution:** 300 DPI

Overview of your health metrics:
- Core Oura scores over time with trend lines
- Sleep and Readiness score distributions
- 7-day rolling averages with variability bands

### 3. Domain Expert Health Metrics (`03_domain_expert_health_metrics.png`)
**Size:** 1.7 MB | **Resolution:** 300 DPI

Evidence-based health insights dashboard featuring:
- Sleep duration vs CDC recommendations
- Resting heart rate distribution with fitness zones
- HRV trend over time
- Sleep efficiency distribution with quality bands
- Sleep stage breakdown (Deep, REM, Light)
- Correlation analyses (RHR vs Readiness, HRV vs Readiness, Sleep Duration vs Quality)
- Last 10 weeks comparison (chronologically ordered)


## How to Use These Figures

### For Personal Tracking
- Review weekly to track progress toward health goals
- Use the 12-week dashboard to identify trends over time
- Track progress against population health benchmarks for demographic characteristics

### For Healthcare Professionals
- Share the Domain Expert Health Metrics dashboard
- Evidence-based insights with correlation coefficients


## Data Source
- **File:** `oura_2024-01-01_2025-10-01_trends.csv`
- **Date Range:** January 1, 2024 - October 1, 2025
- **Records:** 483 daily observations
- **Metrics:** 54 health and activity parameters

## Personal Health Agent Framework

Based on [Google Research's PHA](https://research.google/blog/the-anatomy-of-a-personal-health-agent/), this analysis employs three specialist agents:

### 🔬 Data Science Agent
- Statistical analysis of time-series data
- Trend detection with significance testing
- Rolling averages and variability metrics

### 📚 Domain Expert Agent
- Evidence-based health recommendations
- Clinical correlation analysis
- Fitness zone classifications (CDC, AHA guidelines)

### 💪 Health Coach Agent
- 12-week goal-setting framework
- Behavioral change strategies
- Progress tracking and motivation


## Technical Details

- **Generated:** October 5, 2025
- **Script:** `complete_pha_analysis_fixed.py`
- **Python Version:** 3.x
- **Key Libraries:** pandas, numpy, matplotlib, seaborn, scipy

---

*For questions about the analysis or to regenerate with updated data, run:*
```bash
python complete_pha_analysis_fixed.py
```


