"""
Apple Health Personal Health Agent (PHA) Analysis
Adapted from the Oura Ring PHA framework for Apple Health data

This script parses Apple Health export XML and applies the PHA framework:
- Data Science Agent: Statistical analysis
- Domain Expert Agent: Evidence-based insights
- Health Coach Agent: Behavioral recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import warnings
import os
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Configuration
APPLE_HEALTH_EXPORT_PATH = '/Users/aishwaryaraj/Documents/OURA_DATA/shreyas_apple_health data/apple_health_export/export.xml'
OUTPUT_DIR = '/Users/aishwaryaraj/Documents/OURA_DATA/Apple_Health_PHA_Figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("🍎 APPLE HEALTH PERSONAL HEALTH AGENT ANALYSIS")
print("="*80)
print(f"\n📁 Reading Apple Health data from: {APPLE_HEALTH_EXPORT_PATH}")
print(f"📊 Figures will be saved to: {OUTPUT_DIR}\n")

def parse_apple_health_xml(file_path):
    """
    Parse Apple Health export XML and extract relevant health metrics
    """
    print("⏳ Parsing XML file (this may take a few minutes for large exports)...")
    
    # Parse XML incrementally to handle large files
    records = []
    workout_records = []
    
    # Use iterparse for memory efficiency
    context = ET.iterparse(file_path, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)
    
    record_count = 0
    workout_count = 0
    
    for event, elem in context:
        if event == 'end':
            if elem.tag == 'Record':
                record = {
                    'type': elem.get('type'),
                    'value': elem.get('value'),
                    'unit': elem.get('unit'),
                    'sourceName': elem.get('sourceName'),
                    'startDate': elem.get('startDate'),
                    'endDate': elem.get('endDate'),
                    'creationDate': elem.get('creationDate')
                }
                records.append(record)
                record_count += 1
                
                if record_count % 100000 == 0:
                    print(f"   Processed {record_count:,} records...")
                
            elif elem.tag == 'Workout':
                workout = {
                    'workoutActivityType': elem.get('workoutActivityType'),
                    'duration': elem.get('duration'),
                    'totalDistance': elem.get('totalDistance'),
                    'totalEnergyBurned': elem.get('totalEnergyBurned'),
                    'startDate': elem.get('startDate'),
                    'endDate': elem.get('endDate')
                }
                workout_records.append(workout)
                workout_count += 1
            
            # Clear element to free memory
            elem.clear()
            root.clear()
    
    print(f"\n✅ Parsed {record_count:,} health records and {workout_count:,} workouts")
    
    # Convert to DataFrames
    df_records = pd.DataFrame(records)
    df_workouts = pd.DataFrame(workout_records)
    
    return df_records, df_workouts

def extract_daily_metrics(df_records, df_workouts):
    """
    Extract and aggregate daily health metrics
    """
    print("\n📊 Extracting daily health metrics...")
    
    # Convert date strings to datetime
    df_records['date'] = pd.to_datetime(df_records['startDate'], errors='coerce')
    df_records['value'] = pd.to_numeric(df_records['value'], errors='coerce')
    
    # Extract just the date (no time)
    df_records['day'] = df_records['date'].dt.date
    
    # Key metrics to extract
    metrics_to_extract = {
        'HKQuantityTypeIdentifierStepCount': 'steps',
        'HKQuantityTypeIdentifierDistanceWalkingRunning': 'distance_walking_running',
        'HKQuantityTypeIdentifierActiveEnergyBurned': 'active_energy',
        'HKQuantityTypeIdentifierBasalEnergyBurned': 'basal_energy',
        'HKQuantityTypeIdentifierHeartRate': 'heart_rate',
        'HKQuantityTypeIdentifierRestingHeartRate': 'resting_heart_rate',
        'HKQuantityTypeIdentifierHeartRateVariabilitySDNN': 'hrv',
        'HKQuantityTypeIdentifierVO2Max': 'vo2_max',
        'HKCategoryTypeIdentifierSleepAnalysis': 'sleep',
        'HKQuantityTypeIdentifierBodyMass': 'body_mass',
        'HKQuantityTypeIdentifierBodyMassIndex': 'bmi',
        'HKQuantityTypeIdentifierHeight': 'height',
        'HKQuantityTypeIdentifierRespiratoryRate': 'respiratory_rate',
        'HKQuantityTypeIdentifierOxygenSaturation': 'oxygen_saturation',
        'HKQuantityTypeIdentifierWalkingSpeed': 'walking_speed',
        'HKQuantityTypeIdentifierWalkingStepLength': 'step_length',
        'HKQuantityTypeIdentifierWalkingAsymmetryPercentage': 'walking_asymmetry',
        'HKQuantityTypeIdentifierAppleExerciseTime': 'exercise_minutes',
        'HKQuantityTypeIdentifierAppleStandTime': 'stand_minutes',
    }
    
    # Create aggregated daily metrics
    daily_data = {}
    
    for hk_type, metric_name in metrics_to_extract.items():
        metric_df = df_records[df_records['type'] == hk_type].copy()
        
        if len(metric_df) > 0:
            if metric_name in ['steps', 'active_energy', 'basal_energy', 'distance_walking_running', 
                                'exercise_minutes', 'stand_minutes']:
                # Sum for cumulative metrics
                daily_agg = metric_df.groupby('day')['value'].sum()
            else:
                # Mean for instantaneous measurements
                daily_agg = metric_df.groupby('day')['value'].mean()
            
            daily_data[metric_name] = daily_agg
            print(f"   ✓ {metric_name}: {len(daily_agg)} days of data")
    
    # Create unified daily DataFrame
    df_daily = pd.DataFrame(daily_data)
    df_daily.index = pd.to_datetime(df_daily.index)
    df_daily = df_daily.sort_index()
    
    # Add derived metrics
    if 'active_energy' in df_daily.columns and 'basal_energy' in df_daily.columns:
        df_daily['total_energy'] = df_daily['active_energy'] + df_daily['basal_energy']
    
    # Add time features
    df_daily['day_of_week'] = df_daily.index.day_name()
    df_daily['is_weekend'] = df_daily['day_of_week'].isin(['Saturday', 'Sunday'])
    df_daily['year'] = df_daily.index.year
    df_daily['month'] = df_daily.index.month
    df_daily['week'] = df_daily.index.isocalendar().week
    df_daily['week_year'] = df_daily['year'].astype(str) + '_' + df_daily['week'].astype(str).str.zfill(2)
    
    print(f"\n✅ Created daily dataset: {len(df_daily)} days from {df_daily.index.min().date()} to {df_daily.index.max().date()}")
    print(f"\n📈 Available metrics:")
    for col in df_daily.columns:
        if col not in ['day_of_week', 'is_weekend', 'year', 'month', 'week', 'week_year']:
            non_null = df_daily[col].notna().sum()
            print(f"   • {col}: {non_null} days ({non_null/len(df_daily)*100:.1f}% coverage)")
    
    return df_daily

# Parse the data
print("\n" + "="*80)
print("STEP 1: PARSING APPLE HEALTH DATA")
print("="*80)
df_records, df_workouts = parse_apple_health_xml(APPLE_HEALTH_EXPORT_PATH)

# Extract daily metrics
print("\n" + "="*80)
print("STEP 2: EXTRACTING DAILY METRICS")
print("="*80)
df_daily = extract_daily_metrics(df_records, df_workouts)

# Save processed data
csv_path = OUTPUT_DIR + '/apple_health_daily_metrics.csv'
df_daily.to_csv(csv_path)
print(f"\n💾 Saved processed data to: {csv_path}")

print("\n" + "="*80)
print("STEP 3: PERSONAL HEALTH AGENT ANALYSIS")
print("="*80)

# Filter to recent data (last 365 days with data)
df_recent = df_daily[df_daily.index >= (df_daily.index.max() - pd.Timedelta(days=365))].copy()
print(f"\n📅 Analyzing last 365 days: {df_recent.index.min().date()} to {df_recent.index.max().date()}")

# Basic statistics
print(f"\n📊 KEY STATISTICS:")
if 'steps' in df_recent.columns:
    print(f"   • Average Daily Steps: {df_recent['steps'].mean():.0f} ± {df_recent['steps'].std():.0f}")
if 'resting_heart_rate' in df_recent.columns:
    print(f"   • Average Resting Heart Rate: {df_recent['resting_heart_rate'].mean():.1f} bpm")
if 'hrv' in df_recent.columns:
    print(f"   • Average HRV (SDNN): {df_recent['hrv'].mean():.1f} ms")
if 'active_energy' in df_recent.columns:
    print(f"   • Average Active Energy: {df_recent['active_energy'].mean():.0f} kcal/day")
if 'exercise_minutes' in df_recent.columns:
    print(f"   • Average Exercise Minutes: {df_recent['exercise_minutes'].mean():.0f} min/day")

# Create visualizations
print("\n" + "="*80)
print("STEP 4: GENERATING VISUALIZATIONS")
print("="*80)

# Figure 1: Activity & Energy Dashboard
fig1 = plt.figure(figsize=(20, 14))
gs1 = fig1.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

# 1. Daily Steps Over Time
if 'steps' in df_recent.columns:
    ax1 = fig1.add_subplot(gs1[0, :])
    df_recent['steps_7day'] = df_recent['steps'].rolling(window=7, min_periods=1).mean()
    ax1.plot(df_recent.index, df_recent['steps'], 'o', alpha=0.3, markersize=4, 
             color='#3498db', label='Daily Steps')
    ax1.plot(df_recent.index, df_recent['steps_7day'], linewidth=3, color='#2ecc71', 
             label='7-day Average')
    ax1.axhline(y=10000, color='green', linestyle='--', linewidth=2, alpha=0.7, 
                label='Target: 10,000 steps')
    ax1.set_title('Daily Step Count Progress', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Steps', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

# 2. Steps Distribution
if 'steps' in df_recent.columns:
    ax2 = fig1.add_subplot(gs1[1, 0])
    steps_data = df_recent['steps'].dropna()
    ax2.hist(steps_data, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
    ax2.axvline(steps_data.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {steps_data.mean():.0f}')
    ax2.axvline(10000, color='green', linestyle='--', linewidth=2, label='WHO Target: 10,000')
    ax2.set_xlabel('Daily Steps', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Step Count Distribution', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# 3. Active Energy Over Time
if 'active_energy' in df_recent.columns:
    ax3 = fig1.add_subplot(gs1[1, 1])
    df_recent['energy_7day'] = df_recent['active_energy'].rolling(window=7, min_periods=1).mean()
    energy_data = df_recent['active_energy'].dropna()
    ax3.plot(df_recent.index, df_recent['active_energy'], alpha=0.4, color='#e74c3c', linewidth=1)
    ax3.plot(df_recent.index, df_recent['energy_7day'], linewidth=3, color='#c0392b', 
             label='7-day Average')
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('Active Energy (kcal)', fontsize=11)
    ax3.set_title('Active Energy Expenditure', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

# 4. Exercise Minutes
if 'exercise_minutes' in df_recent.columns:
    ax4 = fig1.add_subplot(gs1[1, 2])
    exercise_data = df_recent['exercise_minutes'].dropna()
    ax4.hist(exercise_data, bins=30, alpha=0.7, color='#2ecc71', edgecolor='black')
    ax4.axvline(exercise_data.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {exercise_data.mean():.0f} min')
    ax4.axvline(30, color='green', linestyle='--', linewidth=2, label='AHA Target: 30 min')
    ax4.set_xlabel('Exercise Minutes', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Daily Exercise Duration', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

# 5. Day of Week Analysis
if 'steps' in df_recent.columns:
    ax5 = fig1.add_subplot(gs1[2, 0])
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    steps_by_day = df_recent.groupby('day_of_week')['steps'].agg(['mean', 'std']).reindex(day_order)
    bars = ax5.bar(day_order, steps_by_day['mean'], yerr=steps_by_day['std'], alpha=0.7, 
                   color='#3498db', edgecolor='black', capsize=5)
    ax5.axhline(y=10000, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax5.set_ylabel('Average Steps', fontsize=11)
    ax5.set_title('Steps by Day of Week', fontweight='bold', fontsize=12)
    ax5.set_xticklabels(day_order, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3, axis='y')

# 6. Weekday vs Weekend
if 'steps' in df_recent.columns:
    ax6 = fig1.add_subplot(gs1[2, 1])
    weekday_data = df_recent[~df_recent['is_weekend']]['steps'].dropna()
    weekend_data = df_recent[df_recent['is_weekend']]['steps'].dropna()
    
    data_to_plot = [weekday_data, weekend_data]
    labels = ['Weekday', 'Weekend']
    bp = ax6.boxplot(data_to_plot, labels=labels, patch_artist=True,
                     boxprops=dict(facecolor='#3498db', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    ax6.axhline(y=10000, color='green', linestyle='--', linewidth=2, alpha=0.5, 
                label='Target: 10,000')
    ax6.set_ylabel('Steps', fontsize=11)
    ax6.set_title('Weekday vs Weekend Activity', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

# 7. Weekly Progress (last 12 weeks)
if 'steps' in df_recent.columns:
    ax7 = fig1.add_subplot(gs1[2, 2])
    weekly_steps = df_recent.groupby('week_year').agg({
        'steps': 'mean',
        df_recent.index: 'first'  # Keep first date for sorting
    }).dropna()
    weekly_steps = weekly_steps.rename(columns={'date': 'week_date'})
    weekly_steps['week_date'] = pd.to_datetime(weekly_steps['week_date'])
    weekly_steps = weekly_steps.sort_values('week_date').tail(12)
    
    x_pos = np.arange(len(weekly_steps))
    bars = ax7.bar(x_pos, weekly_steps['steps'], alpha=0.7, color='#3498db', edgecolor='black')
    ax7.axhline(y=10000, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax7.set_xlabel('Week', fontsize=11)
    ax7.set_ylabel('Average Daily Steps', fontsize=11)
    ax7.set_title('Last 12 Weeks: Step Progress', fontweight='bold', fontsize=12)
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels([f"Wk{w.split('_')[1]}" for w in weekly_steps.index], rotation=45)
    ax7.grid(True, alpha=0.3, axis='y')

# 8. Monthly Trends
if 'steps' in df_recent.columns:
    ax8 = fig1.add_subplot(gs1[3, :])
    monthly_data = df_recent.groupby(df_recent.index.to_period('M')).agg({
        'steps': 'mean',
        'active_energy': 'mean' if 'active_energy' in df_recent.columns else lambda x: np.nan,
        'exercise_minutes': 'mean' if 'exercise_minutes' in df_recent.columns else lambda x: np.nan
    })
    monthly_data.index = monthly_data.index.to_timestamp()
    
    ax8_twin1 = ax8.twinx()
    ax8_twin2 = ax8.twinx()
    ax8_twin2.spines['right'].set_position(('outward', 60))
    
    p1 = ax8.plot(monthly_data.index, monthly_data['steps'], marker='o', linewidth=2.5, 
                  markersize=8, label='Steps', color='#3498db')
    if 'active_energy' in df_recent.columns:
        p2 = ax8_twin1.plot(monthly_data.index, monthly_data['active_energy'], marker='s', 
                           linewidth=2.5, markersize=8, label='Active Energy', color='#e74c3c')
    if 'exercise_minutes' in df_recent.columns:
        p3 = ax8_twin2.plot(monthly_data.index, monthly_data['exercise_minutes'], marker='^', 
                           linewidth=2.5, markersize=8, label='Exercise Min', color='#2ecc71')
    
    ax8.set_xlabel('Month', fontsize=12)
    ax8.set_ylabel('Average Steps', fontsize=12, color='#3498db')
    ax8_twin1.set_ylabel('Active Energy (kcal)', fontsize=12, color='#e74c3c')
    ax8_twin2.set_ylabel('Exercise Minutes', fontsize=12, color='#2ecc71')
    ax8.set_title('Monthly Activity Trends', fontsize=14, fontweight='bold')
    ax8.tick_params(axis='y', labelcolor='#3498db')
    ax8_twin1.tick_params(axis='y', labelcolor='#e74c3c')
    ax8_twin2.tick_params(axis='y', labelcolor='#2ecc71')
    ax8.tick_params(axis='x', rotation=45)
    ax8.grid(True, alpha=0.3)
    
    # Combine legends
    lines = p1
    if 'active_energy' in df_recent.columns:
        lines += p2
    if 'exercise_minutes' in df_recent.columns:
        lines += p3
    labels = [l.get_label() for l in lines]
    ax8.legend(lines, labels, loc='best', fontsize=10)

plt.tight_layout()
fig1_path = f'{OUTPUT_DIR}/01_activity_energy_dashboard.png'
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {fig1_path}")
plt.close()

# Figure 2: Heart Health Dashboard
fig2 = plt.figure(figsize=(20, 12))
gs2 = fig2.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# 1. Resting Heart Rate Over Time
if 'resting_heart_rate' in df_recent.columns:
    ax1 = fig2.add_subplot(gs2[0, :])
    rhr_data = df_recent[['resting_heart_rate']].dropna()
    rhr_data['rhr_7day'] = rhr_data['resting_heart_rate'].rolling(window=7, min_periods=1).mean()
    
    ax1.plot(rhr_data.index, rhr_data['resting_heart_rate'], 'o', alpha=0.3, markersize=4, 
             color='#e74c3c', label='Daily RHR')
    ax1.plot(rhr_data.index, rhr_data['rhr_7day'], linewidth=3, color='#c0392b', 
             label='7-day Average')
    
    # Add fitness zones
    ax1.axhspan(40, 60, alpha=0.1, color='darkgreen', label='Excellent')
    ax1.axhspan(60, 70, alpha=0.1, color='green')
    ax1.axhspan(70, 80, alpha=0.1, color='yellow')
    
    ax1.set_title('Resting Heart Rate Trend', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('RHR (bpm)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

# 2. RHR Distribution
if 'resting_heart_rate' in df_recent.columns:
    ax2 = fig2.add_subplot(gs2[1, 0])
    rhr = df_recent['resting_heart_rate'].dropna()
    ax2.hist(rhr, bins=25, alpha=0.7, color='#e74c3c', edgecolor='black')
    ax2.axvline(rhr.mean(), color='darkred', linestyle='--', linewidth=2, 
                label=f'Your Avg: {rhr.mean():.1f} bpm')
    ax2.axvspan(40, 60, alpha=0.15, color='darkgreen', label='Athlete')
    ax2.axvspan(60, 70, alpha=0.15, color='green', label='Excellent')
    ax2.axvspan(70, 80, alpha=0.15, color='yellow', label='Good')
    ax2.set_xlabel('Resting Heart Rate (bpm)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('RHR Distribution & Fitness Zones', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

# 3. HRV Over Time
if 'hrv' in df_recent.columns:
    ax3 = fig2.add_subplot(gs2[1, 1])
    hrv_data = df_recent[['hrv']].dropna()
    hrv_data['hrv_7day'] = hrv_data['hrv'].rolling(window=7, min_periods=1).mean()
    
    ax3.scatter(hrv_data.index, hrv_data['hrv'], alpha=0.4, s=30, color='#2ecc71')
    ax3.plot(hrv_data.index, hrv_data['hrv_7day'], linewidth=3, color='#27ae60', 
             label='7-day avg')
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('HRV SDNN (ms)', fontsize=11)
    ax3.set_title('Heart Rate Variability Trend', fontweight='bold', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# 4. HRV Distribution
if 'hrv' in df_recent.columns:
    ax4 = fig2.add_subplot(gs2[1, 2])
    hrv = df_recent['hrv'].dropna()
    ax4.hist(hrv, bins=25, alpha=0.7, color='#2ecc71', edgecolor='black')
    ax4.axvline(hrv.mean(), color='darkgreen', linestyle='--', linewidth=2, 
                label=f'Your Avg: {hrv.mean():.1f} ms')
    ax4.set_xlabel('HRV SDNN (ms)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('HRV Distribution', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

# 5. RHR vs Steps Correlation
if 'resting_heart_rate' in df_recent.columns and 'steps' in df_recent.columns:
    ax5 = fig2.add_subplot(gs2[2, 0])
    scatter_data = df_recent[['resting_heart_rate', 'steps']].dropna()
    scatter = ax5.scatter(scatter_data['steps'], scatter_data['resting_heart_rate'],
                         c=scatter_data['resting_heart_rate'], cmap='RdYlGn_r', alpha=0.6, s=50)
    
    # Add trend line
    z = np.polyfit(scatter_data['steps'], scatter_data['resting_heart_rate'], 1)
    p = np.poly1d(z)
    ax5.plot(scatter_data['steps'], p(scatter_data['steps']), "r--", alpha=0.8, linewidth=2)
    
    corr = scatter_data['steps'].corr(scatter_data['resting_heart_rate'])
    ax5.set_xlabel('Daily Steps', fontsize=11)
    ax5.set_ylabel('Resting Heart Rate (bpm)', fontsize=11)
    ax5.set_title(f'Steps vs RHR (r={corr:.3f})', fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='RHR')

# 6. HRV vs Active Energy
if 'hrv' in df_recent.columns and 'active_energy' in df_recent.columns:
    ax6 = fig2.add_subplot(gs2[2, 1])
    scatter_data2 = df_recent[['hrv', 'active_energy']].dropna()
    scatter2 = ax6.scatter(scatter_data2['active_energy'], scatter_data2['hrv'],
                          c=scatter_data2['hrv'], cmap='RdYlGn', alpha=0.6, s=50)
    
    # Add trend line
    z2 = np.polyfit(scatter_data2['active_energy'], scatter_data2['hrv'], 1)
    p2 = np.poly1d(z2)
    ax6.plot(scatter_data2['active_energy'], p2(scatter_data2['active_energy']), 
             "r--", alpha=0.8, linewidth=2)
    
    corr2 = scatter_data2['active_energy'].corr(scatter_data2['hrv'])
    ax6.set_xlabel('Active Energy (kcal)', fontsize=11)
    ax6.set_ylabel('HRV (ms)', fontsize=11)
    ax6.set_title(f'Energy vs HRV (r={corr2:.3f})', fontweight='bold', fontsize=12)
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax6, label='HRV')

# 7. VO2 Max Trend (if available)
if 'vo2_max' in df_recent.columns:
    ax7 = fig2.add_subplot(gs2[2, 2])
    vo2_data = df_recent[['vo2_max']].dropna()
    if len(vo2_data) > 0:
        ax7.plot(vo2_data.index, vo2_data['vo2_max'], marker='o', linewidth=2, 
                markersize=6, color='#9b59b6', label='VO2 Max')
        ax7.set_xlabel('Date', fontsize=11)
        ax7.set_ylabel('VO2 Max (mL/kg/min)', fontsize=11)
        ax7.set_title('Cardio Fitness (VO2 Max)', fontweight='bold', fontsize=12)
        ax7.tick_params(axis='x', rotation=45)
        ax7.legend()
        ax7.grid(True, alpha=0.3)

plt.tight_layout()
fig2_path = f'{OUTPUT_DIR}/02_heart_health_dashboard.png'
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {fig2_path}")
plt.close()

print(f"\n{'='*80}")
print("✅ ANALYSIS COMPLETE!")
print(f"{'='*80}")
print(f"\n📁 All figures saved to: {OUTPUT_DIR}")
print(f"\n📊 Generated:")
print(f"   1. Activity & Energy Dashboard")
print(f"   2. Heart Health Dashboard")
print(f"   3. Processed CSV: apple_health_daily_metrics.csv")
print(f"\n✨ Apple Health PHA Analysis Complete! ✨")


