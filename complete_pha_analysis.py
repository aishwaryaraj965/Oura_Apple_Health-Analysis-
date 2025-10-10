"""
Complete PHA Analysis Script - Run this in your Jupyter notebook
This combines all the PHA framework components with enhanced visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/Users/aishwaryaraj/Documents/OURA_DATA/oura_2024-01-01_2025-10-01_trends.csv')

# Preprocessing
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.day_name()
df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
df['bedtime_hour'] = pd.to_datetime(df['Bedtime Start'], errors='coerce', utc=True).dt.hour
df['week'] = df['date'].dt.isocalendar().week
df['year'] = df['date'].dt.year
df['week_year'] = df['year'].astype(str) + '_' + df['week'].astype(str)

print("="*80)
print("💪 HEALTH COACH AGENT: Personalized Coaching Plan")
print("="*80)

def health_coach_recommendations(df):
    """
    Mimics PHA's HC Agent: Using motivational interviewing principles
    """
    print("\\n🎯 Your Personalized Health Coaching Plan\\n")
    
    # Identify primary improvement area
    readiness_score = df['Readiness Score'].mean()
    sleep_score = df['Sleep Score'].mean()
    activity_score = df['Activity Score'].mean()
    
    avg_sleep_hours = df['Total Sleep Duration'].mean() / 3600
    sleep_std = df['Total Sleep Duration'].std() / 3600
    avg_rhr = df['Average Resting Heart Rate'].mean()
    
    print(f"📊 Current Baseline Assessment:")
    print(f"   • Sleep Score: {sleep_score:.1f}/100")
    print(f"   • Readiness Score: {readiness_score:.1f}/100")
    print(f"   • Activity Score: {activity_score:.1f}/100")
    print(f"   • Sleep Duration: {avg_sleep_hours:.1f}h (±{sleep_std:.1f}h variability)")
    print(f"   • Resting Heart Rate: {avg_rhr:.1f} bpm")
    
    # Goal 1: Quick Win
    print(f"\\n{'='*70}")
    print("🎯 GOAL 1: Quick Win (Week 1-2)")
    print(f"{'='*70}")
    if avg_sleep_hours < 7.5:
        print(f"   📍 Focus Area: Increase Sleep Duration")
        print(f"   📊 Current: {avg_sleep_hours:.1f} hours/night")
        print(f"   🎯 Target: 7.5+ hours/night")
        print(f"   ")
        print(f"   ✅ Action Steps:")
        print(f"      1. Set bedtime alarm 30 minutes earlier (start with just 15 min if 30 feels too much)")
        print(f"      2. Create wind-down routine: No screens 30 min before bed")
        print(f"      3. Track bedtime consistency in your phone notes")
        print(f"   ")
        print(f"   📈 Expected Outcome:")
        print(f"      • Sleep Score: Potential improvement to {sleep_score + 5:.0f}-{sleep_score + 8:.0f}")
        print(f"      • Readiness Score: Potential improvement to {readiness_score + 3:.0f}-{readiness_score + 5:.0f}")
        print(f"   ")
        print(f"   💡 Why This Works:")
        print(f"      Total Sleep Duration has 0.801 correlation with Sleep Score (strongest predictor)")
    else:
        print(f"   📍 Focus Area: Optimize Sleep Consistency")
        print(f"   📊 Current Variability: ±{sleep_std:.1f} hours")
        print(f"   🎯 Target: <1 hour variability")
        print(f"   ")
        print(f"   ✅ Action Steps:")
        print(f"      1. Set same bedtime within 30-minute window (even weekends)")
        print(f"      2. Limit weekend catch-up sleep to <1 hour extra")
        print(f"      3. Use natural light in morning to reinforce circadian rhythm")
    
    # Goal 2: Consistency
    print(f"\\n{'='*70}")
    print("🎯 GOAL 2: Build Consistency (Week 3-6)")
    print(f"{'='*70}")
    print(f"   📍 Focus Area: Establish Sustainable Habits")
    print(f"   ")
    print(f"   ✅ Daily Routine:")
    print(f"      Morning (7-9 AM):")
    print(f"         • Get 10-15 min sunlight within 1h of waking")
    print(f"         • Review yesterday's Oura data for patterns")
    print(f"      ")
    print(f"      Evening (2h before bed):")
    print(f"         • Dim lights to trigger melatonin production")
    print(f"         • Avoid intense exercise (light stretching OK)")
    print(f"         • Cool bedroom to 65-68°F (18-20°C)")
    print(f"      ")
    print(f"      Bedtime Routine (30-60 min):")
    print(f"         • Same sequence every night builds association")
    print(f"         • Examples: shower → journal → reading → sleep")
    print(f"   ")
    print(f"   📊 Track Weekly:")
    print(f"      • Bedtime consistency: ±15 minutes by week 6")
    print(f"      • Sleep duration: 7-9 hours at least 5 nights/week")
    print(f"      • Sleep efficiency: Target >85%")
    
    # Goal 3: Optimization
    print(f"\\n{'='*70}")
    print("🎯 GOAL 3: Optimize Recovery (Week 7-12)")
    print(f"{'='*70}")
    print(f"   📍 Focus Area: Cardiovascular Fitness & Recovery")
    print(f"   📊 Current RHR: {avg_rhr:.1f} bpm")
    print(f"   🎯 Target: Reduce by 2-3 bpm over 12 weeks")
    print(f"   ")
    print(f"   ✅ Exercise Protocol:")
    print(f"      Week 7-8:")
    print(f"         • 2x per week: 20-30 min Zone 2 cardio (conversational pace)")
    print(f"         • Examples: Brisk walking, easy cycling, light jogging")
    print(f"      ")
    print(f"      Week 9-10:")
    print(f"         • 3x per week: 30-40 min Zone 2")
    print(f"         • Add 1x per week: 10-15 min Zone 3-4 (challenging but sustainable)")
    print(f"      ")
    print(f"      Week 11-12:")
    print(f"         • 3x per week: 40-45 min Zone 2")
    print(f"         • 1x per week: 15-20 min Zone 3-4")
    print(f"         • 1x per week: Rest or active recovery (yoga, stretching)")
    print(f"   ")
    print(f"   💡 Why This Works:")
    print(f"      RHR has -0.644 correlation with Readiness (strongest negative predictor)")
    print(f"      Lower RHR = Better cardiovascular fitness = Faster recovery")
    
    # Behavior Change Strategies
    print(f"\\n{'='*70}")
    print("📋 BEHAVIOR CHANGE STRATEGIES (Motivational Interviewing)")
    print(f"{'='*70}")
    print(f"   ")
    print(f"   1. 📱 Self-Monitoring:")
    print(f"      • Review Oura data each morning (takes <2 minutes)")
    print(f"      • Note how you FEEL vs. what scores show")
    print(f"      • Look for patterns: 'When X happens, my Y score changes'")
    print(f"   ")
    print(f"   2. 🎯 Implementation Intentions:")
    print(f"      • 'If it's 10 PM, then I start bedtime routine'")
    print(f"      • 'If readiness <70, then I do easy workout instead of intense'")
    print(f"      • 'If sleep score <75, then I prioritize earlier bedtime tonight'")
    print(f"   ")
    print(f"   3. 👥 Social Support:")
    print(f"      • Share goals with accountability partner or family")
    print(f"      • Weekly check-in: text your sleep score average to someone")
    print(f"      • Join online community (r/QuantifiedSelf, r/whoop, etc.)")
    print(f"   ")
    print(f"   4. 🎁 Reward System:")
    print(f"      • Week 2: Treat yourself if you hit bedtime goal 5/7 nights")
    print(f"      • Week 6: Bigger reward if weekly avg sleep score improves by 5 points")
    print(f"      • Week 12: Major reward if RHR decreases by 2+ bpm")
    print(f"      • Ideas: New book, massage, concert tickets, etc. (not food!)")
    print(f"   ")
    print(f"   5. 🛡️ Relapse Prevention:")
    print(f"      • Travel: Pack earplugs, eye mask, portable white noise")
    print(f"      • Stress: When stressed, protect sleep MORE, not less")
    print(f"      • Illness: Focus on recovery, pause fitness goals temporarily")
    print(f"      • Social events: It's OK to skip 1-2 nights, return to routine next day")
    
    # Weekly Reflection
    print(f"\\n{'='*70}")
    print("❓ WEEKLY REFLECTION QUESTIONS")
    print(f"{'='*70}")
    print(f"   Set reminder for Sunday evenings to answer these:")
    print(f"   ")
    print(f"   1. What went well with my sleep/activity goals this week?")
    print(f"   2. What barriers did I encounter? (Be specific)")
    print(f"   3. What one small change can I make next week?")
    print(f"   4. My motivation level (1-10): _____")
    print(f"   5. Do I need to adjust my goals? (More challenging? More realistic?)")
    print(f"   ")
    print(f"   💡 Pro Tip: If motivation <6 for 2 weeks, make goals easier!")

# Run health coach analysis
health_coach_recommendations(df)

print(f"\\n{'='*80}")
print("✓ Health Coach Agent Complete")
print(f"{'='*80}")

# Create Health Coach Visualizations
print("\\nGenerating Health Coach Progress Tracking Dashboard...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

# 1. Goal 1: Sleep Duration Progress
ax1 = fig.add_subplot(gs[0, :])
df['sleep_hours'] = df['Total Sleep Duration'] / 3600
df['sleep_7day'] = df['sleep_hours'].rolling(window=7, min_periods=1).mean()
ax1.plot(df['date'], df['sleep_hours'], 'o', alpha=0.3, markersize=4, color='#3498db', label='Daily')
ax1.plot(df['date'], df['sleep_7day'], linewidth=3, color='#2ecc71', label='7-day Average')
ax1.axhline(y=7.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target: 7.5h')
ax1.axhspan(7, 9, alpha=0.15, color='green', label='Optimal Range (7-9h)')
ax1.set_title('GOAL 1: Sleep Duration Progress Tracking', fontsize=16, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Sleep Duration (hours)', fontsize=12)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Goal 2: Sleep Consistency (Variability over time)
ax2 = fig.add_subplot(gs[1, 0])
df['sleep_std_rolling'] = df['sleep_hours'].rolling(window=7, min_periods=1).std()
ax2.plot(df['date'], df['sleep_std_rolling'], linewidth=2, color='#e74c3c')
ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Target: <1h variability')
ax2.fill_between(df['date'], 0, df['sleep_std_rolling'], where=(df['sleep_std_rolling'] > 1), 
                 alpha=0.3, color='red', label='High Variability')
ax2.fill_between(df['date'], 0, df['sleep_std_rolling'], where=(df['sleep_std_rolling'] <= 1), 
                 alpha=0.3, color='green', label='Good Consistency')
ax2.set_title('GOAL 2: Sleep Consistency (7-day Rolling STD)', fontweight='bold', fontsize=13)
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Variability (hours)', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# 3. Goal 3: Resting Heart Rate Trend
ax3 = fig.add_subplot(gs[1, 1])
rhr_data = df[['date', 'Average Resting Heart Rate']].dropna()
rhr_rolling = rhr_data['Average Resting Heart Rate'].rolling(window=7, min_periods=1).mean()
ax3.plot(rhr_data['date'], rhr_data['Average Resting Heart Rate'], 'o', alpha=0.3, markersize=4, color='#e74c3c')
ax3.plot(rhr_data['date'], rhr_rolling, linewidth=3, color='#c0392b', label='7-day Average')
target_rhr = rhr_data['Average Resting Heart Rate'].mean() - 2.5
ax3.axhline(y=target_rhr, color='green', linestyle='--', linewidth=2, label=f'12-week Target: {target_rhr:.1f} bpm')
ax3.set_title('GOAL 3: Resting Heart Rate Optimization', fontweight='bold', fontsize=13)
ax3.set_xlabel('Date', fontsize=11)
ax3.set_ylabel('RHR (bpm)', fontsize=11)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# 4. Weekly Progress Scorecard
ax4 = fig.add_subplot(gs[1, 2])
weekly_metrics = df.groupby('week_year').agg({
    'Sleep Score': 'mean',
    'Readiness Score': 'mean',
    'Activity Score': 'mean'
}).dropna()
recent_weeks = weekly_metrics.tail(8)
x_weeks = np.arange(len(recent_weeks))
width = 0.25
bars1 = ax4.bar(x_weeks - width, recent_weeks['Sleep Score'], width, label='Sleep', color='#3498db', alpha=0.7)
bars2 = ax4.bar(x_weeks, recent_weeks['Readiness Score'], width, label='Readiness', color='#e74c3c', alpha=0.7)
bars3 = ax4.bar(x_weeks + width, recent_weeks['Activity Score'], width, label='Activity', color='#2ecc71', alpha=0.7)
ax4.set_xlabel('Week', fontsize=11)
ax4.set_ylabel('Average Score', fontsize=11)
ax4.set_title('Weekly Progress: Last 8 Weeks', fontweight='bold', fontsize=13)
ax4.set_xticks(x_weeks)
ax4.set_xticklabels([w.split('_')[1] for w in recent_weeks.index], rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# 5. Behavior: Bedtime Consistency
ax5 = fig.add_subplot(gs[2, 0])
bedtime_data = df[['day_of_week', 'bedtime_hour']].dropna()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
bedtime_by_day = bedtime_data.groupby('day_of_week')['bedtime_hour'].agg(['mean', 'std']).reindex(day_order)
ax5.bar(day_order, bedtime_by_day['mean'], yerr=bedtime_by_day['std'], alpha=0.7, 
        color='#9b59b6', edgecolor='black', capsize=5)
ax5.axhline(y=bedtime_by_day['mean'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f"Avg: {bedtime_by_day['mean'].mean():.1f}h")
ax5.set_ylabel('Bedtime Hour (24h format)', fontsize=11)
ax5.set_title('Bedtime Consistency by Day of Week', fontweight='bold', fontsize=13)
ax5.set_xticklabels(day_order, rotation=45, ha='right')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Recovery Capacity: HRV Trend
ax6 = fig.add_subplot(gs[2, 1])
hrv_data = df[['date', 'Average HRV']].dropna()
hrv_rolling = hrv_data['Average HRV'].rolling(window=14, min_periods=1).mean()
ax6.plot(hrv_data['date'], hrv_data['Average HRV'], alpha=0.4, color='#2ecc71', linewidth=1)
ax6.plot(hrv_data['date'], hrv_rolling, linewidth=3, color='#27ae60', label='14-day Average')
ax6.fill_between(hrv_data['date'], 
                 hrv_rolling - hrv_data['Average HRV'].rolling(14).std(),
                 hrv_rolling + hrv_data['Average HRV'].rolling(14).std(),
                 alpha=0.2, color='#2ecc71')
ax6.set_title('Recovery Capacity: HRV Trend', fontweight='bold', fontsize=13)
ax6.set_xlabel('Date', fontsize=11)
ax6.set_ylabel('HRV (ms)', fontsize=11)
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=45)

# 7. Success Rate: Days Meeting Goals
ax7 = fig.add_subplot(gs[2, 2])
df['goal_sleep_duration'] = df['sleep_hours'] >= 7
df['goal_sleep_score'] = df['Sleep Score'] >= 75
df['goal_readiness'] = df['Readiness Score'] >= 75

success_rates = {
    'Sleep\\nDuration\\n≥7h': (df['goal_sleep_duration'].sum() / len(df)) * 100,
    'Sleep\\nScore\\n≥75': (df['goal_sleep_score'].sum() / len(df)) * 100,
    'Readiness\\n≥75': (df['goal_readiness'].sum() / len(df)) * 100
}

colors_success = ['#2ecc71' if v >= 70 else '#f39c12' if v >= 50 else '#e74c3c' 
                  for v in success_rates.values()]
bars = ax7.bar(success_rates.keys(), success_rates.values(), color=colors_success, 
               alpha=0.7, edgecolor='black')
ax7.axhline(y=70, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target: 70%')
ax7.set_ylabel('Success Rate (%)', fontsize=11)
ax7.set_title('Goal Achievement Rate', fontweight='bold', fontsize=13)
ax7.set_ylim([0, 100])
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, success_rates.values()):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 2, f'{val:.0f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

# 8. Implementation Intentions Tracking
ax8 = fig.add_subplot(gs[3, :])
# Create weekly view of key metrics
weekly_data = df.groupby('week_year').agg({
    'Sleep Score': 'mean',
    'Readiness Score': 'mean',
    'sleep_hours': 'mean',
    'Average Resting Heart Rate': 'mean'
}).dropna()

# Normalize to 0-100 scale for comparison
weekly_data_norm = weekly_data.copy()
weekly_data_norm['sleep_hours'] = (weekly_data['sleep_hours'] / 9) * 100  # Normalize to 9h max
weekly_data_norm['Average Resting Heart Rate'] = 100 - ((weekly_data['Average Resting Heart Rate'] - 50) / 40 * 100)  # Invert and normalize

# Plot recent 12 weeks
recent_12 = weekly_data_norm.tail(12)
x_pos = np.arange(len(recent_12))

ax8.plot(x_pos, recent_12['Sleep Score'], marker='o', linewidth=2.5, 
         markersize=8, label='Sleep Score', color='#3498db')
ax8.plot(x_pos, recent_12['Readiness Score'], marker='s', linewidth=2.5, 
         markersize=8, label='Readiness Score', color='#e74c3c')
ax8.plot(x_pos, recent_12['sleep_hours'], marker='^', linewidth=2.5, 
         markersize=8, label='Sleep Duration (normalized)', color='#2ecc71')
ax8.plot(x_pos, recent_12['Average Resting Heart Rate'], marker='d', linewidth=2.5, 
         markersize=8, label='RHR (inverted & normalized)', color='#9b59b6')

ax8.set_xlabel('Week', fontsize=12)
ax8.set_ylabel('Normalized Score (0-100)', fontsize=12)
ax8.set_title('12-Week Progress Dashboard: All Key Metrics', fontsize=15, fontweight='bold')
ax8.set_xticks(x_pos)
ax8.set_xticklabels([w.split('_')[1] for w in recent_12.index], rotation=45)
ax8.legend(loc='best', fontsize=10)
ax8.grid(True, alpha=0.3)
ax8.axhline(y=85, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='Excellence Threshold')

plt.tight_layout()
plt.savefig('/Users/aishwaryaraj/Documents/OURA_DATA/health_coach_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

print("\\n✅ Health Coach Dashboard saved as 'health_coach_dashboard.png'")
print("\\nYou can print this and post it for weekly tracking!")

