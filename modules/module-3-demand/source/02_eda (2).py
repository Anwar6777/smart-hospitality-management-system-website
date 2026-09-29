"""
Module 3: Room Demand Forecasting
Step 2: Exploratory Data Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110

df = pd.read_csv('data/daily_demand.csv', parse_dates=['date'])
df['day_of_week'] = df['date'].dt.day_name()
df['month'] = df['date'].dt.month_name()
df['year'] = df['date'].dt.year

# ------------------------------------------------------------------
# 1. Overall demand trend
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(df['date'], df['Total'], label='Total', color='#4C72B0', linewidth=1)
ax.plot(df['date'], df['City Hotel'], label='City Hotel', color='#DD8452', alpha=0.7, linewidth=1)
ax.plot(df['date'], df['Resort Hotel'], label='Resort Hotel', color='#55A868', alpha=0.7, linewidth=1)
ax.set_title('Daily Room Demand Over Time')
ax.set_ylabel('Rooms Occupied (arrivals)')
ax.legend()
plt.tight_layout()
plt.savefig('plots/01_demand_trend.png')
plt.close()

# ------------------------------------------------------------------
# 2. 7-day rolling average (smooths daily noise, reveals trend)
# ------------------------------------------------------------------
df['total_7d_avg'] = df['Total'].rolling(7).mean()
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(df['date'], df['Total'], alpha=0.3, label='Daily', color='gray')
ax.plot(df['date'], df['total_7d_avg'], label='7-day rolling avg', color='#C44E52', linewidth=2)
ax.set_title('Total Demand: Daily vs 7-Day Rolling Average')
ax.legend()
plt.tight_layout()
plt.savefig('plots/02_rolling_average.png')
plt.close()

# ------------------------------------------------------------------
# 3. Day-of-week seasonality
# ------------------------------------------------------------------
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.boxplot(data=df, x='day_of_week', y='Total', order=dow_order, ax=ax, color='#4C72B0')
ax.set_title('Demand Distribution by Day of Week')
plt.tight_layout()
plt.savefig('plots/03_day_of_week.png')
plt.close()

# ------------------------------------------------------------------
# 4. Monthly seasonality
# ------------------------------------------------------------------
month_order = ['January','February','March','April','May','June','July',
                'August','September','October','November','December']
fig, ax = plt.subplots(figsize=(11, 4.5))
sns.boxplot(data=df, x='month', y='Total', order=month_order, ax=ax, color='#55A868')
ax.set_title('Demand Distribution by Month')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('plots/04_monthly_seasonality.png')
plt.close()

# ------------------------------------------------------------------
# 5. Summary stats
# ------------------------------------------------------------------
print("Overall demand stats:")
print(df['Total'].describe())
print("\nAverage demand by day of week:")
print(df.groupby('day_of_week')['Total'].mean().reindex(dow_order).round(1))
print("\nPlots saved to plots/ directory")
