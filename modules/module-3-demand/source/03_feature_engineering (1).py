"""
Module 3: Room Demand Forecasting
Step 3: Feature Engineering (lag features, rolling stats, calendar features)

We forecast 'Total' daily room demand (across both hotel types).
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/daily_demand.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

TARGET = 'Total'

# ------------------------------------------------------------------
# 1. Calendar features
# ------------------------------------------------------------------
df['day_of_week'] = df['date'].dt.dayofweek       # 0=Monday
df['day_of_month'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# cyclical encoding so the model understands e.g. Dec(12) is close to Jan(1)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# ------------------------------------------------------------------
# 2. Lag features (yesterday, 2 days ago, same day last week)
# ------------------------------------------------------------------
for lag in [1, 2, 3, 7, 14]:
    df[f'lag_{lag}'] = df[TARGET].shift(lag)

# ------------------------------------------------------------------
# 3. Rolling window statistics (based only on past values, no leakage)
# ------------------------------------------------------------------
df['rolling_mean_7'] = df[TARGET].shift(1).rolling(7).mean()
df['rolling_std_7'] = df[TARGET].shift(1).rolling(7).std()
df['rolling_mean_14'] = df[TARGET].shift(1).rolling(14).mean()
df['rolling_mean_30'] = df[TARGET].shift(1).rolling(30).mean()

# ------------------------------------------------------------------
# 4. Drop rows with NaN from lag/rolling creation (first ~30 days)
# ------------------------------------------------------------------
before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"Dropped {before - len(df)} rows with NaN (from lag/rolling window warm-up)")
print(f"Final shape: {df.shape}")

df.to_csv('data/demand_featured.csv', index=False)
print("Saved -> data/demand_featured.csv")
