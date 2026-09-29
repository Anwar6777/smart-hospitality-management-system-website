"""
Module 3: Room Demand Forecasting
Step 4: Preprocessing (chronological train/test split, scaling)

IMPORTANT: time series data must NOT be shuffled before splitting -
we train on the past and test on the future, exactly as a real
deployment would work.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('data/demand_featured.csv', parse_dates=['date'])

TARGET = 'Total'
feature_cols = [c for c in df.columns if c not in ['date', 'Total', 'City Hotel', 'Resort Hotel',
                                                     'total_7d_avg', 'day_of_week', 'month']]
# (day_of_week/month kept only in their sin/cos + is_weekend form to avoid duplicate raw+cyclical signal)

print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

X = df[feature_cols]
y = df[TARGET]
dates = df['date']

# ------------------------------------------------------------------
# 1. Chronological 80/20 split (last 20% of days = test set)
# ------------------------------------------------------------------
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_train, dates_test = dates.iloc[:split_idx], dates.iloc[split_idx:]

print(f"\nTrain period: {dates_train.min().date()} to {dates_train.max().date()} ({len(X_train)} days)")
print(f"Test period:  {dates_test.min().date()} to {dates_test.max().date()} ({len(X_test)} days)")

# ------------------------------------------------------------------
# 2. Scale features (fit only on train, apply to test - no leakage)
# ------------------------------------------------------------------
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# ------------------------------------------------------------------
# 3. Save
# ------------------------------------------------------------------
np.save('data/X_train_scaled.npy', X_train_scaled)
np.save('data/X_test_scaled.npy', X_test_scaled)
np.save('data/y_train_scaled.npy', y_train_scaled)
np.save('data/y_test_scaled.npy', y_test_scaled)
X_train.to_csv('data/X_train_raw.csv', index=False)
X_test.to_csv('data/X_test_raw.csv', index=False)
y_train.to_csv('data/y_train_raw.csv', index=False)
y_test.to_csv('data/y_test_raw.csv', index=False)
dates_test.to_csv('data/dates_test.csv', index=False)

joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')
joblib.dump(feature_cols, 'models/feature_columns.pkl')

print("\nSaved preprocessed data and scalers")
