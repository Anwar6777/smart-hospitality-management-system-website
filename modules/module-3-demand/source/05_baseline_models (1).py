"""
Module 3: Room Demand Forecasting
Step 5: Baseline Models (Linear Regression + Random Forest)
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train_raw = pd.read_csv('data/X_train_raw.csv')
X_test_raw = pd.read_csv('data/X_test_raw.csv')
y_train = pd.read_csv('data/y_train_raw.csv').values.ravel()
y_test = pd.read_csv('data/y_test_raw.csv').values.ravel()
dates_test = pd.read_csv('data/dates_test.csv', parse_dates=['date'])['date']

results = {}

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    r2 = r2_score(y_true, y_pred)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}
    print(f"\n--- {name} ---")
    for k, v in results[name].items():
        print(f"  {k}: {v:.3f}")

# ------------------------------------------------------------------
# 1. Naive baseline: "tomorrow = today" (lag_1) - the sanity-check floor
# ------------------------------------------------------------------
naive_pred = X_test_raw['lag_1'].values
evaluate('Naive (yesterday = today)', y_test, naive_pred)

# ------------------------------------------------------------------
# 2. Linear Regression
# ------------------------------------------------------------------
lr = LinearRegression()
lr.fit(X_train_raw, y_train)
lr_pred = lr.predict(X_test_raw)
evaluate('Linear Regression', y_test, lr_pred)

# ------------------------------------------------------------------
# 3. Random Forest
# ------------------------------------------------------------------
rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train_raw, y_train)
rf_pred = rf.predict(X_test_raw)
evaluate('Random Forest', y_test, rf_pred)

# ------------------------------------------------------------------
# 4. Plot actual vs predicted (test period)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(dates_test, y_test, label='Actual', color='black', linewidth=1.5)
ax.plot(dates_test, lr_pred, label='Linear Regression', alpha=0.7)
ax.plot(dates_test, rf_pred, label='Random Forest', alpha=0.7)
ax.set_title('Room Demand Forecast: Actual vs Predicted (Test Period)')
ax.set_ylabel('Total Room Demand')
ax.legend()
plt.tight_layout()
plt.savefig('plots/05_baseline_forecast_comparison.png')
plt.close()

# ------------------------------------------------------------------
# 5. Feature importance (RF)
# ------------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X_train_raw.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
importances.plot(kind='barh', ax=ax, color='#55A868')
ax.invert_yaxis()
ax.set_title('Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig('plots/06_feature_importance.png')
plt.close()

# ------------------------------------------------------------------
# 6. Save results & models
# ------------------------------------------------------------------
results_df = pd.DataFrame(results).T
results_df.to_csv('models/baseline_results.csv')
joblib.dump(lr, 'models/linear_regression.pkl')
joblib.dump(rf, 'models/random_forest.pkl')

print("\n\nComparison table:")
print(results_df.round(3))
