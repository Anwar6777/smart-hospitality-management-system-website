"""
Module 3: Room Demand Forecasting
Step 6: Deep Learning Model (LSTM sequence-to-one forecaster)

>>> RUN THIS IN GOOGLE COLAB OR A LOCAL ENVIRONMENT WITH TENSORFLOW INSTALLED <<<
(pip install tensorflow)

Unlike the baseline models (which use flattened lag features), this LSTM
consumes a genuine sliding window of the last N days as a sequence, letting
it learn temporal patterns directly rather than through hand-crafted lags.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

tf.random.set_seed(42)
np.random.seed(42)

WINDOW_SIZE = 14   # use the last 14 days to predict the next day

# ------------------------------------------------------------------
# 1. Load the daily demand series (built in 01_build_timeseries.py)
# ------------------------------------------------------------------
df = pd.read_csv('data/daily_demand.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# also bring in calendar features for a multivariate window
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

feature_cols = ['Total', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
data = df[feature_cols].values

# ------------------------------------------------------------------
# 2. Chronological split BEFORE scaling (fit scaler on train only)
# ------------------------------------------------------------------
split_idx = int(len(df) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx - WINDOW_SIZE:]   # include lookback context for first test window

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# ------------------------------------------------------------------
# 3. Build sliding windows: X = [t-14..t-1] all features, y = Total at t
# ------------------------------------------------------------------
def make_windows(arr, window_size, target_col_idx=0):
    X, y = [], []
    for i in range(window_size, len(arr)):
        X.append(arr[i-window_size:i])
        y.append(arr[i, target_col_idx])
    return np.array(X), np.array(y)

X_train, y_train = make_windows(train_scaled, WINDOW_SIZE)
X_test, y_test = make_windows(test_scaled, WINDOW_SIZE)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ------------------------------------------------------------------
# 4. Build the LSTM
# ------------------------------------------------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, len(feature_cols))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)   # regression output: scaled demand
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

# ------------------------------------------------------------------
# 5. Train
# ------------------------------------------------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=150,
    batch_size=16,
    callbacks=callbacks,
    verbose=1,
    shuffle=False   # preserve temporal order even within training
)

# ------------------------------------------------------------------
# 6. Predict & inverse-transform back to actual room counts
# ------------------------------------------------------------------
y_pred_scaled = model.predict(X_test).ravel()

# inverse transform: reconstruct full feature vector shape for the scaler
def inverse_target(scaled_target, scaler, target_col_idx=0, n_features=5):
    dummy = np.zeros((len(scaled_target), n_features))
    dummy[:, target_col_idx] = scaled_target
    return scaler.inverse_transform(dummy)[:, target_col_idx]

y_pred = inverse_target(y_pred_scaled, scaler, n_features=len(feature_cols))
y_true = inverse_target(y_test, scaler, n_features=len(feature_cols))

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
r2 = r2_score(y_true, y_pred)

print("\n--- LSTM Test Set Performance (original scale) ---")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAPE: {mape:.3f}%")
print(f"R2:   {r2:.3f}")

# ------------------------------------------------------------------
# 7. Plot actual vs predicted
# ------------------------------------------------------------------
test_dates = df['date'].iloc[split_idx:split_idx + len(y_true)].values

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(test_dates, y_true, label='Actual', color='black', linewidth=1.5)
ax.plot(test_dates, y_pred, label='LSTM Prediction', color='#8172B3', alpha=0.8)
ax.set_title('LSTM Room Demand Forecast: Actual vs Predicted (Test Period)')
ax.set_ylabel('Total Room Demand')
ax.legend()
plt.tight_layout()
plt.savefig('plots/07_lstm_forecast.png')
plt.close()

# ------------------------------------------------------------------
# 8. Training curves
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(history.history['loss'], label='Train Loss')
ax.plot(history.history['val_loss'], label='Val Loss')
ax.set_title('LSTM Training Loss (MSE, scaled)')
ax.legend()
plt.tight_layout()
plt.savefig('plots/08_lstm_training_curve.png')
plt.close()

# ------------------------------------------------------------------
# 9. Save model & scaler
# ------------------------------------------------------------------
model.save('models/lstm_demand_model.keras')
import joblib
joblib.dump(scaler, 'models/lstm_scaler.pkl')
print("\nModel saved -> models/lstm_demand_model.keras")
