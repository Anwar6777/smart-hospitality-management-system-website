"""
Module 3: Room Demand Forecasting
Step 6b: Improved LSTM — models the DETRENDED residual, not raw demand

>>> RUN THIS IN GOOGLE COLAB OR A LOCAL ENVIRONMENT WITH TENSORFLOW INSTALLED <<<

Why the first LSTM struggled: the daily demand series has a clear upward
trend across the ~2 years of data. A scaler fit on the training period alone
means the test period's demand sits at a systematically different level than
what the model learned - collapsing R2 even when relative day-to-day and
weekly patterns are captured correctly.

Fix: instead of predicting raw demand, we predict the RESIDUAL from a
trailing 14-day local average (a quantity that's naturally much more
stationary), then add that local average back at the end. The local average
uses only past days, so this is fully causal - no leakage.

We also shrink the network (single LSTM layer, more dropout) since ~600
training days is a small sample for a deep sequence model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

tf.random.set_seed(42)
np.random.seed(42)

WINDOW_SIZE = 14

# ------------------------------------------------------------------
# 1. Load data & build the trailing local average (causal - uses only
#    days strictly before t, so it's known and valid at prediction time)
# ------------------------------------------------------------------
df = pd.read_csv('data/daily_demand.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

df['local_avg_14'] = df['Total'].shift(1).rolling(14).mean()
df['residual'] = df['Total'] - df['local_avg_14']

df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df = df.dropna().reset_index(drop=True)   # drop the first 14 warm-up days
print(f"Usable rows after computing local average: {len(df)}")

feature_cols = ['residual', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
data = df[feature_cols].values

# ------------------------------------------------------------------
# 2. Chronological split, scale (fit on train only)
# ------------------------------------------------------------------
split_idx = int(len(df) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx - WINDOW_SIZE:]

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

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
# 3. Smaller network, more regularization (limited data)
# ------------------------------------------------------------------
model = Sequential([
    LSTM(32, input_shape=(WINDOW_SIZE, len(feature_cols)), kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(16, activation='relu', kernel_regularizer=l2(1e-4)),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=200,
    batch_size=16,
    callbacks=callbacks,
    verbose=1,
    shuffle=False
)

# ------------------------------------------------------------------
# 4. Predict residuals, inverse-transform, then ADD BACK the local average
# ------------------------------------------------------------------
y_pred_scaled = model.predict(X_test).ravel()

def inverse_target(scaled_target, scaler, n_features):
    dummy = np.zeros((len(scaled_target), n_features))
    dummy[:, 0] = scaled_target
    return scaler.inverse_transform(dummy)[:, 0]

residual_pred = inverse_target(y_pred_scaled, scaler, len(feature_cols))
residual_true = inverse_target(y_test, scaler, len(feature_cols))

# the local averages aligned with the test predictions
local_avgs_test = df['local_avg_14'].iloc[split_idx: split_idx + len(residual_pred)].values

y_pred = residual_pred + local_avgs_test
y_true = residual_true + local_avgs_test   # should equal original Total values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
r2 = r2_score(y_true, y_pred)

print("\n--- Improved LSTM (detrended residual) Test Performance ---")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAPE: {mape:.3f}%")
print(f"R2:   {r2:.3f}")

# ------------------------------------------------------------------
# 5. Plot
# ------------------------------------------------------------------
test_dates = df['date'].iloc[split_idx: split_idx + len(y_true)].values

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(test_dates, y_true, label='Actual', color='black', linewidth=1.5)
ax.plot(test_dates, y_pred, label='Improved LSTM Prediction', color='#8172B3', alpha=0.8)
ax.plot(test_dates, local_avgs_test, label='14-day Local Average (reference)', color='gray', linestyle='--', alpha=0.6)
ax.set_title('Improved LSTM Forecast (Detrended Residual Model)')
ax.set_ylabel('Total Room Demand')
ax.legend()
plt.tight_layout()
plt.savefig('plots/09_lstm_improved_forecast.png')
plt.close()

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(history.history['loss'], label='Train Loss')
ax.plot(history.history['val_loss'], label='Val Loss')
ax.set_title('Improved LSTM Training Loss (MSE, scaled residual)')
ax.legend()
plt.tight_layout()
plt.savefig('plots/10_lstm_improved_training_curve.png')
plt.close()

model.save('models/lstm_improved_demand_model.keras')
import joblib
joblib.dump(scaler, 'models/lstm_improved_scaler.pkl')
print("\nModel saved -> models/lstm_improved_demand_model.keras")
