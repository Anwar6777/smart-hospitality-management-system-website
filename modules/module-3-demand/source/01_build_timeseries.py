"""
Module 3: Room Demand Forecasting
Step 1: Build the daily demand time series from booking-level data
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/hotel_booking_cleaned.csv')
print(f"Booking-level rows: {df.shape}")

# ------------------------------------------------------------------
# 1. Reconstruct the actual arrival date
# ------------------------------------------------------------------
month_map = {m: i+1 for i, m in enumerate(
    ['January','February','March','April','May','June','July',
     'August','September','October','November','December'])}
df['arrival_month_num'] = df['arrival_date_month'].map(month_map)

df['arrival_date'] = pd.to_datetime(dict(
    year=df['arrival_date_year'],
    month=df['arrival_month_num'],
    day=df['arrival_date_day_of_month']
), errors='coerce')

before = len(df)
df = df.dropna(subset=['arrival_date'])
print(f"Dropped {before - len(df)} rows with invalid dates")

# ------------------------------------------------------------------
# 2. Demand = actual arrivals only (exclude cancellations - a cancelled
#    booking never consumed a room on the arrival date)
# ------------------------------------------------------------------
actual_stays = df[df['is_canceled'] == 0].copy()
print(f"Actual (non-cancelled) stays: {len(actual_stays)}")

# ------------------------------------------------------------------
# 3. Aggregate to daily demand, per hotel type
# ------------------------------------------------------------------
daily_demand = (actual_stays.groupby(['arrival_date', 'hotel'])
                 .size()
                 .reset_index(name='room_demand'))

# pivot so each hotel type is its own series, then also build a combined total
pivot = daily_demand.pivot(index='arrival_date', columns='hotel', values='room_demand').fillna(0)
pivot['Total'] = pivot.sum(axis=1)
pivot = pivot.reset_index().sort_values('arrival_date')

# fill any missing calendar dates (days with zero arrivals) with 0
full_range = pd.date_range(pivot['arrival_date'].min(), pivot['arrival_date'].max(), freq='D')
pivot = pivot.set_index('arrival_date').reindex(full_range).fillna(0).rename_axis('date').reset_index()

print(f"\nDaily demand series shape: {pivot.shape}")
print(f"Date range: {pivot['date'].min()} to {pivot['date'].max()}")
print(pivot.head())

pivot.to_csv('data/daily_demand.csv', index=False)
print("\nSaved -> data/daily_demand.csv")
