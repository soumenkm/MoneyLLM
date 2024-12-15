import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("data/NIFTY 50_minute_2015-2024.csv")
df['avg'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
df['prev_avg'] = df['avg'].shift(1)
df_avg = df.copy()
df_avg['change'] = ((df_avg['avg'] - df_avg['prev_avg'])/df_avg['prev_avg']) * 100

# Bin the changes into 200 quantiles
df_avg['quantile_bin'] = pd.qcut(df_avg['change'], q=200, duplicates='drop')  # Handle duplicate bins if data doesn't allow full 200 bins

# If you want to access the quantile ranges
quantile_ranges = df_avg['quantile_bin'].unique()

# Optionally convert the bins to numeric indices
df_avg['trends'] = df_avg['quantile_bin'].cat.codes + 1  # Adding 1 to make the range 1 to 200
df_avg["quantile"] = pd.NA
df_avg['quantile'] = df_avg['quantile'].astype('object')
df_avg.at[0, "quantile"] = [(i.left.item(), i.right.item()) for i in quantile_ranges.sort_values().tolist()[:-1]]

# Load the original CSV
df = df_avg.iloc[0:, :].copy()

# Calculate the 12 technical indicators

# 1. Close-Open Difference
df['close_open_diff'] = df['close'] - df['open']

# 2. High-Low Range
df['high_low_range'] = df['high'] - df['low']

# 3. Midprice (High-Low Average)
df['midprice'] = (df['high'] + df['low']) / 2

# 4. Typical Price
df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

# 5. Log Return
df['log_return'] = (df['close'] / df['close'].shift(1)).apply(lambda x: 0 if pd.isna(x) else np.log(x))

# 6. Open-Close Return
df['open_close_return'] = (df['close'] - df['open']) / df['open']

# 7. Simple Moving Average (SMA) - 3 periods for demonstration
df['sma_100'] = df['close'].rolling(window=100).mean()

# 8. Exponential Moving Average (EMA) - 3 periods for demonstration
df['ema_100'] = df['close'].ewm(span=100, adjust=False).mean()

# 9. Average True Range (ATR)
prev_close = df['close'].shift(1).fillna(df['close'])
df['true_range'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(abs(df['high'] - prev_close), abs(df['low'] - prev_close))
)
df['atr'] = df['true_range'].rolling(window=20).mean()

# 10. Rolling Standard Deviation - 3 periods for demonstration
df['rolling_std'] = df['close'].rolling(window=20).std()

# 11. Rate of Change (ROC) - 3 periods for demonstration
df['roc_20'] = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20)) * 100

# 12. Relative Strength Index (RSI) - 3 periods for demonstration
delta = df['close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=20).mean()
avg_loss = loss.rolling(window=20).mean()
rs = avg_gain / avg_loss
df['rsi_20'] = 100 - (100 / (1 + rs))

# Save the updated dataframe to a new CSV
output_file_path = 'data/nifty_trend_data.pkl'
df.to_csv(output_file_path, index=True)
with open(output_file_path, "wb") as f:
    pickle.dump(df, f)