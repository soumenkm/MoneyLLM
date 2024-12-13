import pandas as pd
import numpy as np

df = pd.read_csv("data/NIFTY 50_minute_2015-2024.csv")
df['avg'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
df['prev_avg'] = df['avg'].shift(1)
df_avg = df.copy()
df_avg['change'] = ((df_avg['avg'] - df_avg['prev_avg'])/df_avg['prev_avg']) * 100

# Create trends for positive and negative changes separately
positive_mask = df_avg['change'] >= 0
negative_mask = df_avg['change'] < 0

# Initialize bucket column with NaN
df_avg['trends'] = pd.NA

# Handle positive changes
if positive_mask.any():
    positive_changes = df_avg.loc[positive_mask, 'change']
    n_pos = len(positive_changes)
    pos_labels = [f'u{i+1}' for i in range(100)]
    pos_bins = pd.qcut(positive_changes, q=100, labels=pos_labels)
    df_avg.loc[positive_mask, 'trends'] = pos_bins

# Handle negative changes
if negative_mask.any():
    negative_changes = df_avg.loc[negative_mask, 'change']
    n_neg = len(negative_changes)
    neg_labels = [f'd{i+1}' for i in range(100)]
    neg_bins = pd.qcut(negative_changes, q=100, labels=neg_labels)
    df_avg.loc[negative_mask, 'trends'] = neg_bins

# Load the original CSV
df = df_avg.iloc[1:, :].copy()

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
output_file_path = 'data/nifty_trend_data.csv'
df.to_csv(output_file_path, index=True)