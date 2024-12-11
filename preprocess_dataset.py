import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/NIFTY 50_minute_2015-2024.csv")
df['avg'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
df['prev_avg'] = df['avg'].shift(1)
df_avg = df[['prev_avg', 'avg']].copy()
df_avg['change'] = ((df_avg['avg'] - df_avg['prev_avg'])/df_avg['prev_avg']) * 100

# Create buckets for positive and negative changes separately
positive_mask = df_avg['change'] >= 0
negative_mask = df_avg['change'] < 0

# Initialize bucket column with NaN
df_avg['buckets'] = pd.NA

# Handle positive changes
if positive_mask.any():
    positive_changes = df_avg.loc[positive_mask, 'change']
    n_pos = len(positive_changes)
    pos_labels = [f'u{i+1}' for i in range(100)]
    pos_bins = pd.qcut(positive_changes, q=100, labels=pos_labels)
    df_avg.loc[positive_mask, 'buckets'] = pos_bins

# Handle negative changes
if negative_mask.any():
    negative_changes = df_avg.loc[negative_mask, 'change']
    n_neg = len(negative_changes)
    neg_labels = [f'd{i+1}' for i in range(100)]
    neg_bins = pd.qcut(negative_changes, q=100, labels=neg_labels)
    df_avg.loc[negative_mask, 'buckets'] = neg_bins

df_avg.iloc[1:, :].to_csv("data/bucket_data.csv")