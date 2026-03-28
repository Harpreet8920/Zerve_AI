import pandas as pd

# The timestamp column has mixed ISO8601 formats (some with microseconds, some without).
# Using format='ISO8601' handles all valid ISO8601 variants including timezone offsets.
df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')

print(f"Timestamp dtype: {df['timestamp'].dtype}")
print(f"Sample values:\n{df['timestamp'].head()}")
print(f"\nDate range: {df['timestamp'].min()} → {df['timestamp'].max()}")

df['prop_credits_used'] = df['prop_credits_used'].fillna(0)
