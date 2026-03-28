features = [
    'total_events',
    'active_days',
    'total_sessions',
    'events_per_day',
    'run_block_count',
    'agent_usage_count',
    'blocks_created',
    'credits_used_total',
    'unique_tools_used',
    'events_per_session'
]

X = user_df[features]
y = user_df['success']

print(f"Feature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")
print(f"\nFeatures selected: {features}")
print(f"\nClass distribution:\n{y.value_counts()}")
print(f"\nMissing values per feature:\n{X.isnull().sum()}")
