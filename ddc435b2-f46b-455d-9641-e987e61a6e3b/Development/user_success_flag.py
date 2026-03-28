user_df['success'] = (
    (user_df['active_days'] >= 7) &
    (user_df['run_block_count'] >= 5) &
    (user_df['agent_usage_count'] >= 3) &
    (user_df['credits_used_total'] > 0) 
).astype(int)