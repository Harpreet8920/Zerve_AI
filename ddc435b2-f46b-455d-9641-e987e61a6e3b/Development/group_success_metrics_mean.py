user_df.groupby('success')[[
    'total_events',
    'active_days',
    'run_block_count',
    'agent_usage_count',
    'credits_used_total',
    'unique_tools_used'
]].mean()