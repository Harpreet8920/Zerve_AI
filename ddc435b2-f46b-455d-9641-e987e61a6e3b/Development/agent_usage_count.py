agent_events = df[df['event'].str.contains('agent', na=False)]
agent_usage = agent_events.groupby('distinct_id').size()
user_df['agent_usage_count'] = user_df['distinct_id'].map(agent_usage).fillna(0)