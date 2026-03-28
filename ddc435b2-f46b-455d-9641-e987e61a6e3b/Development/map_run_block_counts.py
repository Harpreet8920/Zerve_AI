run_block = df[df['event'] == 'run_block'].groupby('distinct_id').size()
user_df['run_block_count'] = user_df['distinct_id'].map(run_block).fillna(0)