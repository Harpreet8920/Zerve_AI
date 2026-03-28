blocks = df[df['event'] == 'block_create'].groupby('distinct_id').size()
user_df['blocks_created'] = user_df['distinct_id'].map(blocks).fillna(0)