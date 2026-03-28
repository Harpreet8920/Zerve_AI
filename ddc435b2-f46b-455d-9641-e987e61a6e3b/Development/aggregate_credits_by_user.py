credits = df.groupby('distinct_id')['prop_credits_used'].sum()
user_df['credits_used_total'] = user_df['distinct_id'].map(credits).fillna(0)