user_df = df.groupby('distinct_id').agg({
    'event':'count',
    'timestamp':['min','max']
}).reset_index()