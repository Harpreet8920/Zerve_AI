user_df = df.groupby('distinct_id').agg(
    total_events=('event', 'count'),
    first_event=('timestamp', 'min'),
    last_event=('timestamp', 'max'),
    total_sessions=('prop_session_id', 'nunique')
).reset_index()