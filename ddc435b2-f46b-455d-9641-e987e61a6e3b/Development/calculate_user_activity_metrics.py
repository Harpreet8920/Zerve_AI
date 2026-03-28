user_df['active_days'] = (user_df['last_event'] - user_df['first_event']).dt.days + 1
user_df['events_per_day'] = user_df['total_events'] / user_df['active_days']