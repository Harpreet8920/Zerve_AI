tools = df.groupby('distinct_id')['prop_tool_name'].nunique()
user_df['unique_tools_used'] = user_df['distinct_id'].map(tools).fillna(0)