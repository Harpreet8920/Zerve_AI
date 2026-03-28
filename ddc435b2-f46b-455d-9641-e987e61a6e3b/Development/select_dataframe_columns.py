cols = [
'distinct_id',
'event',
'timestamp',
'prop_session_id',
'prop_tool_name',
'prop_credits_used',
'prop_surface',
'prop_credit_amount',
'prop_$is_identified'
]

df = df[cols]
print(df)