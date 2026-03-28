sns.boxplot(x='success', y='run_block_count', data=user_df)
plt.title("Run Block vs Success")
plt.show()

sns.boxplot(x='success', y='agent_usage_count', data=user_df)
plt.title("Agent Usage vs Success")
plt.show()