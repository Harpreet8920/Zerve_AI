corr = user_df.corr(numeric_only=True)
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.show()