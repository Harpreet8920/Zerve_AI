# Value counts of 'success' column — absolute counts and proportions
success_counts = user_df['success'].value_counts()
success_proportions = user_df['success'].value_counts(normalize=True)

print("Absolute Counts:")
print(success_counts)
print("\nProportions:")
print(success_proportions)