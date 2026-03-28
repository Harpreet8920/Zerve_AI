X = X.replace([np.inf, -np.inf], 0)
X = X.fillna(0)