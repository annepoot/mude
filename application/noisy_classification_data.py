import pandas as pd
import numpy as np
np.random.seed(1)

df = pd.read_csv('classification-data.csv')
df = df[df['deteriorations'].isin([0,1,2,3,16,17,18,19])]

nodes_per_beam = df[df['sample'] == 0].shape[0]
x = df[['dy']].to_numpy().flatten()
X_no_noise = np.reshape(x.T, (-1, nodes_per_beam))  # Shape: [Num_samples x num_features]

print(df.head())
noisy_df = df[['sample', 'node', 'x', 'y', 'dy', 'intervention']].copy()

# Add noise:
X_noisy = x + np.random.randn(*x.shape) * 1.5e-4
noisy_df['dy'] = X_noisy

print(noisy_df.head())
noisy_df.to_csv('classification-data_realistic.csv', index=False)
