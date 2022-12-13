import pandas as pd
import numpy as np
np.random.seed(0)

# Read data
df = pd.read_csv('regression-data.csv')

nodes_per_beam = df[df['sample'] == 0].shape[0]
dx = df[['dx']].to_numpy().flatten()
dy = df[['dy']].to_numpy().flatten()

# print(df.head())
# Select which columns to keep
noisy_df = df[['sample', 'location', 'node', 'x', 'y', 'dx', 'dy']].copy()

# Add noise to specified columns:
dX_noisy = dx + np.random.randn(*dx.shape) * 1.e-5
dY_noisy = dy + np.random.randn(*dy.shape) * 1.e-5
noisy_df['dx'] = dX_noisy
noisy_df['dy'] = dY_noisy

# Save data
# print(noisy_df.head())
noisy_df.to_csv('regression-data_realistic.csv', index=False)
