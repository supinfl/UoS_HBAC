import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the npy file
# E:\PythonCode\Kaggle\TeamProject\UoS_HBAC\original_data\test_spectrograms\853520.npy

spec_id = 853520
spec = pd.read_parquet(f'./original_data/test_spectrograms/{spec_id}.parquet')
spec = spec.fillna(0).values[:, 1:].T # fill NaN values with 0, transpose for (Time, Freq) -> (Freq, Time)
spec = spec.astype("float32")
np.save(f"./original_data/test_spectrograms/{spec_id}.npy",spec)
npy_path = f"./original_data/test_spectrograms/{spec_id}.npy"
npy_data = np.load(npy_path)

# Display the shape of the npy array to understand its dimensions
npy_data_shape = npy_data.shape

npy_data_shape

# Plotting the first few rows of the data as line plots
plt.figure(figsize=(14, 8))

# Plotting the first 5 rows to avoid overcluttering
for i in range(5):
    plt.plot(npy_data[i], label=f'Row {i+1}')

plt.title('Line Plot of the First 5 Rows of the .npy Data')
plt.xlabel('Column Index')
plt.ylabel('Value')
plt.legend()
plt.show()
