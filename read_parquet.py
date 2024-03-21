import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime

# Function to write logs
def write_log(model_name, run_time):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"-------------------------\nTimestamp: {timestamp}\nModel: {model_name}\nRun Time: {run_time} seconds\n"
    with open('run_log.txt', 'a') as file:
        file.write(log_message)

# Start the timer
start_time = time.time()

#Loading specific parquet file
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


# End the timer
end_time = time.time()
run_time = end_time - start_time

write_log("Baseline model", run_time)
# Print out the run time
print(f"The code ran for {run_time:.2f} seconds")
