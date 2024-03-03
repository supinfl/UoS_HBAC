import pandas as pd

df = pd.read_csv('./original_data/train.csv')

for index, row in df.iterrows():
    eeg_id = row['eeg_id']
    spectrogram_id = row['spectrogram_id']
    
    eeg_file_name = f"{eeg_id}.parquet"
    spectrogram_file_name = f"{spectrogram_id}.parquet"
    
    eeg_file_path = f"./original_data/train_eegs/{eeg_file_name}"
    spectrogram_file_path = f"./original_data/train_eegs/{spectrogram_file_name}"

    eeg_df = pd.read_parquet(eeg_file_path)
    spectrogram_df = pd.read_parquet(spectrogram_file_path)

print(df.head())
# print(eeg_df.head())
# print(spectrogram_df.head())