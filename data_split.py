import pandas as pd

df = pd.read_csv('./original_data/train.csv')

class_distribution = df['expert_consensus'].value_counts()
print("Data distribution:\n", class_distribution)


min_samples = class_distribution.min()
desired_samples = 200

balanced_df = pd.DataFrame()
for label in df['expert_consensus'].unique():
    class_samples = df[df['expert_consensus'] == label].sample(desired_samples, replace=False)
    balanced_df = pd.concat([balanced_df, class_samples])

balanced_class_distribution = balanced_df['expert_consensus'].value_counts()
print("\nBalanced data distribution:\n", balanced_class_distribution)

balanced_df.to_csv('balanced_train.csv', index=False)
