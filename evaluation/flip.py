import pandas as pd

def count_flips(row):
    # Drop NaNs, convert to 0/1 based on threshold
    rounded = row.dropna().apply(lambda x: 1 if x >= 0.5 else 0)
    # Compute differences and count changes
    return (rounded.diff().fillna(0) != 0).sum()


for method in ['median']: # 
    for task in ['hypoglycemia']:
        for model_name in ['xgb', 'rf']:
            i = 0
            df = pd.read_csv(f'results/{model_name}_{task}_{method}_{i}_smooth.csv')

            # Apply to each row
            df['flip_count'] = df.apply(count_flips, axis=1)

            print(df['flip_count'].mean())