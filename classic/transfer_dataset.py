import pandas as pd
import numpy as np
from scipy.stats import linregress, kurtosis, skew


def remove_rows(df, label_df, TIME):
    df['charttime'] = pd.to_datetime(df['charttime'])
    df = df.merge(label_df[['stay_id', 'intime', 'Time']], on='stay_id')
    df['charttime'] = (df['charttime'] - df['intime']).dt.total_seconds() / 3600
    
    df = df[df['Time'] >= TIME]
    df = df[df['charttime'] <= (df['Time'] - TIME)]
    df = df.drop(columns=['intime', 'charttime', 'Time'])
    return df

def transfer_loader(time):
    label = pd.read_csv(f"data/transfer/label.csv", usecols=['stay_id', 'intime', 'Label', 'Time'])
    label['intime'] = pd.to_datetime(label['intime'])

    demo = pd.read_csv(f"data/transfer/demo.csv", usecols=['stay_id', 'gender_ind', 'race_ind', 'age'])
    demo = demo[demo['stay_id'].isin(label['stay_id'])]

    vital_df = pd.read_csv(f"data/transfer/numerics.csv", usecols=['stay_id', 'event', 'value', 'charttime'])
    vital_df = remove_rows(vital_df, label, time)
    # vital_df = vital_df[:200]

    med_df = pd.read_csv(f"data/transfer/med.csv", usecols=['stay_id', 'group', 'charttime'])
    med_df = remove_rows(med_df, label, time)

    df_final = pd.concat([med_df[['stay_id']], vital_df[['stay_id']]]).drop_duplicates()
    for c in vital_df['event'].unique():
        df_i = vital_df[vital_df['event'] == c].copy()
        df_i['value'] = pd.to_numeric(df_i['value'], 'coerce')
        df_i = df_i.dropna()
        df2 = df_i.groupby('stay_id')['value'].apply(list).apply(extract_stats).reset_index()
        df2.columns = ['stay_id'] + [f'{c}_{col}' for col in df2.columns[1:]]
        df_final = df_final.merge(df2, on='stay_id', how='outer')

    med_df = med_df.groupby(['stay_id', 'group']).size().reset_index()
    meds = med_df.groupby('group')['stay_id'].nunique().sort_values(ascending=False).reset_index()['group'].unique()[:22]
    for m in meds:
        med_i = med_df[med_df['group']== m].drop(columns=['group'])
        med_i.columns = ['stay_id', f'{m}_count']
        df_final = df_final.merge(med_i, on='stay_id', how='outer')
    
    df_final = df_final.loc[:, df_final.isnull().mean() < 0.9]
    count_columns = [col for col in df_final.columns if '_count' in col]
    df_final[count_columns] = df_final[count_columns].fillna(0)

    df_final = df_final.merge(label[['stay_id', 'Label']], on='stay_id')
    df_final = df_final.merge(demo, on='stay_id')
    df_final = df_final.drop_duplicates()

    X = df_final.drop(columns=['Label', 'stay_id'])
    y = df_final[['Label']]
    ids = df_final[['stay_id']]
    print(df_final['Label'].value_counts())
    
    return X, y, ids


def extract_stats(xs):
    xs = np.array(xs, dtype=float)
    x = np.arange(len(xs))
    slope, _, _, _, _ = linregress(x, xs)

    return pd.Series({
        'min': xs.min(),
        'max': xs.max(),
        'mean': xs.mean(),
        'std': xs.std(),
        'median': np.median(xs),
        'skew': skew(xs),
        'kurtosis': kurtosis(xs),
        'slope': slope,
        'qr_25': np.quantile(xs, 0.25),
        'qr_75': np.quantile(xs, 0.75),
    })