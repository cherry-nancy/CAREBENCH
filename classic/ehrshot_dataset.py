import pandas as pd
import numpy as np
from scipy.stats import linregress, kurtosis, skew

TIME = 1.0
HYPERKALEMIA = ['LOINC/LG7931-1', 'LOINC/LP386618-5', 'LOINC/LG10990-6', 'LOINC/6298-4', 'LOINC/2823-3']
HYPOGLYCEMIA = ['SNOMED/33747003', 'LOINC/LP416145-3', 'LOINC/14749-6']

def hyperkalemia_loader():
    return lab_task_loader('hyperkalemia')

def hypoglycemia_loader():
    return lab_task_loader('hyloglycemia')

def remove_rows(df, label_df, time):
    df = df.merge(label_df[['patient_id', 'Time', 'start_time']], on='patient_id')

    df['start'] = (df['start'] - df['start_time']).dt.total_seconds() / 3600
    df['Time'] = (df['Time'] - df['start_time']).dt.total_seconds() / 3600

    df = df[df['Time'] >= time]
    df = df[df['start'] <= (df['Time'] - time)]
    df = df.drop(columns=['start', 'Time', 'start_time'])
    return df

def lab_task_loader(task_type, task_prefix='lab_'):
    data = pd.read_parquet(f"data/{task_prefix}{task_type}/{task_type}2.parquet", columns=['code', 'value', 'patient_id', 'start'])
    if task_type == 'hyperkalemia':
        data = data[~data['code'].isin(HYPERKALEMIA)]
    elif task_type == 'hypoglycemia':
        data = data[~data['code'].isin(HYPOGLYCEMIA)]

    demo = pd.read_parquet(f"data/{task_prefix}{task_type}/{task_type}_demo.parquet", columns=['patient_id', 'age', 'gender_ind', 'race_ind', 'ethnicity_ind'])
    label = pd.read_parquet(f"data/{task_prefix}{task_type}/{task_type}_label2.parquet")
    label['Time'] = label.apply(lambda x: x['prediction_time'] if x['value'] else x['Sample_time'], axis=1)
    label = label.rename(columns={'start': 'start_time'})

    if 'value_multi' in label.columns:
        label = label.drop(columns=['value_multi'])
    
    data = remove_rows(data, label, TIME)
    data = run_get_classic(data)
    data = data.drop_duplicates()

    data = data.merge(label[['patient_id', 'value']], on='patient_id')
    data = data.merge(demo, on='patient_id')

    X = data.drop(columns=['value', 'patient_id'])
    y = data[['value']]
    ids = data[['patient_id']]    
    return X, y, ids


def run_get_classic(df, start=0, end=30):
    codes = df[~df['code'].str.contains('CARE|Visit|Domain|Race|Ethnicity|Gender|CMS|OMOP|Medicare')]
    codes.groupby('code')['patient_id'].nunique().sort_values(ascending=False).reset_index()
    codes = codes['code'].unique()
    
    df_grouped = df[['patient_id']]
    df = df[df['code'].isin(codes)]
    for c in codes[start:end]:
        print(c)
        df_i = df[df['code'] == c].copy()
        unique_vals = df_i['value'].dropna().unique()

        if len(unique_vals) <= 1:
            df2 = df_i.groupby('patient_id')['code'].count().rename(f'{c}_count').reset_index()
        else:
            df_i['value'] = pd.to_numeric(df_i['value'], 'coerce')
            df_i = df_i.dropna()
            df2 = df_i.groupby('patient_id')['value'].apply(list).apply(extract_stats).reset_index()
            df2.columns = ['patient_id'] + [f'{c}_{col}' for col in df2.columns[1:]]
        
        if df_grouped is None:
            df_grouped = df2
        else:
            df_grouped = df_grouped.merge(df2, on='patient_id', how='outer')
        print(len(df_grouped))

    df_grouped = df_grouped.loc[:, df_grouped.isnull().mean() < 0.9]
    count_columns = [col for col in df_grouped.columns if '_count' in col]
    df_grouped[count_columns] = df_grouped[count_columns].fillna(0)

    return df_grouped

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