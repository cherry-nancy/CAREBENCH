import pandas as pd
import numpy as np
import joblib
from scipy.stats import linregress, kurtosis, skew
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, average_precision_score
import pickle 

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

def run_stats(df, codes):
    df_grouped = None
    for c in codes:
        df_i = df[df['code'] == c].copy()

        df_i['value'] = pd.to_numeric(df_i['value'], 'coerce')
        df_i = df_i.dropna()
        df2 = df_i.groupby('patient_id')['value'].apply(list).apply(extract_stats).reset_index()
        df2.columns = ['patient_id'] + [f'{c}_{col}' for col in df2.columns[1:]]
        
        if df_grouped is None:
            df_grouped = df2
        else:
            df_grouped = df_grouped.merge(df2, on='patient_id', how='outer')
    return df_grouped

def run_count(df, codes):
    df_grouped = None
    for c in codes:
        df_i = df[df['code'] == c].copy()
        df2 = df_i.groupby('patient_id')['code'].count().rename(f'{c}_count').reset_index()
        
        if df_grouped is None:
            df_grouped = df2
        else:
            df_grouped = df_grouped.merge(df2, on='patient_id', how='outer')
    count_columns = [col for col in df_grouped.columns if '_count' in col]
    df_grouped[count_columns] = df_grouped[count_columns].fillna(0)
    return df_grouped

def impute(X_test, imputer):
    features = X_test.columns
    X_test = imputer.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=features)
    return X_test

def compute_results(y_pred, y_pred_prob, y_test, metrics):
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    auc = roc_auc_score(y_test, y_pred_prob)
    auprc = average_precision_score(y_test, y_pred_prob)

    # Save metrics
    metrics['acc'].append(acc)
    metrics['auc'].append(auc)
    metrics['f1'].append(f1)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['auprc'].append(auprc)

def get_model_features(model):
    features = model.feature_names_in_
    features = list(filter(lambda x: '/' in x, features))

    count_features = list(filter(lambda x: 'count' in x, features))
    stat_features = list(filter(lambda x: not 'count' in x, features))

    stat_codes = []
    for x in stat_features:
        c = '_'.join(x.split('_')[:-1])
        if 'qr' in c: c = c[:-3]
        stat_codes.append(c)
    stat_codes = list(set(stat_codes))
    count_codes = list(set(['_'.join(x.split('_')[:-1]) for x in count_features]))
    return stat_codes, count_codes

for method in ['median']: # 
    for task in ['hypoglycemia']:
        for model_name in ['rf', 'xgb']:
            for i in range(0, 5):
                task_prefix = 'guo_' if task == 'icu' else 'lab_'
                metrics = { 'acc': [], 'auc': [], 'f1': [], 'precision': [], 'recall': [], 'auprc': []}
                
                if model_name == 'xgb':
                    from xgboost import XGBClassifier
                    model = XGBClassifier()
                    model.load_model(f'results/{task}/{model_name}_{task}_{method}_{i}.json')
                else:
                    model = joblib.load(f'results/{task}/{model_name}_{task}_{method}_{i}.pkl')

                imputer = joblib.load(f'results/{task}/{model_name}_{task}_{method}_{i}_imputer.pkl')
                stat_codes, count_codes = get_model_features(model)

                label_ids = pd.read_csv(f'results/{task}/{model_name}_{task}_{method}_{i}.csv')
                label_ids = label_ids[['patient_id']]
                df = pd.read_parquet(f'data/{task_prefix}{task}/{task}.parquet', columns=['patient_id', 'start', 'code', 'value'])
                label_df = pd.read_parquet(f'data/{task_prefix}{task}/{task}_label.parquet', columns=['patient_id', 'start', 'Time', 'value'])
                label_df = label_df.rename(columns={'start':'start_time'})
                demo_test = pd.read_parquet(f'data/{task_prefix}{task}/{task}_demo.parquet', columns=['patient_id', 'age', 'gender_ind', 'ethnicity_ind', 'race_ind'])

                df = df[df['patient_id'].isin(label_ids['patient_id'])]
                label_df = label_df[label_df['patient_id'].isin(label_ids['patient_id'])]
                demo_test = demo_test[demo_test['patient_id'].isin(label_ids['patient_id'])]

                df = df.merge(label_df[['patient_id', 'Time', 'start_time']], on='patient_id')
                df['start'] = pd.to_datetime(df['start'])
                df['Time'] = pd.to_datetime(df['Time'])
                df = df[df['start'] >= df['start_time']]
                df = df[df['start'] <= df['Time']]

                df['time'] = (df['Time'] - df['start']).dt.total_seconds() / 3600
                df = df.drop(columns=['Time', 'start'])

                df = df[(df['code'].isin(stat_codes)) | (df['code'].isin(count_codes))]
                TIMES = df[(df['time'] <= 4.0) & (df['time'] >= 0.5)].sort_values('time')['time'].unique()

                for n, TIME in enumerate(TIMES):
                    df_i = df[df['time'] >= TIME]
                    if n > 0:
                        valid_ids = df[(df['time'] > TIMES[n-1]) & (df['time'] <= TIME)]['patient_id'].unique()
                        df_i = df_i[df_i['patient_id'].isin(valid_ids)]
                    df_i = df_i.drop(columns=['time'])

                    if len(count_codes) > 0:
                        df_count = run_count(df_i, count_codes)
                        df_stat = run_stats(df_i, stat_codes)
                        data = df_count.merge(df_stat, on='patient_id', how='outer')
                    else:
                        data = run_stats(df_i, stat_codes)
                    data = (df_i[['patient_id']].drop_duplicates()).merge(data, on='patient_id', how='outer')

                    count_columns = [col for col in data.columns if '_count' in col]
                    data[count_columns] = data[count_columns].fillna(0)

                    data = data.merge(label_df[['patient_id', 'value']], on='patient_id', how='left')
                    data = data.merge(demo_test, on='patient_id', how='left')

                    X = data.drop(columns=['value', 'patient_id'])
                    y = data[['value']]
                    ids = data[['patient_id']]
                    print(data['value'].value_counts())

                    for col in model.feature_names_in_:
                        col = str(col)
                        if col not in X.columns:
                            X[col] = 0 if col.endswith('count') else np.nan
                    X = X[model.feature_names_in_]
                    X = impute(X, imputer)

                    predictions = model.predict(X)
                    probs = model.predict_proba(X)[:, 1]

                    ids[TIME] = probs
                    label_ids = label_ids.merge(ids, on='patient_id', how='outer')
                    # compute_results(predictions, probs, y, metrics)
 
                print(label_ids.iloc[:, 2:].diff(axis=1).abs().max(axis=1).max())
                #with open(f'results/{model_name}_{task}_{method}_{i}_smooth.pkl', 'wb') as f:
                #    pickle.dump(metrics, f)
                label_ids.to_csv(f'results/{model_name}_{task}_{method}_{i}_smooth.csv', index=False)