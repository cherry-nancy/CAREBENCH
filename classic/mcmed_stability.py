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
        print(c)
        df_i = df[df['event'] == c].copy()

        df_i['value'] = pd.to_numeric(df_i['value'], 'coerce')
        df_i = df_i.dropna()
        df2 = df_i.groupby('CSN')['value'].apply(list).apply(extract_stats).reset_index()
        df2.columns = ['CSN'] + [f'{c}_{col}' for col in df2.columns[1:]]
        
        if df_grouped is None:
            df_grouped = df2
        else:
            df_grouped = df_grouped.merge(df2, on='CSN', how='outer')
        print(len(df_grouped))
    return df_grouped

def run_count(df, codes):
    df_grouped = None
    for c in codes:
        print(c)
        df_i = df[df['event'] == c].copy()
        df2 = df_i.groupby('CSN')['event'].count().rename(f'{c}_count').reset_index()
        
        if df_grouped is None:
            df_grouped = df2
        else:
            df_grouped = df_grouped.merge(df2, on='CSN', how='outer')
        print(len(df_grouped))
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
    features = list(filter(lambda x: not ('ind' in x or 'Age' in x), features))

    count_features = list(filter(lambda x: 'count' in x, features))
    stat_features = list(filter(lambda x: not 'count' in x, features))

    stat_codes = list(set([''.join(x.split('_')[0]) for x in stat_features]))
    count_codes = list(set([''.join(x.split('_')[0]) for x in count_features]))
    return stat_codes, count_codes

for method in ['median']:
    for task in ['decomp']:
        for model_name in ['xgb', 'rf']:
            if task == 'sepsis': fname = 'eSOFA'
            elif task == 'decomp': fname = 'decompensation'

            for i in range(0, 5):
                metrics = { 'acc': [], 'auc': [], 'f1': [], 'precision': [], 'recall': [], 'auprc': []}
                
                if model_name == 'xgb':
                    from xgboost import XGBClassifier
                    model = XGBClassifier()
                    model.load_model(f'results/{model_name}_{task}_{method}_{i}.json')
                else:
                    model = joblib.load(f'results/{model_name}_{task}_{method}_{i}.pkl')

                imputer = joblib.load(f'results/{model_name}_{task}_{method}_{i}_imputer.pkl')
                stat_codes, count_codes = get_model_features(model)

                label_ids = pd.read_csv(f'results/{model_name}_{task}_{method}_{i}.csv')
                df = pd.read_parquet(f'data/{fname}_data.parquet')
                if 'time_arrive' in df.columns:
                    df = df.rename(columns={'time_arrive': 'time'})
                label_df = pd.read_parquet(f'data/{fname}.parquet', columns=['CSN', 'Label', 'Trigger_time', 'Sample_time'])
                label_df['Time'] = label_df.apply(lambda x: x['Trigger_time'] if x['Label'] else x['Sample_time'], axis=1)
                demo_test = pd.read_parquet(f'data/{fname}_demo.parquet', columns=['CSN', 'Age', 'Gender_ind', 'Ethnicity_ind', 'Race_ind'])

                df = df[df['CSN'].isin(label_ids['CSN'])]
                label_df = label_df[label_df['CSN'].isin(label_ids['CSN'])]
                demo_test = demo_test[demo_test['CSN'].isin(label_ids['CSN'])]
                
                df = df[(df['event'].isin(stat_codes)) | (df['event'].isin(count_codes))]
                df = df.merge(label_df[['CSN', 'Time']], on='CSN')
                df = df[df['time'] <= df['Time']]
                df['time'] = (df['Time'] - df['time'])
                df = df.drop(columns=['Time'])

                TIMES = np.arange(0.1, 3.0, 0.05)
                #print(len(TIMES))

                for n, TIME in enumerate(np.arange(0.1, 3.0, 0.05)):
                    df_i = df[df['time'] >= TIME]
                    if n == 0:
                        valid_ids = df_i[(df_i['time'] >= TIME) & (df_i['time'] <= 1.5)]['CSN'].unique()
                    else:
                        valid_ids = df_i[(df_i['time'] >= TIMES[n-1]) & (df_i['time'] <= TIME)]['CSN'].unique()
                    df_i = df_i[df_i['CSN'].isin(valid_ids)]
                    df_i = df_i.drop(columns=['time'])

                    # df_count = run_count(df_i, count_codes)
                    df_stat = run_stats(df_i, stat_codes)
                    # data = df_count.merge(df_stat, on='CSN', how='outer')
                    data = df_stat
                    if len(data) == 0: 
                        print(TIME)
                        continue 

                    count_columns = [col for col in data.columns if '_count' in col]
                    data[count_columns] = data[count_columns].fillna(0)

                    data = data.merge(label_df[['CSN', 'Label']], on='CSN')
                    data = data.merge(demo_test, on='CSN')

                    X = data.drop(columns=['Label', 'CSN'])
                    assert(len(X) == df_i['CSN'].nunique())
                    y = data[['Label']]
                    ids = data[['CSN']]
                    print(data['Label'].value_counts())

                    for col in model.feature_names_in_:
                        col = str(col)
                        if col not in X.columns:
                            X[col] = 0 if col.endswith('count') else np.nan
                    X = X[model.feature_names_in_]
                    X = impute(X, imputer)

                    predictions = model.predict(X)
                    probs = model.predict_proba(X)[:, 1]

                    ids[TIME] = probs
                    label_ids = label_ids.merge(ids, on='CSN', how='outer')
                    compute_results(predictions, probs, y, metrics)
 
                print(label_ids.iloc[:, 2:].diff(axis=1).abs().max(axis=1).max())
                with open(f'results/{model_name}_{task}_{method}_{i}_smooth.pkl', 'wb') as f:
                    pickle.dump(metrics, f)
                label_ids.to_csv(f'results/{model_name}_{task}_{method}_{i}_smooth.csv')