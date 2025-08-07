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
        df2 = df_i.groupby('stay_id')['value'].apply(list).apply(extract_stats).reset_index()
        df2.columns = ['stay_id'] + [f'{c}_{col}' for col in df2.columns[1:]]
        
        if df_grouped is None:
            df_grouped = df2
        else:
            df_grouped = df_grouped.merge(df2, on='stay_id', how='outer')
        print(len(df_grouped))
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

    stat_codes = list(set([(x.split('_', 1)[0]) for x in stat_features]))
    count_codes = list(set([(x.split('_')[0]) for x in count_features]))
    return stat_codes, count_codes

for method in ['median']:
    for task in ['transfer']:
        for model_name in ['xgb', 'rf']:
            for i in range(1, 5):
                metrics = { 'acc': [], 'auc': [], 'f1': [], 'precision': [], 'recall': [], 'auprc': []}
                
                if model_name == 'xgb':
                    from xgboost import XGBClassifier
                    model = XGBClassifier()
                    model.load_model(f'results/{task}/{model_name}_{task}_{method}_{i}.json')
                else:
                    model = joblib.load(f'results/{task}/{model_name}_{task}_{method}_{i}.pkl')

                imputer = joblib.load(f'results/{task}/{model_name}_{task}_{method}_{i}_imputer.pkl')
                stat_codes, count_codes = get_model_features(model)

                label_ids = pd.read_csv(f'results/{task}/{model_name}_{task}_{method}_{i}.csv', usecols=['stay_id'])
                label_df = pd.read_csv(f"data/transfer/label.csv", usecols=['stay_id', 'intime', 'Time', 'Label'])
                demo_test = pd.read_csv(f"data/transfer/demo.csv", usecols=['age', 'race_ind', 'gender_ind', 'stay_id'])
                # demo_test['stay_id'] = pd.to_numeric(demo_test.index)

                df = pd.read_csv('data/transfer/numerics.csv', usecols=['stay_id', 'charttime', 'event', 'value'])
                df = df.rename(columns={'charttime': 'time'})

                df = df[(df['event'].isin(stat_codes))]

                label_df = label_df[label_df['stay_id'].isin(label_ids['stay_id'])]
                demo_test = demo_test[demo_test['stay_id'].isin(label_ids['stay_id'])]
                df = df[df['stay_id'].isin(label_ids['stay_id'])]
                df = df.merge(label_df[['stay_id', 'Time', 'intime']], on='stay_id')

                df['time'] = pd.to_datetime(df['time']) 
                df['intime'] = pd.to_datetime(df['intime']) 
                df['time'] = (df['time'] - df['intime']).dt.total_seconds() / 3600
                
                df = df[df['time'] <= (df['Time'])]
                df['time'] = df['Time'] - df['time']
                df = df.drop(columns=['Time', 'intime'])

                TIMES = np.arange(3.0, 9.0, 0.1)

                for n, TIME in enumerate(TIMES):
                    df_i = df[df['time'] >= TIME]
                    if n > 0:
                        valid_ids = df_i[(df_i['time'] >= TIMES[n-1]) & (df_i['time'] <= TIME)]['stay_id'].unique()
                        df_i = df_i[df_i['stay_id'].isin(valid_ids)]
                    df_i = df_i.drop(columns=['time'])

                    # df_count = run_count(df_i, count_codes)
                    data = run_stats(df_i, stat_codes)
                    # data = df_count.merge(df_stat, on='stay_id', how='outer')

                    # count_columns = [col for col in data.columns if '_count' in col]
                    # data[count_columns] = data[count_columns].fillna(0)

                    data = data.merge(label_df[['stay_id', 'Label']], on='stay_id')
                    data = data.merge(demo_test, on='stay_id')

                    X = data.drop(columns=['Label', 'stay_id'])
                    y = data[['Label']]
                    ids = data[['stay_id']]
                    print(data['Label'].value_counts())

                    for col in model.feature_names_in_:
                        col = str(col)
                        if col not in X.columns:
                            X[col] = np.nan
                    if len(X) == 0: continue
                    X = X[model.feature_names_in_]
                    X = impute(X, imputer)

                    predictions = model.predict(X)
                    probs = model.predict_proba(X)[:, 1]

                    ids[TIME] = probs
                    label_ids = label_ids.merge(ids, on='stay_id', how='outer')
                    # compute_results(predictions, probs, y, metrics)
 
                print(label_ids.iloc[:, 2:].diff(axis=1).abs().max(axis=1).max())
                #with open(f'results/{model_name}_{task}_{method}_{i}_smooth.pkl', 'wb') as f:
                #    pickle.dump(metrics, f)
                label_ids.to_csv(f'results/{model_name}_{task}_{method}_{i}_smooth.csv')