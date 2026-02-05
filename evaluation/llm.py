import json
import pandas as pd

task = 'penn'

if task == 'penn':
    label_df = pd.read_parquet('data/sepsis/sepsis_label.parquet', columns=['pat_enc_csn_id', 'Time'])

elif task == 'physionet':
    label_df = pd.read_parquet('data/physionet/visits_B.parquet', columns=['ID', 'CriteriaTime'])
    label_df['pat_enc_csn_id'] = label_df['ID'].apply(lambda x: x[1:])

# Open and read the JSON file
with open(f'output/qwen32b.json', 'r') as file:
    data = json.load(file)

predictions = data['all_predictions']

all_probs = {}
all_times = {}
correct_labels = {}
for prediction in predictions:
    csn = prediction['CSN']
    if not csn in correct_labels.keys():
        correct_labels[csn] = prediction['TrueLabel']
    if csn in all_probs.keys():
        print(csn)
    else:
        all_probs[csn] = (prediction['TrueLabel'], prediction['PredictedLabel'], prediction['PredictedProb'])

rows = []
for key, tuples in all_probs.items():
    row = {'pat_enc_csn_id': key}
    label, prediction, probs = tuples
    row['probs'] = probs
    row['predict'] = prediction
    row['label'] = label
    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)
# df = df.drop(columns=[None])

# Save to CSV
df.to_csv(f'output/qwen_{task}_llm_0.csv', index=False)

print(df)

