import pandas as pd

task = 'mortality'
model = 'other'

df = None
for i in range(60, 100, 10):
    df_i = pd.read_csv(f'{task}_stability/{model}_prob_{i}.csv')
    if df is None: df = df_i
    else: df = pd.concat([df, df_i])
df.to_csv(f'{task}_linear_gpt_prob.csv')