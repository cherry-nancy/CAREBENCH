import numpy as np
import pandas as pd 
import statistics

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

# --- Step 4: Compute Estimated Calibration Index (ECI) ---
def compute_eci(y_true, y_proba, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    eci = np.sqrt(np.mean((prob_true - prob_pred) ** 2))
    return eci

cal = 'platt'
for task in ['B', 'mcmed', 'last']:
    for model in ['gpt2', 'mamba']:
        for method in ['binary']:
            ECI = []
            for i in [1234, 3177, 5848, 8725, 9175]:
                if method == 'binary' or method == 'linear' or method == 'nesy':
                    y_proba_raw = pd.read_csv(f'output/done/{task}-{method}/{task}_{method}_{model}_{i}_smooth.csv', usecols=['pat_enc_csn_id', 'probs'])
                elif method == 'llm':
                    y_proba_raw = pd.read_csv(f'output/qwen_penn_llm_0_smooth.csv')
                    y_proba_raw = y_proba_raw.dropna()
                else:
                    y_proba_raw = pd.read_csv(f'output/done/{task}/{model}_{task}_{method}_{i}_smooth.csv', usecols=['pat_enc_csn_id', 'probs'])
                
                y_proba_raw['pat_enc_csn_id'] = y_proba_raw['pat_enc_csn_id'].astype(str)

                if task == 'sepsis' or task == 'last':
                    y_true = pd.read_parquet(f'data/sepsis/sepsis_label.parquet', columns=['pat_enc_csn_id', 'sepsis_flag'])
                    y_true['Label'] = y_true['sepsis_flag']
                elif task == 'B':
                    y_true = pd.read_parquet(f'data/physionet/visits_{task}.parquet', columns=['ID', 'Label'])
                    y_true['pat_enc_csn_id'] = y_true['ID'].apply(lambda x: x[1:])
                elif task == 'mcmed':
                    y_true = pd.read_parquet(f'data/{task}/eSOFA.parquet', columns=['CSN', 'Label'])
                    y_true['pat_enc_csn_id'] = y_true['CSN'].astype(str)

                y = y_proba_raw.merge(y_true, on='pat_enc_csn_id')
                y_proba_raw = y[['probs']].values
                y_true = y['Label'].values.ravel()

                # --- Step 2: Split for calibration (Stratified) ---
                X_cal, X_test, y_cal, y_test = train_test_split(
                    y_proba_raw, y_true, test_size=0.5, stratify=y_true, random_state=42
                )

                if cal == 'platt':
                    # --- Step 3: Platt Calibration (Logistic Regression) ---
                    platt = LogisticRegression(solver='lbfgs')
                    platt.fit(X_cal, y_cal)
                    y_proba_calibrated = platt.predict_proba(y_proba_raw)[:, 1]

                elif cal == 'istonic':
                    iso_reg = IsotonicRegression(out_of_bounds='clip')
                    iso_reg.fit(X_cal.ravel(), y_cal)
                    y_proba_calibrated = iso_reg.predict(y_proba_raw.ravel())                    

                elif cal == 'temp':
                    from scipy.special import logit, expit
                    from scipy.optimize import minimize

                    # Convert probabilities to logits (inverse sigmoid)
                    logits_cal = logit(np.clip(X_cal.ravel(), 1e-15, 1 - 1e-15))
                    logits_all = logit(np.clip(y_proba_raw.ravel(), 1e-15, 1 - 1e-15))

                    # Define negative log-likelihood loss function
                    def nll_loss(temp):
                        probs = expit(logits_cal / temp)
                        return -np.mean(y_cal * np.log(probs + 1e-15) + (1 - y_cal) * np.log(1 - probs + 1e-15))

                    # Optimize temperature
                    res = minimize(nll_loss, x0=[1.0], bounds=[(0.05, 10)])
                    optimal_temp = res.x[0]

                    # Apply calibrated temperature to all logits
                    y_proba_calibrated = expit(logits_all / optimal_temp)
                        
                eci = compute_eci(y_true, y_proba_calibrated)
                ECI.append(eci)
            
            print(method, model, task)
            average = statistics.mean(ECI)
            print(f"Average: {average}")

            if len(ECI) > 1:
                std_dev = statistics.stdev(ECI)
                print(f"Standard Deviation: {std_dev}")