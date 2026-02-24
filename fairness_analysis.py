import pandas as pd
import joblib
import numpy as np

# Load the model and dataset
model_filename = 'loan_approval_model.joblib'
model = joblib.load(model_filename)

data_path = "German credit risk data set.csv"
df = pd.read_csv(data_path)

# 1. Get predictions for the entire dataset
X = df.drop('Risk', axis=1)
y_actual = df['Risk']
y_pred = model.predict(X)

# Add predictions back to the dataframe for analysis
df['PredictedRisk'] = y_pred
df['IsApproved'] = df['PredictedRisk'] == 'No Risk'

def calculate_fairness_metrics(df, group_col, group_name):
    print(f"\n--- Fairness Analysis for: {group_name} ---")
    groups = df.groupby(group_col)
    
    metrics = []
    for name, group in groups:
        predicted_approval_rate = group['IsApproved'].mean()
        actual_approval_rate = (group['Risk'] == 'No Risk').mean()
        total = len(group)
        metrics.append({'Group': name, 'PredictedRate': predicted_approval_rate, 'ActualRate': actual_approval_rate, 'Count': total})
        print(f"Group: {name:10} | Pred Approval: {predicted_approval_rate:.4f} | Actual Approval: {actual_approval_rate:.4f} | Count: {total}")
    
    # Calculate Disparate Impact (Ratio of approval rates)
    if len(metrics) == 2:
        # Sort by approval rate to find privileged vs unprivileged
        sorted_metrics = sorted(metrics, key=lambda x: x['PredictedRate'], reverse=True)
        privileged = sorted_metrics[0]
        unprivileged = sorted_metrics[1]
        
        di = unprivileged['PredictedRate'] / privileged['PredictedRate']
        print(f"\nDisparate Impact ({unprivileged['Group']} / {privileged['Group']}): {di:.4f}")
        
        if di < 0.8:
            print("WARNING: Technical Bias Detected (Disparate Impact < 0.8)")
        elif di > 1.25:
            print("WARNING: Technical Bias Detected (Disparate Impact > 1.25)")
        else:
            print("Fairness: Passed 80% rule (No significant disparate impact)")

# 2. Gender Fairness Analysis
calculate_fairness_metrics(df, 'Sex', 'Gender (Sex)')

# 3. Age Fairness Analysis (Young < 25)
df['AgeGroup'] = np.where(df['Age'] < 25, 'Young (<25)', 'Adult (>=25)')
calculate_fairness_metrics(df, 'AgeGroup', 'Age Group')

# 4. Intersectionality (Optional but interesting)
print("\n--- Intersectional Analysis (Sex + AgeGroup) ---")
intersectional = df.groupby(['Sex', 'AgeGroup'])['IsApproved'].mean().reset_index()
print(intersectional)

print("\nFairness Analysis Completed.")
