import pandas as pd
import joblib

# Load the model
model_filename = 'loan_approval_model.joblib'
model = joblib.load(model_filename)

# Load some sample data from the original dataset
data_path = "German credit risk data set.csv"
df = pd.read_csv(data_path)

# Take 5 random samples
samples = df.sample(5, random_state=42)
X_samples = samples.drop('Risk', axis=1)
y_actual = samples['Risk']

# Make predictions
y_pred = model.predict(X_samples)

# Display results
print("--- Loan Approval Predictions on Random Samples ---")
for i in range(len(samples)):
    print(f"\nSample {i+1}:")
    print(f"  Actual Risk:    {y_actual.iloc[i]}")
    print(f"  Predicted Risk: {y_pred[i]}")
    status = "Correct" if y_actual.iloc[i] == y_pred[i] else "Incorrect"
    print(f"  Result:         {status}")

# Test with a manual "High Risk" case if possible, or just another set
print("\n--- Additional Samples ---")
samples_2 = df.iloc[10:15] # Rows 10 to 14
X_samples_2 = samples_2.drop('Risk', axis=1)
y_actual_2 = samples_2['Risk']
y_pred_2 = model.predict(X_samples_2)

for i in range(len(samples_2)):
    print(f"\nSample {i+6}:")
    print(f"  Actual Risk:    {y_actual_2.iloc[i]}")
    print(f"  Predicted Risk: {y_pred_2[i]}")
    status = "Correct" if y_actual_2.iloc[i] == y_pred_2[i] else "Incorrect"
    print(f"  Result:         {status}")
