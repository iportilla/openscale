import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load('loan_approval_model.joblib')

# Load datasets
orig_df = pd.read_csv("German credit risk data set.csv")
drift_df = pd.read_csv("German_credit_drifted.csv")

def evaluate(df, label):
    X = df.drop('Risk', axis=1)
    y_true = df['Risk']
    y_pred = model.predict(X)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"\n--- Evaluation: {label} ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    return acc, y_pred

acc_orig, pred_orig = evaluate(orig_df, "Original Data (Reference)")
acc_drift, pred_drift = evaluate(drift_df, "Drifted Data (Year 2)")

# Compare prediction distributions
print("\n--- Prediction Distribution Shift ---")
print(f"Original Approval Rate: {(pd.Series(pred_orig) == 'No Risk').mean():.4f}")
print(f"Drifted Approval Rate:  {(pd.Series(pred_drift) == 'No Risk').mean():.4f}")

# Visualize Accuracy Drop
plt.figure(figsize=(8, 5))
plt.bar(['Original', 'Drifted'], [acc_orig, acc_drift], color=['green', 'red'])
plt.ylabel('Accuracy')
plt.title('Model Performance Degradation (Drift)')
plt.ylim(0, 1.0)
plt.savefig('drift_accuracy_comparison.png')
print("\nSaved drift_accuracy_comparison.png")

print("\nDrift Analysis Completed.")
