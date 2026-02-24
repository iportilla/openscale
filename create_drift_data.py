import pandas as pd
import numpy as np

# Load original data
data_path = "German credit risk data set.csv"
df = pd.read_csv(data_path)

# Create a drifted version (Year 2)
# 1. Feature Drift: Loan amounts and durations increase due to inflation/economy
drifted_df = df.copy()
drifted_df['LoanAmount'] = (drifted_df['LoanAmount'] * 1.4).astype(int)
drifted_df['LoanDuration'] = (drifted_df['LoanDuration'] * 1.3).astype(int)

# 2. Concept Drift: Risk labels change. In a downturn, what was "No Risk" might now be "Risk"
# Let's randomly flip 20% of "No Risk" to "Risk" to simulate higher default rates
np.random.seed(42)
no_risk_indices = drifted_df[drifted_df['Risk'] == 'No Risk'].index
flip_indices = np.random.choice(no_risk_indices, size=int(len(no_risk_indices) * 0.25), replace=False)
drifted_df.loc[flip_indices, 'Risk'] = 'Risk'

# Save the drifted dataset
drifted_df.to_csv("German_credit_drifted.csv", index=False)

print("Drifted dataset created: German_credit_drifted.csv")
print(f"Original Risk Distribution:\n{df['Risk'].value_counts(normalize=True)}")
print(f"\nDrifted Risk Distribution:\n{drifted_df['Risk'].value_counts(normalize=True)}")
