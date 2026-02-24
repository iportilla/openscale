import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the dataset
data_path = "/Users/ivanp/Downloads/openscale/German credit risk data set.csv"
df = pd.read_csv(data_path)

# 1. Basic Information
print("--- Dataset Info ---")
print(df.info())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Summary Statistics ---")
print(df.describe())

# 2. Target Variable Distribution
print("\n--- Target Variable 'Risk' Distribution ---")
print(df['Risk'].value_counts(normalize=True))

plt.figure(figsize=(6, 4))
sns.countplot(x='Risk', data=df)
plt.title('Distribution of Risk')
plt.savefig('risk_distribution.png')
print("Saved risk_distribution.png")

# 3. Correlation (for numerical features)
plt.figure(figsize=(10, 8))
numerical_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
print("Saved correlation_heatmap.png")

# 4. Boxplot for key features by Risk
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Risk', y='LoanAmount', data=df)
plt.title('Loan Amount vs Risk')

plt.subplot(1, 2, 2)
sns.boxplot(x='Risk', y='LoanDuration', data=df)
plt.title('Loan Duration vs Risk')
plt.savefig('boxplots_risk.png')
print("Saved boxplots_risk.png")

print("\nEDA Completed.")
