# Loan Approval Prediction - Model Quality Report

This document provides a comprehensive overview of the loan approval prediction model, its architecture, and performance metrics.

## Model Overview
The model is designed to classify loan applicants into "Risk" or "No Risk" categories based on financial and demographic data from the German Credit Risk dataset.

### Pipeline Architecture
The model uses a scikit-learn Pipeline that handles both preprocessing and classification in a single object.

```mermaid
graph TD
    A[Raw Data] --> B{ColumnTransformer}
    B --> C[Numerical Pipeline]
    B --> D[Categorical Pipeline]
    C --> E[StandardScaler]
    D --> F[OneHotEncoder]
    E --> G[Random Forest Classifier]
    F --> G
    G --> H[Risk Prediction]
```

## Performance Metrics

The model achieved an overall accuracy of **81.3%** on the hold-out test set (20% of the total data).

### Classification Report
| | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **No Risk** | 0.81 | 0.93 | 0.87 | 666 |
| **Risk** | 0.81 | 0.58 | 0.67 | 334 |
| **Accuracy** | | | **0.81** | 1000 |
| **Macro Avg** | 0.81 | 0.75 | 0.77 | 1000 |
| **Weighted Avg** | 0.81 | 0.81 | 0.80 | 1000 |

### Confusion Matrix
The confusion matrix shows the distribution of correct and incorrect predictions:

```mermaid
graph LR
    subgraph Actual_No_Risk
    A1[Correct: 620]
    A2[False Pos: 46]
    end
    subgraph Actual_Risk
    B1[False Neg: 141]
    B2[Correct: 193]
    end
```

| | Predicted No Risk | Predicted Risk |
|---|---|---|
| **Actual No Risk** | 620 | 46 |
| **Actual Risk** | 141 | 193 |

## Fairness Analysis
A fairness audit was conducted to investigate potential bias against protected groups (Gender and Age).

### Gender Fairness (Sex)
The model shows no significant bias against female applicants. In fact, female applicants have a slightly higher approval rate, which aligns with the training data.

- **Female Approval Rate**: 71.57%
- **Male Approval Rate**: 66.62%
- **Disparate Impact Ratio**: 0.93 (Passes 80% rule)

### Age Fairness
A significant disparity was detected between Age groups. Unexpectedly, young applicants (<25) have a much higher predicted approval rate than adults.

- **Young (<25) Approval Rate**: 94.55%
- **Adult (>=25) Approval Rate**: 63.07%
- **Disparate Impact Ratio**: 0.67 (**Fails 80% rule**)

> [!IMPORTANT]
> The bias against adults is inherent in the dataset. In the original data, 93.5% of applicants under 25 are labeled as "No Risk", compared to only 61.0% of adults. The model has learned this pattern accurately.

```mermaid
graph LR
    subgraph Gender_Fairness
    G1[Female: 71.6%]
    G2[Male: 66.6%]
    G3{FAIR}
    G1 --> G3
    G2 --> G3
    end
    subgraph Age_Fairness
    A1[Young: 94.6%]
    A2[Adult: 63.1%]
    A3{BIAS DETECTED}
    A1 --> A3
    A2 --> A3
    end
```

## Data Process Workflow
## Data Process Workflow
The following diagram illustrates the end-to-end data processing and modeling workflow:

```mermaid
sequenceDiagram
    participant D as Dataset
    participant P as Preprocessing
    participant M as Model (RF)
    participant E as Evaluation

    D->>P: Load Raw CSV
    Note right of P: Scaled numerical & encoded categorical
    P->>M: Train on 80% Data
    M->>E: Predict on 20% Data
    E->>E: Calculate Accuracy/F1
    Note left of E: 81.3% Accuracy achieved
```

## How to Reproduce
1.  **Exploration**: Run `eda_loan.py` to see data distributions.
2.  **Training**: Run `train_loan.py` to preprocess, train, and save the model.
3.  **Fairness**: Run `fairness_analysis.py` to see bias metrics.
4.  **Prediction**: Use `predict_loan.py` to run predictions on specific samples.

## Files Description
- `German credit risk data set.csv`: Original dataset.
- `eda_loan.py`: Exploratory Data Analysis script.
- `train_loan.py`: Main training and preprocessing pipeline.
- `fairness_analysis.py`: Fairness auditing script.
- `predict_loan.py`: Script for demonstrating model usage.
- `loan_approval_model.joblib`: Persisted model file.
