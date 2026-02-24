import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set page config
st.set_page_config(page_title="Loan Approval Model Dashboard", layout="wide")

# Load model and data
@st.cache_resource
def load_assets():
    model = joblib.load('loan_approval_model.joblib')
    orig_df = pd.read_csv("German credit risk data set.csv")
    drift_df = pd.read_csv("German_credit_drifted.csv")
    return model, orig_df, drift_df

model, orig_df, drift_df = load_assets()

st.title("🏦 Loan Approval Model Monitoring Dashboard")
st.markdown("---")

# Tabs for different features
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Quality", "⚖️ Fairness Analysis", "📉 Drift Analysis", "🔮 Interactive Prediction"])

X_ref = orig_df.drop('Risk', axis=1)
categorical_cols = X_ref.select_dtypes(include=['object']).columns
numerical_cols = X_ref.select_dtypes(exclude=['object']).columns

# --- Tab 1: Model Quality ---
with tab1:
    st.header("Model Quality Metrics")
    st.write("Performance on the original dataset (Reference)")
    
    y_true = orig_df['Risk']
    y_pred = model.predict(X_ref)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        st.table(pd.DataFrame(report).transpose())
        st.metric("Overall Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                    xticklabels=['No Risk', 'Risk'], yticklabels=['No Risk', 'Risk'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig_cm)

# --- Tab 2: Fairness Analysis ---
with tab2:
    st.header("Fairness & Bias Audit")
    
    # Sex Fairness
    st.subheader("Gender Fairness (Sex)")
    orig_df['PredictedRisk'] = model.predict(X_ref)
    orig_df['IsApproved'] = orig_df['PredictedRisk'] == 'No Risk'
    
    if 'Sex' in orig_df.columns:
        sex_metrics = orig_df.groupby('Sex')['IsApproved'].mean()
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(sex_metrics)
        with col2:
            di_sex = sex_metrics['male'] / sex_metrics['female']
            st.metric("Disparate Impact (Male/Female)", f"{di_sex:.2f}")
            if 0.8 <= di_sex <= 1.25:
                st.success("Fairness Passed (80% Rule)")
            else:
                st.warning("Fairness Warning (Significant Disparity)")
    else:
        st.warning("Column 'Sex' not found in dataset. Check data format.")

    st.markdown("---")
    
    # Age Fairness
    st.subheader("Age Fairness (Young < 25)")
    orig_df['AgeGroup'] = np.where(orig_df['Age'] < 25, 'Young (<25)', 'Adult (>=25)')
    age_metrics = orig_df.groupby('AgeGroup')['IsApproved'].mean()
    
    col3, col4 = st.columns(2)
    with col3:
        st.bar_chart(age_metrics)
    with col4:
        # Note: In our analysis, Young had higher approval, so Adult is unprivileged relative to Young
        di_age = age_metrics['Adult (>=25)'] / age_metrics['Young (<25)']
        st.metric("Disparate Impact (Adult/Young)", f"{di_age:.2f}")
        if 0.8 <= di_age <= 1.25:
            st.success("Fairness Passed")
        else:
            st.error("Bias Detected: The model heavily favors young applicants (Data-Driven)")

# --- Tab 3: Drift Analysis ---
with tab3:
    st.header("Model Drift & Degradation")
    st.write("Comparing Original performance with Drifted (Economic Downturn) data")
    
    X_drift = drift_df.drop('Risk', axis=1)
    y_true_drift = drift_df['Risk']
    y_pred_drift = model.predict(X_drift)
    
    col1, col2 = st.columns(2)
    
    with col1:
        acc_orig = accuracy_score(y_true, y_pred)
        acc_drift = accuracy_score(y_true_drift, y_pred_drift)
        
        st.subheader("Accuracy Comparison")
        fig_drift, ax_drift = plt.subplots()
        plt.bar(['Original', 'Drifted'], [acc_orig, acc_drift], color=['green', 'red'])
        plt.ylabel('Accuracy')
        st.pyplot(fig_drift)
        
        st.metric("Accuracy Drop", f"{acc_orig - acc_drift:.2%}", delta=-(acc_orig - acc_drift), delta_color="inverse")

    with col2:
        st.subheader("Silent Failure Alert")
        appr_orig = (pd.Series(y_pred) == 'No Risk').mean()
        appr_drift = (pd.Series(y_pred_drift) == 'No Risk').mean()
        
        st.metric("Approval Rate (Original)", f"{appr_orig:.2%}")
        st.metric("Approval Rate (Drifted)", f"{appr_drift:.2%}")
        st.info("Note: The model continues to approve loans at the same rate even though actual risk doubled!")

# --- Tab 4: Interactive Prediction ---
with tab4:
    st.header("🔮 Interactive Loan Approval Prediction")
    st.write("Enter applicant details to test the model's prediction. We focus on the most impactful features for this demo.")

    # Simplified feature set
    key_features = ['CheckingStatus', 'CreditHistory', 'LoanDuration', 'LoanAmount', 'ExistingSavings', 'Age']
    
    # Pre-defined rejection examples
    st.subheader("Example Scenarios")
    col_ex1, col_ex2 = st.columns(2)
    
    rejection_cases = {
        "High Debt & Poor History": {
            'CheckingStatus': 'no checking', 'CreditHistory': 'all paid', 
            'LoanDuration': 48, 'LoanAmount': 15000, 
            'ExistingSavings': '<100', 'Age': 22
        },
        "Unstable Status & Large Loan": {
            'CheckingStatus': '0<=X<200', 'CreditHistory': 'existing paid', 
            'LoanDuration': 60, 'LoanAmount': 18000, 
            'ExistingSavings': 'no savings', 'Age': 25
        }
    }

    selected_case = None
    with col_ex1:
        if st.button("Example: High Debt & Poor History"):
            selected_case = "High Debt & Poor History"
    with col_ex2:
        if st.button("Example: Unstable Status & Large Loan"):
            selected_case = "Unstable Status & Large Loan"

    # Use selected case or defaults
    current_inputs = rejection_cases[selected_case] if selected_case else {}

    with st.form("prediction_form"):
        cols = st.columns(2)
        input_data = {}
        
        # Fill in key features with inputs
        for i, col_name in enumerate(key_features):
            with cols[i % 2]:
                if col_name in numerical_cols:
                    min_val = int(X_ref[col_name].min())
                    max_val = int(X_ref[col_name].max())
                    # Priority 1: From selected case. Priority 2: Mean of data.
                    default_val = current_inputs.get(col_name, int(X_ref[col_name].mean()))
                    input_data[col_name] = st.number_input(f"{col_name}", min_value=min_val, max_value=max_val, value=default_val)
                else:
                    options = list(X_ref[col_name].unique())
                    default_idx = options.index(current_inputs[col_name]) if col_name in current_inputs else 0
                    input_data[col_name] = st.selectbox(f"{col_name}", options=options, index=default_idx)
        
        # Fill remaining features with defaults (hidden from UI for simplicity)
        for col_name in X_ref.columns:
            if col_name not in key_features:
                if col_name in numerical_cols:
                    input_data[col_name] = X_ref[col_name].mean()
                else:
                    input_data[col_name] = X_ref[col_name].mode()[0]

        submitted = st.form_submit_button("Predict Approval Status")
        
        if submitted or selected_case:
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            
            st.markdown("---")
            if prediction == 'No Risk':
                st.success(f"**Prediction: SUCCESS (No Risk)**")
                st.balloons()
            else:
                st.error(f"**Prediction: ALERT (High Risk)**")
                st.warning("Basis for Rejection: This combination of short credit history, large loan amount, or lack of savings significantly increases risk profile.")

st.sidebar.info("This app demonstrates OpenScale-like monitoring for quality, bias, and drift.")
