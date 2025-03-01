import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

st.set_page_config(layout='wide')
# Load dataset to get feature names
df = pd.read_csv("diabetes_dataset.csv")
feature_columns = df.drop(columns=["Outcome"]).columns

# Train models (If not already trained, otherwise load them from saved files)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
log_reg = LogisticRegression()
grad_boost = GradientBoostingClassifier()
log_reg.fit(X_scaled, y)
grad_boost.fit(X, y)  # No scaling needed

# Save models
joblib.dump(scaler, "scaler.pkl")
joblib.dump(log_reg, "log_reg.pkl")
joblib.dump(grad_boost, "grad_boost.pkl")

# Load models
def load_models():
    scaler = joblib.load("scaler.pkl")
    log_reg = joblib.load("log_reg.pkl")
    grad_boost = joblib.load("grad_boost.pkl")
    return scaler, log_reg, grad_boost

scaler, log_reg, grad_boost = load_models()

# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Enter your health parameters below, and the app will predict whether you have diabetes.")

show_hist = st.checkbox("Show Distribution of Every Variable", value=False) 
if show_hist:
    num_features = len(df.columns)
    rows = (num_features //4) +1 
    cols = min(4, num_features)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20,12))
    axes = axes.flatten()[:num_features]
    for i, col in enumerate(df.columns):
        df[col].hist(ax=axes[i], bins=30, color='blue')
        axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

show_corr = st.checkbox("Show Correlation Matrix", value=False)

if show_corr:
    st.subheader("Feature Correlation Matrix")

    # Compute correlation matrix
    corr_matrix = df.corr()

    

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,  
        cbar=True,
        ax=ax
    )

    st.pyplot(fig)


# User inputs
input_data = pd.DataFrame(
    np.nan,  # Fill with NaN initially
    index=[0],  # Single-row input
    columns=feature_columns
)

# Create an editable table
user_input_df = st.data_editor(input_data, num_rows="dynamic")

# Handle missing values (impute with median)
for col in feature_columns:
    user_input_df[col] = user_input_df[col].fillna(df[col].median())

# Standardize numerical data
input_scaled = scaler.transform(user_input_df)

prediction_btn = st.button('Press here to make the Prediction')
if prediction_btn:
# Make predictions
    log_reg_pred = log_reg.predict(input_scaled)[0]
    grad_boost_pred = grad_boost.predict(user_input_df)[0]
    
    # Display results
    st.subheader("Prediction Results:")
    st.write(f"**Logistic Regression Prediction:** {'Diabetic' if log_reg_pred else 'Non-Diabetic'}")
    st.write(f"**Gradient Boosting Prediction:** {'Diabetic' if grad_boost_pred else 'Non-Diabetic'}")
