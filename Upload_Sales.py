import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import csv
import chardet
from datetime import timedelta
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set page configuration for wide layout
st.set_page_config(layout="wide", page_title="Data Analysis Dashboard", page_icon="📊")

# Detect encoding of the file
def detect_encoding(file):
    raw_data = file.read(10000)
    result = chardet.detect(raw_data)
    file.seek(0)
    return result['encoding']

# Detect delimiter
def detect_delimiter(file):
    try:
        encoding = detect_encoding(file)
        sample = file.read(4096).decode(encoding)
        file.seek(0)
        sniffer = csv.Sniffer()
        return sniffer.sniff(sample).delimiter
    except csv.Error:
        return ','

# Load data
@st.cache_data
def load_data(file=None):
    try:
        if file is not None:
            if file.name.endswith('.csv'):
                delimiter = detect_delimiter(file)
                data = pd.read_csv(file, delimiter=delimiter)
            elif file.name.endswith('.xlsx'):
                xl = pd.ExcelFile(file)
                sheet_name = xl.sheet_names[0]
                data = xl.parse(sheet_name)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return None
        else:
            data = pd.read_csv("./retail_sales_dataset.csv")
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Sidebar for file upload and data category selection
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
data_category = st.sidebar.selectbox("Select Data Category", ["Select Category", "Sales", "Finance", "Marketing"])
data = load_data(uploaded_file) if uploaded_file else load_data()

if data is not None and data_category != "Select Category":
    st.title(f"📊 {data_category} Data Analysis and Prediction Dashboard")

    if data_category == "Sales":
        date_column, sales_column = None, None
        for col in data.columns:
            if "date" in col.lower() or "time" in col.lower():
                date_column = col
            if "sale" in col.lower() or "amount" in col.lower() or "price" in col.lower() or "total" in col.lower():
                sales_column = col

        if not date_column or not sales_column:
            st.error("Could not automatically detect the required columns (Date and Sales). Please check your dataset.")
        else:
            data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
            data[sales_column] = pd.to_numeric(data[sales_column], errors='coerce').fillna(0)
            
            if data[date_column].isna().all():
                st.error(f"The column '{date_column}' does not contain valid date information.")
                st.stop()

            total_sales = data[sales_column].sum()
            total_records = len(data)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Sales", f"${total_sales:,.2f}")
            col2.metric("Total Records", total_records)
            col3.metric("Average Sales", f"${(total_sales / total_records):,.2f}")

            # XGBoost Model for Prediction
            data['TimeIndex'] = np.arange(len(data))
            X = data[['TimeIndex']]
            y = data[sales_column]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = XGBRegressor(objective='reg:squarederror')
            model.fit(X_train, y_train)

            future_indices = np.arange(len(data), len(data) + 30).reshape(-1, 1)
            future_sales = model.predict(future_indices)
            
            predictions = model.predict(X)
            data['Predicted'] = predictions
            data['Error'] = data[sales_column] - data['Predicted']
            
            st.subheader("Actual vs Predicted Sales")
            fig = px.line(data, x=date_column, y=[sales_column, 'Predicted'], title="Actual vs Predicted Sales")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Prediction Error")
            fig_error = px.line(data, x=date_column, y='Error', title="Prediction Errors")
            st.plotly_chart(fig_error, use_container_width=True)

    elif data_category == "Finance":
        total_inflows = data[data['Amount'] > 0]['Amount'].sum()
        total_outflows = abs(data[data['Amount'] < 0]['Amount'].sum())
        net_cash_flow = total_inflows - total_outflows
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Inflows", f"${total_inflows:,.2f}")
        col2.metric("Total Outflows", f"${total_outflows:,.2f}")
        col3.metric("Net Cash Flow", f"${net_cash_flow:,.2f}")
        
        st.subheader("Finance Line Chart")
        fig_line = px.line(data, x=data.index, y='Amount', title="Financial Data Line Chart")
        st.plotly_chart(fig_line, use_container_width=True)
        
        st.subheader("Finance Bar Chart")
        fig_bar = px.bar(data, x=data.index, y='Amount', title="Financial Data Bar Chart")
        st.plotly_chart(fig_bar, use_container_width=True)

    elif data_category == "Marketing":
        impressions = data['Impressions'].sum()
        clicks = data['Clicks'].sum()
        conversions = data['Conversions'].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Impressions", f"{impressions:,}")
        col2.metric("Total Clicks", f"{clicks:,}")
        col3.metric("Total Conversions", f"{conversions:,}")
        
        st.subheader("Marketing Line Chart")
        fig_line = px.line(data, x=data.index, y='Impressions', title="Marketing Data Line Chart")
        st.plotly_chart(fig_line, use_container_width=True)
        
        st.subheader("Marketing Bar Chart")
        fig_bar = px.bar(data, x=data.index, y='Impressions', title="Marketing Data Bar Chart")
        st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.warning("Please upload a dataset to proceed.")
