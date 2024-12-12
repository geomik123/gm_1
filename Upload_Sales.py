import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv
from datetime import timedelta
import chardet


# Set page configuration for wide layout
st.set_page_config(layout="wide", page_title="Sales Dashboard", page_icon="📊")

# Detect the encoding of the file
def detect_encoding(file):
    raw_data = file.read(10000)
    result = chardet.detect(raw_data)
    file.seek(0)
    return result['encoding']
# Helper function to detect the delimiter in a CSV file
def detect_delimiter(file):
    try:
        encoding = detect_encoding(file)
        sample = file.read(4096).decode(encoding)  # Ensure UTF-8 decoding
        file.seek(0)  # Reset the file pointer
        sniffer = csv.Sniffer()
        return sniffer.sniff(sample).delimiter
    except csv.Error as e:
        st.warning(f"Could not detect delimiter automatically. Defaulting to comma.")
        return ','  # Default fallback
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return ','


# Helper function to load data
@st.cache_data
def load_data(file=None):
    try:
        if file is not None:
            if file.name.endswith('.csv'):
                delimiter = detect_delimiter(file)
                data = pd.read_csv(file, delimiter=delimiter)
            elif file.name.endswith('.xlsx'):
                xl = pd.ExcelFile(file)
                sheet_name = xl.sheet_names[0]  # Default to the first sheet
                data = xl.parse(sheet_name)
            else:
                st.error("Unsupported file format! Please upload a CSV or Excel file.")
                return None
        else:
            # Default dataset
            file_path = "./retail_sales_dataset.csv"
            data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Sidebar
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
category  = st.sidebar.selectbox("Select Data Category", ["Choose Category", "Financial Data - Bank Statements", "Sales and Commercial Data", "Marketing Data", "Other type of Data"]) 
data = load_data(uploaded_file) if uploaded_file else load_data()

if data is not None and category != "Select Category":
    st.title(f"📊 {category} Analysis Dashboard")
    
    if category == "Financial Data - Bank Statements":
        st.subheader("Financial Data Analysis")
        
        if 'Amount' in data.columns:
            data['Amount'] = pd.to_numeric(data['Amount'], errors='coerce')
        
        inflow = data[data['Amount'] > 0]['Amount'].sum()
        outflow = abs(data[data['Amount'] < 0]['Amount'].sum())
        net_flow = inflow - outflow

        st.metric("Total Inflows", f"${inflow:,.2f}")
        st.metric("Total Outflows", f"${outflow:,.2f}")
        st.metric("Net Cash Flow", f"${net_flow:,.2f}")
        
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            monthly_flow = data.groupby(data['Date'].dt.to_period("M"))['Amount'].sum().reset_index()
            fig = px.line(monthly_flow, x='Date', y='Amount', title="Monthly Cash Flow")
            st.plotly_chart(fig, use_container_width=True)

    elif category == "Sales and Commercial Data":
        st.subheader("Sales Data Analysis")
        
        if 'Date' in data.columns and 'Sales' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data['Sales'] = pd.to_numeric(data['Sales'], errors='coerce').fillna(0)
            
            total_sales = data['Sales'].sum()
            total_records = len(data)
            average_sales = total_sales / total_records
            
            st.metric("Total Sales", f"${total_sales:,.2f}")
            st.metric("Total Records", total_records)
            st.metric("Average Sales", f"${average_sales:,.2f}")
            
            monthly_sales = data.groupby(data['Date'].dt.to_period("M"))['Sales'].sum().reset_index()
            fig = px.line(monthly_sales, x='Date', y='Sales', title="Monthly Sales Trend")
            st.plotly_chart(fig, use_container_width=True)
            
            # XGBoost Model for Sales Prediction
            data['TimeIndex'] = np.arange(len(data))
            X = data[['TimeIndex']]
            y = data['Sales']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = XGBRegressor()
            model.fit(X_train, y_train)

            future_indices = np.arange(len(data), len(data) + 30).reshape(-1, 1)
            future_sales = model.predict(future_indices)
            future_dates = pd.date_range(start=data['Date'].max(), periods=30, freq='D')

            future_df = pd.DataFrame({'Date': future_dates, 'Predicted Sales': future_sales})
            st.write("Future Sales Prediction")
            st.write(future_df)

    elif category == "Marketing Data":
        st.subheader("Marketing Data Analysis")
        
        impressions = data['Impressions'].sum() if 'Impressions' in data.columns else 0
        clicks = data['Clicks'].sum() if 'Clicks' in data.columns else 0
        conversions = data['Conversions'].sum() if 'Conversions' in data.columns else 0
        spend = data['Spend'].sum() if 'Spend' in data.columns else 0
        
        if impressions > 0:
            ctr = (clicks / impressions) * 100
        else:
            ctr = 0

        if clicks > 0:
            conversion_rate = (conversions / clicks) * 100
        else:
            conversion_rate = 0

        st.metric("Total Impressions", f"{impressions:,}")
        st.metric("Total Clicks", f"{clicks:,}")
        st.metric("Click-Through Rate (CTR)", f"{ctr:.2f}%")
        st.metric("Total Conversions", f"{conversions:,}")
        st.metric("Conversion Rate", f"{conversion_rate:.2f}%")
        
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            daily_performance = data.groupby(data['Date'].dt.to_period("D"))[['Impressions', 'Clicks', 'Conversions']].sum().reset_index()
            daily_performance['Date'] = daily_performance['Date'].dt.to_timestamp()
            
            fig = px.line(daily_performance, x='Date', y=['Impressions', 'Clicks', 'Conversions'], title="Daily Performance")
            st.plotly_chart(fig, use_container_width=True)
            
else:
    if category == "Select Category":
        st.warning("Please select a data category to proceed.")
    else:
        st.warning("Please upload a dataset to proceed.")
