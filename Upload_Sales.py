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
st.set_page_config(layout="wide", page_title="Sales Dashboard", page_icon="📊")

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

            monthly_sales = data.groupby(data[date_column].dt.to_period("M"))[sales_column].sum().reset_index()
            monthly_sales[date_column] = monthly_sales[date_column].dt.to_timestamp()
            fig_trend = px.line(monthly_sales, x=date_column, y=sales_column, title="Monthly Sales Trend")
            st.plotly_chart(fig_trend, use_container_width=True)

            numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

           
            row1_col1, row1_col2 = st.columns(2)

            with row1_col1:
                st.subheader("Boxplot for Outliers")
                if numerical_columns:
                    selected_boxplot_col = st.selectbox("Select a numerical column for boxplot:", numerical_columns, key="boxplot")
                    fig_boxplot = px.box(data, y=selected_boxplot_col, title=f"Boxplot of {selected_boxplot_col}")
                    st.plotly_chart(fig_boxplot, use_container_width=True)

            with row1_col2:
                st.subheader("Category Distribution")
                if categorical_columns:
                    selected_category_col = st.selectbox("Select a categorical column for donut chart:", categorical_columns, key="donutchart")
                    category_counts = data[selected_category_col].value_counts().reset_index()
                    category_counts.columns = ['Category', 'Count']
                    fig_pie = px.pie(category_counts, names='Category', values='Count', title=f"Distribution of {selected_category_col}", hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
            row2_col1, = st.columns(1)
            
            with row2_col1:
                # XGBoost Model for Prediction
                # Create a time index and prepare features for XGBoost
                data['TimeIndex'] = np.arange(len(data))
                X = data[['TimeIndex']]
                y = data[sales_column]
                
                # Split the data for training and testing
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train the XGBoost Regressor
                model = XGBRegressor(objective='reg:squarederror', random_state=42)
                model.fit(X_train, y_train)
                
                # Predict on existing data (to visualize how the model performs on the historical data)
                historical_predictions = model.predict(X)
                
                # Prepare for future prediction (30 days ahead)
                future_indices = np.arange(len(data), len(data) + 30).reshape(-1, 1)
                future_sales = model.predict(future_indices)
                
                # Create future dates to match the future predictions
                last_date = data[date_column].max()
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
                
                # Create a DataFrame for future predictions
                future_df = pd.DataFrame({
                    date_column: future_dates,
                    sales_column: np.nan,  # No actuals for future dates
                    'Predicted': future_sales
                })
                
                # Combine original data with predictions
                data['Predicted'] = historical_predictions  # Attach historical predictions to the original data
                combined_data = pd.concat([data[[date_column, sales_column, 'Predicted']], future_df], ignore_index=True)
                
                # Plot Actual vs Predicted
                st.subheader("Actual vs Predicted Sales (Including 30-Day Forecast)")
                
                # Plot the combined data (including both the actuals and future predictions)
                fig = px.line(
                    combined_data, 
                    x=date_column, 
                    y=[sales_column, 'Predicted'], 
                    title="Actual vs Predicted Sales (with 30-day Forecast)"
                )
                
                st.plotly_chart(fig, use_container_width=True)


    
    elif data_category == "Finance":
        total_inflows = data[data['Amount'] > 0]['Amount'].sum()
        total_outflows = abs(data[data['Amount'] < 0]['Amount'].sum())
        net_cash_flow = total_inflows - total_outflows
        st.metric("Total Inflows", f"${total_inflows:,.2f}")
        st.metric("Total Outflows", f"${total_outflows:,.2f}")
        st.metric("Net Cash Flow", f"${net_cash_flow:,.2f}")

    elif data_category == "Marketing":
        impressions = data['Impressions'].sum()
        clicks = data['Clicks'].sum()
        conversions = data['Conversions'].sum()
        ctr = (clicks / impressions) * 100 if impressions > 0 else 0
        conversion_rate = (conversions / clicks) * 100 if clicks > 0 else 0
        st.metric("Total Impressions", f"{impressions:,}")
        st.metric("Total Clicks", f"{clicks:,}")
        st.metric("CTR", f"{ctr:.2f}%")
        st.metric("Total Conversions", f"{conversions:,}")
        st.metric("Conversion Rate", f"{conversion_rate:.2f}%")

else:
    st.warning("Please upload a dataset to proceed.")
