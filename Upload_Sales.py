import streamlit as st
import pandas as pd
import plotly.express as px
from ydata_profiling import ProfileReport
import csv

# Set page configuration for wide layout
st.set_page_config(layout="wide")

# Helper function to detect the delimiter in a CSV file
def detect_delimiter(file):
    sample = file.read(1024).decode("utf-8")
    file.seek(0)  # Reset the file pointer
    sniffer = csv.Sniffer()
    return sniffer.sniff(sample).delimiter

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
            file_path = "customer_shopping_data.csv"
            data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Function to automatically detect columns
def auto_detect_columns(data):
    date_column = None
    sales_column = None

    # Detect date column
    for col in data.columns:
        if "date" in col.lower() or "time" in col.lower():
            date_column = col
            break

    # Detect sales column
    for col in data.columns:
        if "sale" in col.lower() or "amount" in col.lower() or "revenue" in col.lower() or "total" in col.lower():
            sales_column = col
            break

    return date_column, sales_column

# File uploader
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

data = load_data(uploaded_file) if uploaded_file else load_data()

if data is not None:
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Automatically detect columns
    date_column, sales_column = auto_detect_columns(data)

    if not date_column or not sales_column:
        st.error("Could not automatically detect the required columns (Date and Sales). Please check your dataset.")
    else:
        try:
            data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
            data[sales_column] = pd.to_numeric(data[sales_column], errors='coerce').fillna(0)
        except Exception as e:
            st.error(f"Error processing columns: {e}")
            st.stop()

        st.title("Automated Sales Dashboard")
        st.markdown("Insights generated from the uploaded dataset.")

        total_sales = data[sales_column].sum()
        total_records = len(data)

        col1, col2 = st.columns(2)
        col1.metric("Total Sales", f"${total_sales:,.2f}")
        col2.metric("Total Records", total_records)

        st.header("Monthly Sales Trend")
        time_series = data.groupby(data[date_column].dt.to_period("M"))[sales_column].sum().reset_index()
        time_series[date_column] = time_series[date_column].dt.to_timestamp()
        fig_trend = px.line(time_series, x=date_column, y=sales_column, title="Monthly Sales Trend")
        st.plotly_chart(fig_trend, use_container_width=True)

        # Custom Visualizations Section
        st.header("Custom Visualizations")
        st.markdown("Create your own charts and insights from the dataset.")

        # Dropdowns for column selection
        x_axis = st.selectbox("Select X-axis:", options=data.columns)
        y_axis = st.selectbox("Select Y-axis (optional):", options=[None] + data.columns.tolist())
        chart_type = st.selectbox("Select Chart Type:", options=["Scatter Plot", "Bar Chart", "Line Chart"])

        # Optional filter for the dataset
        filter_column = st.selectbox("Select a column to filter (optional):", options=[None] + data.columns.tolist())
        if filter_column:
            unique_values = data[filter_column].dropna().unique()
            filter_value = st.selectbox(f"Select a value for {filter_column}:", options=unique_values)
            filtered_data = data[data[filter_column] == filter_value]
        else:
            filtered_data = data

        # Create the custom chart
        if x_axis and chart_type:
            if chart_type == "Scatter Plot":
                st.subheader("Scatter Plot")
                fig = px.scatter(filtered_data, x=x_axis, y=y_axis, title=f"{chart_type} of {x_axis} vs {y_axis}")
            elif chart_type == "Bar Chart":
                st.subheader("Bar Chart")
                fig = px.bar(filtered_data, x=x_axis, y=y_axis, title=f"{chart_type} of {x_axis} vs {y_axis}")
            elif chart_type == "Line Chart":
                st.subheader("Line Chart")
                fig = px.line(filtered_data, x=x_axis, y=y_axis, title=f"{chart_type} of {x_axis} vs {y_axis}")

            st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please upload a dataset to proceed.")

