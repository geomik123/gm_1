import streamlit as st
import pandas as pd
import plotly.express as px
from ydata_profiling import ProfileReport
import csv
from plotly.figure_factory import create_annotated_heatmap

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
                # Detect delimiter automatically
                delimiter = detect_delimiter(file)
                data = pd.read_csv(file, delimiter=delimiter)
            elif file.name.endswith('.xlsx'):
                # Handle Excel files with multiple sheets
                xl = pd.ExcelFile(file)
                sheet_name = xl.sheet_names[0]  # Default to the first sheet
                data = xl.parse(sheet_name)
            else:
                st.error("Unsupported file format! Please upload a CSV or Excel file.")
                return None
        else:
            # Default dataset
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
        # Ensure the date column is in datetime format
        try:
            data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
            if data[date_column].isna().all():
                st.warning(f"Could not parse valid dates in column '{date_column}'. Time-based analysis will be skipped.")
                date_column = None
        except Exception as e:
            st.warning(f"Error parsing date column '{date_column}': {e}. Time-based analysis will be skipped.")
            date_column = None

        # Ensure the sales column is numeric
        try:
            data[sales_column] = pd.to_numeric(data[sales_column], errors='coerce')
            if data[sales_column].isna().all():
                st.error(f"The sales column '{sales_column}' contains no valid numeric data. Please check your dataset.")
                st.stop()
            else:
                # Fill NaN values with 0 (optional, depending on the dataset)
                data[sales_column] = data[sales_column].fillna(0)
        except Exception as e:
            st.error(f"Error converting sales column '{sales_column}' to numeric: {e}")
            st.stop()

        if date_column:
            st.title("Automated Sales Dashboard")
            st.markdown("Insights generated from the uploaded dataset.")

            total_sales = data[sales_column].sum()
            total_records = len(data)

            col1, col2 = st.columns(2)
            col1.metric("Total Sales", f"${total_sales:,.2f}")
            col2.metric("Total Records", total_records)

            # Time Series Analysis
            st.header("Monthly Sales Trend")
            time_series = data.groupby(data[date_column].dt.to_period("M"))[sales_column].sum().reset_index()
            time_series[date_column] = time_series[date_column].dt.to_timestamp()
            fig_trend = px.line(
                time_series, x=date_column, y=sales_column, title="Monthly Sales Trend"
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            # Distribution of Sales Data
            st.header("Distribution of Sales")
            fig_distribution = px.histogram(
                data, x=sales_column, nbins=20, title=f"Distribution of {sales_column}"
            )
            st.plotly_chart(fig_distribution, use_container_width=True)

            # Profile Report Section
            st.header("Data Profiling Report")
            st.markdown(
                "Below is the profiling report generated by **ydata-profiling**, available for download."
            )

            # Generate and save the profile report
            profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)
            profile_path = "profiling_report.html"
            profile.to_file(profile_path)

            # Provide the report for download
            with open(profile_path, "rb") as file:
                btn = st.download_button(
                    label="Download Profiling Report",
                    data=file,
                    file_name="profiling_report.html",
                    mime="text/html"
                )

            st.success("Analysis completed automatically!")
else:
    st.warning("Please upload a dataset to proceed.")

