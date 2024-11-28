import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from plotly.figure_factory import create_annotated_heatmap
from ydata_profiling import ProfileReport
import csv
from datetime import timedelta

# Set page configuration for wide layout
st.set_page_config(layout="wide", page_title="Sales Dashboard", page_icon="📊")

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
            # Default dataset
            file_path = "customer_shopping_data.csv"
            data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Sidebar
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
data = load_data(uploaded_file) if uploaded_file else load_data()

if data is not None:
    st.title("📊 Sales Data Analysis and Prediction Dashboard")

    # Identify date and sales columns
    date_column, sales_column = None, None
    for col in data.columns:
        if "date" in col.lower() or "time" in col.lower():
            date_column = col
        if "sale" in col.lower() or "amount" in col.lower() or "price" in col.lower() or "total" in col.lower():
            sales_column = col

    if not date_column or not sales_column:
        st.error("Could not automatically detect the required columns (Date and Sales). Please check your dataset.")
    else:
        # Convert to appropriate formats
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        data[sales_column] = pd.to_numeric(data[sales_column], errors='coerce').fillna(0)
        
        if data[date_column].isna().all():
            st.error(f"The column '{date_column}' does not contain valid date information.")
            st.stop()

        # Display metrics
        total_sales = data[sales_column].sum()
        total_records = len(data)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sales", f"${total_sales:,.2f}")
        col2.metric("Total Records", total_records)
        col3.metric("Average Sales", f"${(total_sales / total_records):,.2f}")

        # Create charts
        st.header("Visualizations")

        # Row 1: Monthly Trend
        with st.container():
            st.subheader("Monthly Sales Trend")
            monthly_sales = data.groupby(data[date_column].dt.to_period("M"))[sales_column].sum().reset_index()
            monthly_sales[date_column] = monthly_sales[date_column].dt.to_timestamp()
            fig_trend = px.line(monthly_sales, x=date_column, y=sales_column, title="Monthly Sales Trend")
            st.plotly_chart(fig_trend, use_container_width=True)

        # Row 2: Correlation, Boxplot, and Pie Chart
        st.subheader("Custom Visualizations")
        numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

        col1, col2, col3 = st.columns(3)

        # Correlation Heatmap
        if len(numerical_columns) > 1:
            with col1:
                st.write("Correlation Heatmap")
                correlation_matrix = data[numerical_columns].corr()
                fig_corr = px.imshow(
                    correlation_matrix,
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig_corr, use_container_width=True)

        # Boxplot for Outliers
        if numerical_columns:
            with col2:
                st.write("Boxplot for Outliers")
                selected_boxplot_col = st.selectbox("Select a numerical column for boxplot:", numerical_columns, key="boxplot")
                fig_boxplot = px.box(data, y=selected_boxplot_col, title=f"Boxplot of {selected_boxplot_col}")
                st.plotly_chart(fig_boxplot, use_container_width=True)

        # Pie Chart for Category Distribution
        if categorical_columns:
            with col3:
                st.write("Category Distribution")
                selected_category_col = st.selectbox("Select a categorical column for pie chart:", categorical_columns, key="piechart")
                category_counts = data[selected_category_col].value_counts().reset_index()
                category_counts.columns = ['Category', 'Count']
                fig_pie = px.pie(
                    category_counts,
                    names='Category',
                    values='Count',
                    title=f"Distribution of {selected_category_col}",
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # Regression Analysis and Prediction
        st.header("Regression Analysis and Sales Prediction")

        # Prepare data for regression
        try:
            data['Month'] = data[date_column].dt.to_period("M").dt.to_timestamp()
            monthly_sales = data.groupby('Month')[sales_column].sum().reset_index()

            # Linear Regression Model
            X = np.arange(len(monthly_sales)).reshape(-1, 1)  # Time index
            y = monthly_sales[sales_column].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            # Display results
            st.subheader("Regression Results")
            st.write(f"Mean Squared Error: {mse:.2f}")

            # Future Prediction
            future_months = pd.date_range(monthly_sales['Month'].max(), periods=4, freq='M')[1:]
            future_indices = np.arange(len(monthly_sales), len(monthly_sales) + len(future_months)).reshape(-1, 1)
            future_sales = model.predict(future_indices)

            # Display future predictions
            predictions = pd.DataFrame({
                "Month": future_months,
                "Predicted Sales": future_sales
            })
            st.write(predictions)

            # Visualization
            fig_future = px.line(monthly_sales, x='Month', y=sales_column, title="Sales Prediction")
            fig_future.add_scatter(x=future_months, y=future_sales, mode='lines+markers', name="Predictions")
            st.plotly_chart(fig_future, use_container_width=True)
        except Exception as e:
            st.error(f"Regression analysis failed: {e}")
else:
    st.warning("Please upload a dataset to proceed.")

