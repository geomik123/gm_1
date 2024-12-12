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
category  = st.sidebar.selectbox("Select Data Category", ["Financial Data - Bank Statements", "Sales and Commercial Data", "Marketing Data", "Other type of Data"]) 
data = load_data(uploaded_file) if uploaded_file else load_data()

if data is not None and category != "Select Category":
    st.title(f"📊 {category} Analysis and Prediction Dashboard")

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

        # Row 1: Monthly Trend
        st.header("Monthly Sales Trend")
        monthly_sales = data.groupby(data[date_column].dt.to_period("M"))[sales_column].sum().reset_index()
        monthly_sales[date_column] = monthly_sales[date_column].dt.to_timestamp()
        fig_trend = px.line(monthly_sales, x=date_column, y=sales_column, title="Monthly Sales Trend")
        st.plotly_chart(fig_trend, use_container_width=True)

        # Row 2: Box Plot and Donut Chart
        st.header("Visualizations")
        numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

        col1, col2 = st.columns(2)

        # Boxplot for Outliers
        with col1:
            st.subheader("Boxplot for Outliers")
            if numerical_columns:
                selected_boxplot_col = st.selectbox("Select a numerical column for boxplot:", numerical_columns, key="boxplot")
                fig_boxplot = px.box(data, y=selected_boxplot_col, title=f"Boxplot of {selected_boxplot_col}")
                st.plotly_chart(fig_boxplot, use_container_width=True)

        # Donut Chart for Category Distribution
        with col2:
            st.subheader("Category Distribution")
            if categorical_columns:
                selected_category_col = st.selectbox("Select a categorical column for donut chart:", categorical_columns, key="donutchart")
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

        # Row 3: Correlation Table
        st.header("Correlation Table")
        if len(numerical_columns) > 1:
            st.subheader("Correlation Heatmap")
            correlation_matrix = data[numerical_columns].corr()
            fig_corr = px.imshow(
                correlation_matrix,
                labels=dict(x="Features", y="Features", color="Correlation"),
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                color_continuous_scale="Viridis",
                title="Correlation Heatmap",
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # Regression Analysis and Prediction
        st.header("Regression Analysis and Sales Prediction")

        # Prepare data for regression
        try:
            data['Day'] = data[date_column]
            data['Month'] = data[date_column].dt.to_period("M").dt.to_timestamp()
            data['Quarter'] = data[date_column].dt.to_period("Q").dt.to_timestamp()
            data['Year'] = data[date_column].dt.to_period("Y").dt.to_timestamp()

            # Group data for different filters
            filters = {
                "Day": data.groupby('Day')[sales_column].sum().reset_index(),
                "Month": data.groupby('Month')[sales_column].sum().reset_index(),
                "Quarter": data.groupby('Quarter')[sales_column].sum().reset_index(),
                "Year": data.groupby('Year')[sales_column].sum().reset_index()
            }

            # User selects the filter
            time_filter = st.radio("Select Time Filter:", options=["Day", "Month", "Quarter", "Year"])
            filtered_data = filters[time_filter]

            # Linear Regression Model
            X = np.arange(len(filtered_data)).reshape(-1, 1)  # Time index
            y = filtered_data[sales_column].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            # Future Prediction
            future_indices = np.arange(len(filtered_data), len(filtered_data) + 30).reshape(-1, 1)
            future_dates = pd.date_range(filtered_data.iloc[-1, 0], periods=31, freq="D")[1:] if time_filter == "Day" else None
            future_sales = model.predict(future_indices)

            # Combine actual and predicted data
            filtered_data['Type'] = 'Actual'
            future_data = pd.DataFrame({
                time_filter: future_dates,
                sales_column: future_sales,
                'Type': 'Prediction'
            })
            combined_data = pd.concat([filtered_data, future_data])

            # Visualization
            fig = px.line(
                combined_data,
                x=time_filter,
                y=sales_column,
                color='Type',
                line_dash='Type',
                title=f"{time_filter}-Level Sales and Forecast",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display Results
            st.write(f"Mean Squared Error: {mse:.2f}")
            st.write(future_data)

        except Exception as e:
            st.error(f"Regression analysis failed: {e}")
else:
    st.warning("Please upload a dataset to proceed.")
