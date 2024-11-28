import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.figure_factory import create_annotated_heatmap

st.title("Custom Visualizations")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    # Separate numerical and categorical columns
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

    if numerical_columns or categorical_columns:
        st.header("Custom Visualizations")

        # Correlation Heatmap
        if len(numerical_columns) > 1:
            st.subheader("Correlation Heatmap")
            correlation_matrix = data[numerical_columns].corr()
            fig_corr = create_annotated_heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns.tolist(),
                y=correlation_matrix.columns.tolist(),
                colorscale="Viridis"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # Boxplot for Outliers
        st.subheader("Boxplot for Outliers")
        if numerical_columns:
            selected_boxplot_col = st.selectbox("Select a numerical column for boxplot:", numerical_columns)
            fig_boxplot = px.box(data, y=selected_boxplot_col, title=f"Boxplot of {selected_boxplot_col}")
            st.plotly_chart(fig_boxplot, use_container_width=True)
        else:
            st.warning("No numerical columns available for boxplot.")

        # Pie Chart for Category Distribution
        st.subheader("Category Distribution")
        if categorical_columns:
            selected_category_col = st.selectbox("Select a categorical column for pie chart:", categorical_columns)
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
        else:
            st.warning("No categorical columns available for pie chart.")

        # Scatter Plot for Relationships
        st.subheader("Scatter Plot for Relationships")
        if len(numerical_columns) > 1:
            x_axis = st.selectbox("Select X-axis:", numerical_columns)
            y_axis = st.selectbox("Select Y-axis:", numerical_columns)
            category_column = st.selectbox("Select a categorical column for coloring:", categorical_columns) if categorical_columns else None
            if category_column:
                fig_scatter = px.scatter(
                    data,
                    x=x_axis,
                    y=y_axis,
                    color=category_column,
                    title=f"Scatter Plot of {x_axis} vs {y_axis}"
                )
            else:
                fig_scatter = px.scatter(
                    data,
                    x=x_axis,
                    y=y_axis,
                    title=f"Scatter Plot of {x_axis} vs {y_axis}"
                )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Not enough numerical columns for scatter plot.")
    else:
        st.error("The dataset does not contain numerical or categorical columns.")


