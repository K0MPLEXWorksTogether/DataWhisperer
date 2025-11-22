"""
This python file is responsible for displaying all the data visualization
for the passed input file.

It utilizes the classes from src.knowledge, for acquiring all the data.
"""

import streamlit as st
import pandas as pd

from src.preprocess import DataPreProcessor
from src.knowledge import Knowledge
from src.utils import detect_problem_type
from pandas.api.types import is_numeric_dtype

st.title("Visualize Your Dataset!")
st.write("""
    This data visualization is generated using the train split while
    preprocessing the dataset.
    
    You can use any column, or verify for the whole dataset.
    The mean is always 0 and the standard deviation is always 1.
    This indicates that the dataset is completely scaled.
    """
         )

st.write("Scroll Down To View Column Wise Data")

# Checking for session state variables.
if "data" in st.session_state:
    if "index" in st.session_state:
        dataframe = st.session_state["data"]
        target = st.session_state["index"]
        
        # Detect or retrieve problem type
        if "problem_type" not in st.session_state:
            problem_type = detect_problem_type(dataframe, target)
            st.session_state["problem_type"] = problem_type
        else:
            problem_type = st.session_state["problem_type"]

        # Initializing datapreprocessor class.
        preprocessor = DataPreProcessor(dataframe, target, problem_type)

        # Separating features from target.
        target_feature = preprocessor.y
        preprocessor.categories_numerical()

        # Finding numerical and categorical variables.
        numerical = preprocessor.numerical
        categorical = preprocessor.categories

        # Finding numerical and categorical features.
        if is_numeric_dtype(target):
            numerical.append(target_feature.name)
        else:
            categorical.append(target_feature.name)

        # Finding train, test and validation split.
        result = preprocessor.preprocess(impute_missing=True)

        # Initializing the knowledge class.
        knowledge = Knowledge(input_dataframe=result[0], numerical=numerical, categorical=categorical)
        basic_data = knowledge.basic_data(before_preprocess=False)

        # Placing sidebar for representing the columns of the dataset.
        st.sidebar.title("Select A Column")
        column_names = list(result[0].columns)
        selected_column = st.sidebar.selectbox("Column", column_names)

        # Describe all the data from dd.describe() as a dataframe.
        st.header("Dataset Description")
        st.table(result[0].compute().describe())

        # Representing the found correlation matrix.
        st.header("Correlation Matrix")
        st.image(basic_data["Correlation"])

        # Representing the found feature histogram.
        st.header("Feature Histogram")
        st.image(basic_data["Histogram"])

        # Finding categorical columns.
        is_categorical = selected_column in knowledge.categorical

        # Per-Column Analysis
        st.header(f"Analysis Of Column: `{selected_column}`")
        column_index = column_names.index(selected_column)
        column_data = knowledge.column_data(is_categorical=is_categorical, column_index=column_index)

        # Display column statistics.
        st.subheader("Column Statistics")
        st.table(pd.DataFrame(column_data["Description"].items(), columns=["Metric", "Value"]))

        # Display column graph.
        st.subheader("Column Distribution With Percentile")
        st.write("The red line is the original normal distribution.")
        st.image(column_data["Graph"], use_column_width=True)
