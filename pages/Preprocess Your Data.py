import streamlit as st

from src.preprocess import DataPreProcessor
from src.utils import detect_problem_type

st.title("Preprocess Your Dataset")
st.markdown(
    """
    The process of preprocessing any dataset is as follows:
    - **Handle Missing Values:** Missing values are handled using KNN Imputation
    for numerical, and most_frequent imputation for categories.
    - **Handle Exact Duplicates:** Exact duplicates, are simply handled
    by removal.
    - **Removing Outliers:** Outliers from the dataset, are handled by removal.
    - **Feature Category Encoding:** Features which are categories are encoded
    using the OneHotEncoder.
    - **Target Encoding:** For classification, targets are encoded using LabelEncoder. 
    For regression, targets remain as continuous values.
    - **Data Splitting:** Data is split into training, test and validation splits,
    60% training, 20% for validation and testing.
    - **Scaling Numerical Features:** All numerical features are scaled separately
    for training, testing and validation.
    """
)

if "data" in st.session_state:
    dataframe = st.session_state["data"]
    target = st.session_state["index"]
    
    # Detect or retrieve problem type
    if "problem_type" not in st.session_state:
        problem_type = detect_problem_type(dataframe, target)
        st.session_state["problem_type"] = problem_type
    else:
        problem_type = st.session_state["problem_type"]
    
    st.info(f"**Problem Type:** {problem_type.upper()}")
    
    preprocessor = DataPreProcessor(input_df=dataframe, target_index=target, problem_type=problem_type)
    result = preprocessor.preprocess(impute_missing=True)

    st.write("Here's your dataset after train, test and validation: ")

    names = ["Train", "Test", "Validation"]
    name_index = 0
    for res in result:
        st.subheader(f"{names[name_index]} Split: ")
        st.dataframe(res.compute().head())
        name_index += 1
