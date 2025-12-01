import streamlit as st
import pandas as pd
import dask.dataframe as dd
from src.utils import detect_problem_type

st.title("Upload Your Dataset")

st.markdown(
    """
    You can upload any .csv or .xlsx file, but there are some constraints:
    - **Target:** Make sure your dataset has only one target variable.
    - **Header:** Make sure your file has its first column as column headers.
    - **Data:** Make sure that there is at least one categorical feature in the
    dataset.
    """
)


def upload_file():
    """Store raw uploaded file under 'uploaded_file' without overwriting processed data."""
    uploaded_file = st.file_uploader("Choose Your Dataset", type=["csv", "xlsx"], key="uploader")
    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file
        st.success("File Uploaded Successfully")


# Call the upload function.
upload_file()

# If a file is uploaded, load it and show column selector
if "uploaded_file" in st.session_state:
    # Load data first if not already loaded
    if "data" not in st.session_state:
        raw = st.session_state["uploaded_file"]

        try:
            if raw.type == "text/csv":
                df = pd.read_csv(raw)
            else:
                df = pd.read_excel(raw)
        except Exception as read_err:
            st.error(f"Could not read file: {read_err}")
        else:
            st.session_state["data"] = dd.from_pandas(df)
    
    # Show dropdown for target column selection
    if "data" in st.session_state:
        columns = st.session_state["data"].columns.tolist()
        selected_column = st.selectbox("Select Target Column", options=columns, key="target_column_selector")
        
        # Convert column name to index
        if selected_column:
            target_index = columns.index(selected_column)
            st.session_state["index"] = target_index
            
            # Re-detect problem type whenever target column changes
            try:
                problem_type = detect_problem_type(st.session_state["data"], target_index)
                st.session_state["problem_type"] = problem_type
                st.success(f"Detected Problem Type: {problem_type.upper()}")
            except Exception as detect_err:
                st.warning(f"Problem type detection failed: {detect_err}")

    # Legacy: Re-detect problem type if index was set directly (backward compatibility)
    if "index" in st.session_state and "data" in st.session_state and "target_column_selector" not in st.session_state:
        target_index = st.session_state["index"]
        try:
            problem_type = detect_problem_type(st.session_state["data"], target_index)
            st.session_state["problem_type"] = problem_type
            st.success(f"Detected Problem Type: {problem_type.upper()}")
        except Exception as detect_err:
            st.warning(f"Problem type detection failed: {detect_err}")

# If already processed, display the dataframe and show detected problem type
if "data" in st.session_state:
    st.write("Here's a preview of your dataset:")
    st.dataframe(st.session_state["data"].compute().head())
    
    if "problem_type" in st.session_state:
        st.info(f"Problem Type: {st.session_state['problem_type'].upper()}")
