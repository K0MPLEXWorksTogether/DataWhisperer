import streamlit as st

st.markdown(
    """
    # DataWhisperer - Interact with your dataset.
    
    DataWhisperer is a ML automation tool that can provide interactive
    building of machine learning models and generate feature-related insights
    for your dataset.
    
    ## Key Features
    - **Upload Your Dataset:** Any .csv or .xlsx is supported, with automatic 
    problem type detection (classification or regression).
    - **Data Visualization:** Check and visualize everything, from the 
    most basic details to column-wise normalization checks.
    - **Data Preprocessing:** Your dataset is automatically checked for 
    missing values, duplicates and encodes categories and scales numerical data.
    - **Automatic Model Selection:** A search for the model with the best
    performance score is done, and finally returns the best model.
    - **Model Evaluation:** 
        - **Classification:** Get classification reports, confusion matrix, and accuracy metrics.
        - **Regression:** Get RMSE, MAE, R² scores, residual plots, and predicted vs actual plots.
    - **Dataset Insights:** Basic insights like feature importance and model 
    suitability are performed, plus advanced methods like association rule mining,
    clustering, and hierarchical analysis.
    """
)

st.info(
    """
    ✨ **New:** DataWhisperer now supports both **Classification** (binary/multiclass) 
    and **Regression** tasks with automatic problem type detection!
    """
)
