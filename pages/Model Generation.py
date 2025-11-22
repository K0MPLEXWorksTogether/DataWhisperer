import streamlit as st
from src.preprocess import DataPreProcessor
from src.utils import detect_problem_type, is_regression
from evalml.automl import AutoMLSearch
from evalml.utils import infer_feature_types
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.title("Model Generation")
st.markdown(
    """
    The model generation is performed via the training, test and validation
    split in data preprocessing.
    
    - The validation data is used to find the best model.
    - The training data is used to train the best model.
    - The testing data is used to evaluate the model.
    
    **For Classification:** Metrics include Log Loss, Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    
    **For Regression:** Metrics include RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R² Score.
    """
)


def extract_pipeline_description(pipeline_name):
    # Split the pipeline name by whitespace to process each word.
    words = pipeline_name.split()

    # Initialize an empty description.
    description = []

    # Iterate through words until 'Classifier' or 'Regressor' is encountered.
    for word in words:
        description.append(word)
        if 'Classifier' in word or 'Regressor' in word:
            break

    # Join the description into a single string and return.
    return ' '.join(description)


def return_splits(dataframe, target_index) -> list:
    preprocess = DataPreProcessor(dataframe, target_index)
    result = preprocess.preprocess()
    return result


# Check if 'data' and 'index' are in session state
if "data" in st.session_state and "index" in st.session_state:
    with st.spinner("Finding the best models for the dataset..."):
        # Detect problem type first
        problem_type = detect_problem_type(st.session_state["data"], st.session_state["index"])
        
        # Store in session state for other pages
        st.session_state["problem_type"] = problem_type
        
        # Get preprocessed splits with problem type
        preprocessor = DataPreProcessor(st.session_state["data"], st.session_state["index"], problem_type)
        train_test_validation = preprocessor.preprocess()

        train = train_test_validation[0]
        test = train_test_validation[1]
        validation = train_test_validation[2]

        target_column = train.columns[-1]

        # Split the datasets into features and targets
        X_train = train.drop(columns=[target_column]).compute()
        y_train = train[target_column].compute()
        X_test = test.drop(columns=[target_column]).compute()
        y_test = test[target_column].compute()
        X_val = validation.drop(columns=[target_column]).compute()
        y_val = validation[target_column].compute()

        # Convert your data to evalML format
        X_train = infer_feature_types(X_train)
        y_train = infer_feature_types(y_train)
        X_test = infer_feature_types(X_test)
        y_test = infer_feature_types(y_test)
        X_val = infer_feature_types(X_val)
        y_val = infer_feature_types(y_val)

        # Initialize AutoMLSearch
        automl = AutoMLSearch(
            X_train=X_val,
            y_train=y_val,
            problem_type=problem_type,
            max_batches=5,
            objective='auto',
            random_seed=42,
            max_time=60,
            optimize_thresholds=(problem_type in ['binary', 'multiclass']),
            n_jobs=-1,
            patience=75,
            ensembling=True,
        )

        # Run AutoMLSearch
        automl.search()

    st.success("We found the best model!")
    rankings = automl.rankings.head(1)
    model_list = rankings.index.tolist()

    st.markdown("## The best model:")
    # Display models in the main screen
    for idx in model_list:
        row = rankings.loc[idx]
        st.subheader(f"Model: {extract_pipeline_description(row['pipeline_name'])}")
        st.metric(label="Percent Better Than Baseline Predictor", value=row['percent_better_than_baseline'], delta=f"{100 - round(row['percent_better_than_baseline'], 2)}")
        st.metric(label="Mean Cross Validation Score", value=row['mean_cv_score'], delta=f"{100 - round(row['mean_cv_score'], 2)}")
        st.metric(label="Standard Deviation Cross Validation Score" , value=row['standard_deviation_cv_score'], delta=f"{100 - round(row['standard_deviation_cv_score'], 2)}")

    with st.spinner("Training your model with training data: "):
        pipeline = automl.best_pipeline
        pipeline.fit(X_train, y_train)

    st.success("Training was successful!")

    # Run predictions on training and test data
    train_predictions = pipeline.predict(X_train)
    predictions = pipeline.predict(X_test)

    # Display metrics based on problem type
    if problem_type == "regression":
        # Regression metrics
        st.markdown("## Regression Metrics: Training Data")
        
        train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
        train_mae = mean_absolute_error(y_train, train_predictions)
        train_r2 = r2_score(y_train, train_predictions)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="RMSE (Training)", value=f"{train_rmse:.4f}")
        with col2:
            st.metric(label="MAE (Training)", value=f"{train_mae:.4f}")
        with col3:
            st.metric(label="R² Score (Training)", value=f"{train_r2:.4f}")
        
        st.markdown("## Regression Metrics: Test Data")
        
        test_rmse = mean_squared_error(y_test, predictions, squared=False)
        test_mae = mean_absolute_error(y_test, predictions)
        test_r2 = r2_score(y_test, predictions)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="RMSE (Test)", value=f"{test_rmse:.4f}", delta=f"{train_rmse - test_rmse:.4f}")
        with col2:
            st.metric(label="MAE (Test)", value=f"{test_mae:.4f}", delta=f"{train_mae - test_mae:.4f}")
        with col3:
            st.metric(label="R² Score (Test)", value=f"{test_r2:.4f}", delta=f"{test_r2 - train_r2:.4f}")
        
        # Residual plot
        st.markdown("## Residual Plot")
        residuals = y_test - predictions
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(predictions, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        st.pyplot(fig)
        
        # Prediction vs Actual plot
        st.markdown("## Predicted vs Actual Values")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, predictions, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Actual Values')
        st.pyplot(fig)
        
    else:
        # Classification metrics
        st.markdown("## Classification Report: Training Data")
        training_report = classification_report(y_train, train_predictions, output_dict=True)
        training_report_df = pd.DataFrame(training_report).transpose()
        st.dataframe(training_report_df)

        # Display classification report
        st.markdown("## Classification Report: Test Data")
        report = classification_report(y_test, predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Display confusion matrix
        st.markdown("## Confusion Matrix")
        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

else:
    st.warning('Please upload data and specify the target column index.')
