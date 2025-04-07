import streamlit as st
from src.preprocess import DataPreProcessor
from evalml.automl import AutoMLSearch
from evalml.utils import infer_feature_types
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Model Generation")
st.markdown(
    """
    The model generation is performed via the training, test and validation
    split in data preprocessing.
    
    - The validation data is used to find the best model.
    - The training data is used to train the best model.
    - The testing data is used to find the confusion matrix.
    
    The metrics used to find the best model is **Log Loss**: A simple log loss of a model from validation is used,
    and compared with a baseline model, which is a random or majority class
    predictor. 
    """
)


def extract_pipeline_description(pipeline_name):
    # Split the pipeline name by whitespace to process each word.
    words = pipeline_name.split()

    # Initialize an empty description.
    description = []

    # Iterate through words until 'Classifier' is encountered.
    for word in words:
        description.append(word)
        if 'Classifier' in word:
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
        train_test_validation = return_splits(st.session_state["data"], st.session_state["index"])

        train = train_test_validation[0]
        test = train_test_validation[1]
        validation = train_test_validation[2]

        target_column = train.columns[-1]
        classification_type = ""
        unique = train[target_column].compute().nunique()
        if unique == 2:
            classification_type += "binary"
        else:
            classification_type += "multiclass"

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
            problem_type=classification_type,
            max_batches=5,
            objective='auto',
            random_seed=42,
            max_time=60,
            optimize_thresholds=True,
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

    st.markdown("## Classification Report: Before Testing")
    train_predictions = pipeline.predict(X_train)
    training_report = classification_report(y_train, train_predictions, output_dict=True)
    training_report_df = pd.DataFrame(training_report).transpose()
    st.dataframe(training_report_df)

    # Run predictions on the selected test records
    predictions = pipeline.predict(X_test)

    # Display classification report
    st.markdown("## Classification Report: After Testing")
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
