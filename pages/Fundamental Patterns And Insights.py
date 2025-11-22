"""
The file is responsible for finding all the patterns and insights from the
dataframe. It utilizes all the classes from models for this task.
"""

import streamlit as st

from models.model_suitability import ModelSuitability
from models.feature_importance import FeatureImportance
from models.apriori import Apriori
from models.dbscan import DBScan
from models.hierarchies import Hierarchies
from src.utils import detect_problem_type

st.title("Fundamental Patterns")
st.markdown(
    """
    Here, you can find some of the most basic patterns identified in 
    your dataset like:
    
    - **Model Suitability:** For classification - linear separability check. For regression - linear vs polynomial fit.
    - **Important Features:** Using Random Forest to identify key predictors.
    
    Also, high-level insights like:
    - Frequently Occurring Itemsets With A Possible Support (`Apriori`).
    - Mined Associate Rules From The Dataset With Confidence (`Apriori`).
    - Clusters Within The Dataset (`DBSCAN`). 
    - Hierarchical Clusters Within The Dataset. (`scipy.cluster`)
    """
)

if "data" in st.session_state and "index" in st.session_state:
    # Detect problem type
    problem_type = detect_problem_type(st.session_state["data"], st.session_state["index"])
    
    # Display problem type
    st.info(f"**Detected Problem Type:** {problem_type.upper()}")
    
    st.subheader("Model Suitability Analysis: ")
    model_suit = ModelSuitability(st.session_state["data"], st.session_state["index"], problem_type)
    suitability_result = model_suit.prediction()

    for key, value in suitability_result.items():
        st.write(f"`{key}`: {value}")

    st.subheader("Feature Importance In The Dataset: ")
    feature_importance = FeatureImportance(st.session_state["data"], st.session_state["index"], problem_type)
    with st.expander("Graph Of Feature Importance"):
        st.image(feature_importance.visualize_importance())

    apriori = Apriori(st.session_state["data"])
    st.subheader("Frequently Occurring Itemsets With Support:")
    with st.expander("A Dataframe Of Frequently Occurring Itemsets"):
        st.dataframe(apriori.itemsets())

    st.subheader("Associate Rules From The Dataset With Confidence:")
    with st.expander("A Dataframe Of Associate Rules From The Dataset"):
        st.dataframe(apriori.rules())

    st.subheader("Density Based Clusters In The Dataset:")
    dbscan = DBScan(st.session_state["data"], st.session_state["index"], problem_type)
    dbscan.cluster()
    with st.expander("Scatter Plot Of Clusters"):
        st.image(dbscan.visualize())
        st.metric(label="Silhouette Score", value=dbscan.score())

    st.subheader("Hierarchical Clusters In The Dataset: ")
    hierarchy = Hierarchies(st.session_state["data"], st.session_state["index"], problem_type)
    with st.expander("Hierarchies In The Dataset"):
        st.image(hierarchy.dendrogram())