import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Config
st.set_page_config(page_title="Data Visualizer", layout="wide")
st.title("📊 Interactive Data Visualizer (CSV/XLSX)")

# Upload File
uploaded_file = st.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"])

# Visualization Logic
def plot_visualization(df, feature_x, feature_y, graph_type):
    try:
        df = df[[feature_x, feature_y]].dropna()
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))

        if graph_type == "Scatter Plot":
            sns.scatterplot(x=feature_x, y=feature_y, data=df, ax=ax)

        elif graph_type == "Box Plot":
            sns.boxplot(x=feature_x, y=feature_y, data=df, ax=ax)

        elif graph_type == "Histogram":
            sns.histplot(df[feature_x], kde=True, ax=ax)

        elif graph_type in ["Line Graph", "Line Chart"]:
            df_sorted = df.sort_values(by=feature_x)
            ax.plot(df_sorted[feature_x], df_sorted[feature_y], marker='o', linestyle='-')

        elif graph_type == "Bar Chart":
            if df[feature_x].nunique() < 20:
                sns.barplot(x=feature_x, y=feature_y, data=df, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            else:
                st.warning("Too many unique values in X for a bar chart.")

        elif graph_type == "Pie Chart":
            # Use counts instead of sum for categorical pie
            pie_data = df[feature_x].value_counts()
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')

        elif graph_type == "Scatter Chart":
            ax.scatter(df[feature_x], df[feature_y])

        ax.set_title(f"{graph_type}: {feature_x} vs {feature_y}", fontsize=14)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error generating {graph_type}: {e}")

# Main UI
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully!")

        st.markdown("### 📄 Data Preview")
        st.dataframe(df.head())

        cols = df.columns.tolist()
        feature_x = st.selectbox("📌 Select X-axis feature", cols)
        feature_y = st.selectbox("📌 Select Y-axis feature", cols)

        graph_type = st.selectbox("📈 Choose graph type", [
            "Scatter Plot", "Box Plot", "Histogram", "Line Graph",
            "Bar Chart", "Pie Chart", "Line Chart", "Scatter Chart"
        ])

        if st.button("Generate Graph"):
            plot_visualization(df, feature_x, feature_y, graph_type)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
