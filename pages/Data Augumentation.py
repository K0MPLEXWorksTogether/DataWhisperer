import streamlit as st
import dask.dataframe as dd
import pandas as pd
import numpy as np
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Data Augmentation using Correlation", layout="wide")
st.title("📈 Data Augmentation with Correlation (with NaN and Category Support)")

uploaded_file = st.file_uploader("Upload your CSV or XLSX file", type=["csv", "xlsx"])

def categorize_corr(val):
    if val >= 0.7:
        return 'High'
    elif val >= 0.3:
        return 'Mid'
    else:
        return 'Low'

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df_pandas = pd.read_csv(uploaded_file)
    else:
        df_pandas = pd.read_excel(uploaded_file)

    df = dd.from_pandas(df_pandas, npartitions=4)
    columns = df.columns.tolist()
    target = st.selectbox("Select the target variable", columns)

    if st.button("Run Correlation Analysis and Augment Data"):
        with st.spinner("Processing..."):
            df_full = df.compute()
            numeric_df = df_full.select_dtypes(include=['float64', 'int64'])

            if target not in numeric_df.columns:
                st.error("Target must be a numeric column.")
            else:
                numeric_df = numeric_df.dropna()
                df_corr = numeric_df.corr()[target].drop(target)
                corr_df = df_corr.to_frame(name="Correlation")
                corr_df["Correlation Level"] = corr_df["Correlation"].abs().apply(categorize_corr)

                st.subheader("🔍 Correlation with Target")
                st.dataframe(corr_df)

                augmented_df = df_full.copy()

                for col in corr_df.index:
                    level = corr_df.loc[col, "Correlation Level"]
                    if level == "High":
                        augmented_df[f"{col}_squared"] = augmented_df[col] ** 2
                        augmented_df[f"{col}_x_{target}"] = augmented_df[col] * augmented_df[target]
                    elif level == "Mid":
                        try:
                            imp = SimpleImputer(strategy='mean')
                            binner = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
                            reshaped = imp.fit_transform(augmented_df[[col]])
                            binned = binner.fit_transform(reshaped)
                            augmented_df[f"{col}_binned"] = binned
                        except Exception as e:
                            st.warning(f"Could not bin column {col}: {e}")
                    else:
                        # Optionally do something for 'Low' correlation
                        pass

                # Encode non-numeric columns
                cat_cols = df_full.select_dtypes(include=['object', 'category']).columns
                for col in cat_cols:
                    try:
                        le = LabelEncoder()
                        augmented_df[col + "_encoded"] = le.fit_transform(augmented_df[col].astype(str))
                    except Exception as e:
                        st.warning(f"Could not encode column {col}: {e}")

                st.subheader("✅ Augmented Data Preview")
                st.dataframe(augmented_df.head())

                csv = augmented_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Augmented Data", csv, "augmented_data.csv", "text/csv")

