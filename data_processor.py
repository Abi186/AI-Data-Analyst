"""
data_processor.py
------------------
Pandas-based data processing layer:
  CSV / Excel -> Data Processing (Pandas) -> Statistics + summaries fed to the AI Analyst
"""

import pandas as pd
import numpy as np


def load_file(uploaded_file) -> pd.DataFrame:
    """Load a CSV or Excel file (Streamlit UploadedFile object) into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a .csv, .xlsx, or .xls file.")
    return df


def basic_stats(df: pd.DataFrame) -> dict:
    """Compute core statistics: shape, dtypes, missing values, numeric summary."""
    numeric_df = df.select_dtypes(include=np.number)
    stats = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": numeric_df.describe().to_dict() if not numeric_df.empty else {},
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return stats


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return correlation matrix for numeric columns (empty DF if <2 numeric cols)."""
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr()


def build_context_summary(df: pd.DataFrame, stats: dict, max_rows_preview: int = 5) -> str:
    """
    Builds a compact text summary of the dataset to feed into the LLM prompt.
    Kept short deliberately -- small local models have limited context windows.
    """
    preview = df.head(max_rows_preview).to_string()
    summary = f"""
Dataset shape: {stats['rows']} rows x {stats['columns']} columns
Columns and types: {stats['dtypes']}
Missing values per column: {stats['missing_values']}
Duplicate rows: {stats['duplicate_rows']}

Sample rows:
{preview}

Numeric summary (describe()):
{stats['numeric_summary']}
""".strip()
    return summary
