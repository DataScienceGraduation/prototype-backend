import pandas as pd
import numpy as np

def extract_enhanced_metadata(df: pd.DataFrame, sample_size=5) -> dict:
    """Extracts rich metadata with statistical summaries"""
    metadata = {
        "overview": {
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_percentage": (df.isnull().mean() * 100).round(2).to_dict()
        },
        "variables": {},
        "correlation_insights": {}
    }
    for col in df.columns:
        col_meta = {"type": str(df[col].dtype)}
        coldata = df[col].dropna()
        if np.issubdtype(df[col].dtype, np.number):
            col_meta.update({
                "stats": {
                    "min": coldata.min(),
                    "max": coldata.max(),
                    "mean": coldata.mean(),
                    "std": coldata.std(),
                    "skew": coldata.skew(),
                    "kurtosis": coldata.kurtosis(),
                    "zeros": (df[col] == 0).sum()
                },
                "percentiles": coldata.quantile([0.25, 0.5, 0.75]).to_dict()
            })
        else:
            unique_vals = coldata.nunique()
            col_meta["stats"] = {
                "unique_count": unique_vals,
                "top_values": coldata.value_counts().nlargest(3).to_dict()
            }
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                col_meta["stats"].update({
                    "min_date": coldata.min().isoformat(),
                    "max_date": coldata.max().isoformat()
                })
        metadata["variables"][col] = col_meta
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        high_corr = corr.unstack().sort_values(key=abs, ascending=False)
        high_corr = high_corr[high_corr.index.get_level_values(0) != high_corr.index.get_level_values(1)]
        metadata["correlation_insights"] = {
            "top_correlated": high_corr.head(3).reset_index().to_dict(orient='records'),
            "correlation_matrix_shape": corr.shape
        }
    return metadata