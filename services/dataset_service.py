import os
import pandas as pd
import numpy as np
import config

DATASET_MAP = {
    "placement": {"name": "Student Placement Dataset", "file": "Placement.csv"},
    "placement_yn": {"name": "Placement Binary Dataset", "file": "Placement_Y_N.csv"},
    "cgpa_iq": {"name": "CGPA & IQ Multivariate Dataset", "file": "Cgpa_iq_package.csv"},
    "realestate": {"name": "Real Estate Valuation Dataset", "file": "Real estate - Real estate.csv"},
    "titanic": {"name": "Titanic Passenger Survival Dataset", "file": "titanic.csv"},
    "diabetes": {"name": "Pima Indians Diabetes Dataset", "file": "diabetes.csv"},
    "heart": {"name": "UCI Heart Disease Cleveland Dataset", "file": "heart.csv"},
    "breast_cancer": {"name": "Wisconsin Breast Cancer Dataset", "file": "breast-cancer.csv"},
    "banknote": {"name": "Banknote Authentication Dataset", "file": "data_banknote_authentication.csv"},
    "messages": {"name": "Campus Message Detective Teaching Dataset", "file": "campus_56_messages.csv"},
    "messages_222": {"name": "Campus Expanded Messages Dataset", "file": "college_messages_222.csv"}
}

def get_available_datasets():
    return [
        {"id": k, "name": v["name"], "file": v["file"]}
        for k, v in DATASET_MAP.items()
    ]

def profile_dataframe(df, dataset_id="custom", name="Custom Uploaded Dataset", filename="custom.csv"):
    """
    Profiles any Pandas DataFrame to generate dimensions, preview records,
    data types, missing value counts, descriptive statistics, Pearson correlation matrix,
    and distribution histogram buckets.
    """
    clean_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
    df = df[clean_cols]

    shape = df.shape
    columns = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in columns}
    null_counts = {c: int(df[c].isna().sum()) for c in columns}
    
    # Preview rows (convert NaN to None for clean JSON serialization)
    preview = df.head(10).to_dict(orient="records")
    for r in preview:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None

    # Numeric summary
    numeric_df = df.select_dtypes(include=[np.number])
    stats = []
    corr_data = {"columns": [], "values": []}
    distributions = {}

    if not numeric_df.empty:
        stats_df = numeric_df.describe().T.reset_index()
        stats = stats_df.rename(columns={"index": "column"}).to_dict(orient="records")
        for row in stats:
            for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
                if k in row and row[k] is not None and not np.isnan(row[k]):
                    row[k] = round(float(row[k]), 2)

        # Correlation Matrix
        corr = numeric_df.corr().round(2)
        corr_cols = list(corr.columns)
        corr_matrix = []
        for row_vals in corr.values:
            corr_matrix.append([round(float(v), 2) if not np.isnan(v) else 0.0 for v in row_vals])

        corr_data = {
            "columns": corr_cols,
            "matrix": corr_matrix,
            "values": corr_matrix
        }

        # Histogram distribution buckets for all numeric columns (up to 10 features)
        for col in numeric_df.columns[:10]:
            col_data = numeric_df[col].dropna()
            if len(col_data) > 0:
                counts, bin_edges = np.histogram(col_data, bins=10)
                distributions[col] = {
                    "bins": [f"{bin_edges[i]:.1f} – {bin_edges[i+1]:.1f}" for i in range(len(counts))],
                    "counts": counts.tolist()
                }

    # Text / categorical column stats if present
    cat_df = df.select_dtypes(exclude=[np.number])
    cat_summary = {}
    for col in cat_df.columns[:4]:
        vc = df[col].value_counts().head(8).to_dict()
        cat_summary[col] = {"labels": list(vc.keys()), "counts": list(vc.values())}

    return {
        "id": dataset_id,
        "name": name,
        "filename": filename,
        "rows": shape[0],
        "cols": shape[1],
        "columns": columns,
        "dtypes": dtypes,
        "null_counts": null_counts,
        "preview": preview,
        "stats": stats,
        "correlation": corr_data,
        "distributions": distributions,
        "cat_summary": cat_summary
    }

def load_dataset_details(dataset_id="placement"):
    meta = DATASET_MAP.get(dataset_id, DATASET_MAP["placement"])
    path = os.path.join(config.DATA_RAW_DIR, meta["file"])

    if not os.path.exists(path):
        return {"error": f"Dataset file '{meta['file']}' not found on server."}

    df = pd.read_csv(path)
    return profile_dataframe(df, dataset_id=dataset_id, name=meta["name"], filename=meta["file"])
