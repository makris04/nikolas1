import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

# --------------------------------------------------
# Scenario data
# --------------------------------------------------
SCENARIO_A = pd.DataFrame(
    {
        "ID": ["S1", "S2", "S3"],
        "Limb_Length_cm": [94.0, 92.5, 96.0],
        "Cal_10mm_mm": [10.0, 10.0, 10.0],
        "ANT_T1_cm": [68.5, 66.0, 70.5],
        "PM_T1_cm": [102.0, 99.5, 104.0],
        "PL_T1_cm": [98.0, 95.0, 101.0],
        "ANT_T4_cm": [69.0, 66.5, 71.0],
        "PM_T4_cm": [102.5, 100.0, 104.5],
        "PL_T4_cm": [98.5, 95.5, 101.5],
    }
)

SCENARIO_B = pd.DataFrame(
    {
        "ID": ["S1", "S2", "S3"],
        "Limb_Length_cm": [94.0, 92.5, 96.0],
        "Cal_10mm_mm": [11.5, 11.5, 11.5],
        "ANT_T1_cm": [62.0, 60.5, 64.0],
        "PM_T1_cm": [88.2, 86.5, 91.0],
        "PL_T1_cm": [82.1, 80.0, 85.0],
        "ANT_T4_cm": [68.0, 66.5, 70.0],
        "PM_T4_cm": [95.0, 93.0, 98.0],
        "PL_T4_cm": [89.0, 87.0, 92.0],
    }
)

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def compute_metrics(df: pd.DataFrame, apply_bias_correction: bool = True) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "Limb_Length_cm",
        "Cal_10mm_mm",
        "ANT_T1_cm",
        "PM_T1_cm",
        "PL_T1_cm",
        "ANT_T4_cm",
        "PM_T4_cm",
        "PL_T4_cm",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Bias = X_measured - X_true
    df["Bias_mm"] = df["Cal_10mm_mm"] - 10.0
    df["Bias_cm"] = df["Bias_mm"] / 10.0

    if apply_bias_correction:
        df["ANT_T1_cm_corr"] = df["ANT_T1_cm"] - df["Bias_cm"]
        df["PM_T1_cm_corr"] = df["PM_T1_cm"] - df["Bias_cm"]
        df["PL_T1_cm_corr"] = df["PL_T1_cm"] - df["Bias_cm"]
        df["ANT_T4_cm_corr"] = df["ANT_T4_cm"] - df["Bias_cm"]
        df["PM_T4_cm_corr"] = df["PM_T4_cm"] - df["Bias_cm"]
        df["PL_T4_cm_corr"] = df["PL_T4_cm"] - df["Bias_cm"]
    else:
        df["ANT_T1_cm_corr"] = df["ANT_T1_cm"]
        df["PM_T1_cm_corr"] = df["PM_T1_cm"]
        df["PL_T1_cm_corr"] = df["PL_T1_cm"]
        df["ANT_T4_cm_corr"] = df["ANT_T4_cm"]
        df["PM_T4_cm_corr"] = df["PM_T4_cm"]
        df["PL_T4_cm_corr"] = df["PL_T4_cm"]

    # Composite Score (%) = [ (Anterior + Posteromedial + Posterolateral) / (3 * Limb Length) ] * 100
    df["Composite_T1_pct"] = (
        (df["ANT_T1_cm_corr"] + df["PM_T1_cm_corr"] + df["PL_T1_cm_corr"])
        / (3.0 * df["Limb_Length_cm"])
    ) * 100.0

    df["Composite_T4_pct"] = (
        (df["ANT_T4_cm_corr"] + df["PM_T4_cm_corr"] + df["PL_T4_cm_corr"])
        / (3.0 * df["Limb_Length_cm"])
    ) * 100.0

    # % Change = [ (Composite T4 - Composite T1) / Composite T1 ] * 100
    df["Pct_Change"] = (
        (df["Composite_T4_pct"] - df["Composite_T1_pct"]) / df["Composite_T1_pct"]
    ) * 100.0

    return df


def paired_ttest_pvalue(df: pd.DataFrame) -> float:
    x = df["Composite_T1_pct"].dropna().to_numpy()
    y = df["Composite_T4_pct"].dropna().to_numpy()

    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return np.nan

    _, p_value = stats.ttest_rel(x, y, nan_policy="omit")
    return float(p_value)


# --------------------------------------------------
# Streamlit app
# --------------------------------------------------
st.set_page_config(page_title="Y-Balance Calibration & Learning Effect", layout="wide")

st.title("Y-Balance Test: Calibration Bias and Learning Effect")

# Sidebar controls
st.sidebar.header("Controls")
scenario = st.sidebar.selectbox("Scenario Selector", ["Scenario A", "Scenario B"])
apply_bias_correction = st.sidebar.checkbox("Apply calibration bias correction", value=True)
st.sidebar.markdown("Bias formula: **Bias = Cal_10mm_mm - 10.0 (mm)**")

# Pick scenario
if scenario == "Scenario A":
    source_df = SCENARIO_A.copy()
else:
    source_df = SCENARIO_B.copy()

# Compute results
df = compute_metrics(source_df, apply_bias_correction=apply_bias_correction)
p_value = paired_ttest_pvalue(df)
mean_bias = df["Bias_mm"].mean()
avg_comp_t1 = df["Composite_T1_pct"].mean()
avg_comp_t4 = df["Composite_T4_pct"].mean()
avg_pct_change = df["Pct_Change"].mean()

# Tabs
tab1, tab2, tab3 = st.tabs(["Data & Calibration Bias", "Learning Effect", "Diagnostic Alerts"])

with tab1:
    st.subheader("Raw Data")
    st.dataframe(source_df.reset_index(drop=True), use_container_width=True)

    st.subheader("Calibration Bias")
    bias_view = df[["ID", "Cal_10mm_mm", "Bias_mm", "Bias_cm"]].copy()
    st.dataframe(bias_view, use_container_width=True)

    st.subheader("Corrected Reach Measures")
    corrected_cols = [
        "ID",
        "ANT_T1_cm_corr",
        "PM_T1_cm_corr",
        "PL_T1_cm_corr",
        "ANT_T4_cm_corr",
        "PM_T4_cm_corr",
        "PL_T4_cm_corr",
    ]
    st.dataframe(df[corrected_cols], use_container_width=True)

with tab2:
    st.subheader("Composite Scores and Percent Change")
    score_view = df[["ID", "Composite_T1_pct", "Composite_T4_pct", "Pct_Change"]].copy()
    st.dataframe(score_view, use_container_width=True)

    st.metric("Mean Composite Score T1 (%)", round(float(avg_comp_t1), 4))
    st.metric("Mean Composite Score T4 (%)", round(float(avg_comp_t4), 4))
    st.metric("Mean % Change (T1 -> T4)", round(float(avg_pct_change), 4))
    if pd.notna(p_value):
        st.metric("Paired t-test p-value", round(float(p_value), 6))
    else:
        st.metric("Paired t-test p-value", "N/A")

with tab3:
    st.subheader("Diagnostic Alerts")

    if pd.notna(p_value) and p_value < 0.05:
        st.error("Statistically significant change detected between T1 and T4 (p < 0.05).")
    else:
        st.success("No statistically significant difference detected between T1 and T4 (p >= 0.05).")

    if mean_bias > 0:
        st.warning("Calibration bias is positive. Mean Bias > 0 mm means the measured value is greater than the true value.")
    else:
        st.info("Calibration bias is not positive. Mean Bias <= 0 mm.")

    st.write(f"Mean Bias (mm): {mean_bias:.4f}")
    st.write(f"Mean % Change (T1 -> T4): {avg_pct_change:.4f}%")
    if pd.notna(p_value):
        st.write(f"Paired t-test p-value: {p_value:.6f}")
    else:
        st.write("Paired t-test p-value: N/A")
