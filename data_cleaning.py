"""
data_cleaning.py
GlobalReach Analytics — Data Cleaning & Transformation Module
Author: Utkarsh Chandra | SP Jain School of Global Management
Step 2: Data Cleaning & Transformation (10 Marks)
"""

import pandas as pd
import numpy as np

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw dataset, applies cleaning & transformation,
    and returns a clean DataFrame ready for EDA.
    """

    # ── 1. Load Raw Data ───────────────────────────────────────────────────
    df = pd.read_csv(filepath)
    print(f"[INFO] Raw data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── 2. Handle Missing Values ───────────────────────────────────────────
    missing_before = df.isnull().sum()

    # Numeric columns: fill with median (robust to outliers)
    numeric_fill = ['Lead_Score', 'CTR_Percent', 'Discount_Offered_Percent', 'NPS_Score']
    for col in numeric_fill:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"[CLEAN] {col}: filled {missing_before[col]} nulls with median ({median_val:.2f})")

    # ── 3. Data Type Conversion ────────────────────────────────────────────
    df['Date'] = pd.to_datetime(df['Date'])
    df['Lead_Score'] = df['Lead_Score'].astype(int)
    df['NPS_Score'] = df['NPS_Score'].astype(int)
    print("[CLEAN] Date converted to datetime; Lead_Score & NPS_Score cast to int")

    # ── 4. Derived Columns ────────────────────────────────────────────────
    # ROI: Revenue earned per dollar of ad spend
    df['ROI_Percent'] = np.where(
        df['Ad_Spend_USD'] > 0,
        ((df['Revenue_USD'] - df['Ad_Spend_USD']) / df['Ad_Spend_USD'] * 100).round(2),
        0
    )

    # CLV:CAC Ratio (efficiency of customer acquisition)
    df['CLV_CAC_Ratio'] = np.where(
        df['CAC_USD'] > 0,
        (df['CLV_USD'] / df['CAC_USD']).round(2),
        0
    )

    # Lead Conversion Rate (leads → clicks)
    df['Lead_Conversion_Rate'] = np.where(
        df['Clicks'] > 0,
        (df['Leads_Generated'] / df['Clicks'] * 100).round(2),
        0
    )

    # Quarter
    df['Quarter'] = df['Date'].dt.to_period('Q').astype(str)

    # Won/Lost binary flag
    df['Is_Won'] = (df['Deal_Stage'] == 'Closed Won').astype(int)

    print("[TRANSFORM] New columns added: ROI_Percent, CLV_CAC_Ratio, Lead_Conversion_Rate, Quarter, Is_Won")

    # ── 5. Outlier Detection & Capping (IQR Method) ───────────────────────
    outlier_cols = ['Ad_Spend_USD', 'Revenue_USD', 'CAC_USD', 'CLV_USD']
    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"[OUTLIER] {col}: {outliers} outliers capped (IQR method)")

    # ── 6. Encoding Categorical Variables ─────────────────────────────────
    # Ordinal: Company Size
    size_map = {'SME': 1, 'Mid-Market': 2, 'Enterprise': 3}
    df['Company_Size_Enc'] = df['Company_Size'].map(size_map)

    # Ordinal: Deal Stage (funnel order)
    stage_map = {'Prospect': 1, 'Qualified': 2, 'Proposal': 3,
                 'Negotiation': 4, 'Closed Won': 5, 'Closed Lost': 0}
    df['Deal_Stage_Enc'] = df['Deal_Stage'].map(stage_map)

    print("[ENCODE] Ordinal encoding applied: Company_Size_Enc, Deal_Stage_Enc")

    # ── 7. Remove Duplicates ──────────────────────────────────────────────
    dupes = df.duplicated(subset=['Lead_ID']).sum()
    df.drop_duplicates(subset=['Lead_ID'], inplace=True)
    print(f"[CLEAN] Duplicates removed: {dupes}")

    # ── 8. Validation Report ──────────────────────────────────────────────
    print(f"\n[DONE] Clean data: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[DONE] Remaining nulls: {df.isnull().sum().sum()}")

    return df


def get_cleaning_report(filepath: str) -> dict:
    """Returns a summary report of cleaning steps for display in Streamlit."""
    df_raw = pd.read_csv(filepath)
    df_clean = load_and_clean_data(filepath)
    return {
        "raw_shape": df_raw.shape,
        "clean_shape": df_clean.shape,
        "missing_before": int(df_raw.isnull().sum().sum()),
        "missing_after": int(df_clean.isnull().sum().sum()),
        "duplicates_removed": int(df_raw.duplicated(subset=['Lead_ID']).sum()),
        "new_columns": ['ROI_Percent', 'CLV_CAC_Ratio', 'Lead_Conversion_Rate', 'Quarter', 'Is_Won',
                        'Company_Size_Enc', 'Deal_Stage_Enc'],
        "outlier_cols_treated": ['Ad_Spend_USD', 'Revenue_USD', 'CAC_USD', 'CLV_USD']
    }


if __name__ == "__main__":
    df = load_and_clean_data("data/globalreach_200rows.csv")
    print(df.head())
