import streamlit as st
import pandas as pd
import numpy as np

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Real Estate Stress Test", layout="wide")

# --------------------------------------
# Utility functions
# --------------------------------------
def npv(rate, cashflows):
    return np.sum(cashflows / (1 + rate) ** np.arange(len(cashflows)))

def irr(cashflows):
    return np.irr(cashflows)


# --------------------------------------
# Main App
# --------------------------------------
st.title("🏗️ Real Estate Project — Stress Test Simulator")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

# User inputs
st.sidebar.header("Parameters")

wacc = st.sidebar.number_input("WACC (%)", value=12.0, step=0.1) / 100
fx_rate = st.sidebar.number_input("DOP/USD Exchange Rate", value=60.0, step=0.5)

scenario_type = st.sidebar.radio(
    "Choose Stress Scenario",
    [
        "Income Reduction",
        "Income Redistribution",
        "Cost Inflation",
    ],
)

# Scenario Parameters
income_reduction_pct = None
redistribute_year = None
redistribute_pct = None
inflation_rate = None

if scenario_type == "Income Reduction":
    income_reduction_pct = st.sidebar.slider(
        "Reduce projected income by (%)", 0, 80, 20
    ) / 100

elif scenario_type == "Income Redistribution":
    redistribute_year = st.sidebar.number_input(
        "Year to shift income FROM (index based, e.g., 1=2026)", min_value=0
    )
    redistribute_pct = st.sidebar.slider(
        "Percentage to shift (%)", 0, 100, 50
    ) / 100

elif scenario_type == "Cost Inflation":
    inflation_rate = st.sidebar.slider(
        "Increase all costs by (%)", 0, 40, 10
    ) / 100


# --------------------------------------
# PROCESSING THE FILE
# --------------------------------------
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name=0)

    st.subheader("📄 Raw Data")
    st.dataframe(df)

    df_stressed = df.copy()

    # Assume columns contain years and rows contain categories
    year_cols = df.columns[1:]     # first column = "category"

    # INCOME ROWS → detect rows containing "Ventas" or "Colocación"
    income_rows = df[df.iloc[:,0].str.contains("Ventas|Colocación", case=False, na=False)].index

    # COST ROWS → detect everything below "Soft Costs"
    cost_rows = df[df.iloc[:,0].str.contains("Costs|Costos|Permisos|Vigilancia|Etapa", case=False, na=False)].index


    # ------------------------------------------------------
    # 1. INCOME REDUCTION SCENARIO
    # ------------------------------------------------------
    if scenario_type == "Income Reduction":

        df_stressed.loc[income_rows, year_cols] *= (1 - income_reduction_pct)

        st.success(f"Income reduced by {income_reduction_pct*100:.0f}%")

    # ------------------------------------------------------
    # 2. INCOME REDISTRIBUTION (shift % to next year)
    # ------------------------------------------------------
    if scenario_type == "Income Redistribution":

        col_from = year_cols[redistribute_year]
        col_to = year_cols[min(redistribute_year + 1, len(year_cols)-1)]

        df_stressed.loc[income_rows, col_to] += (
            df_stressed.loc[income_rows, col_from] * redistribute_pct
        )

        df_stressed.loc[income_rows, col_from] *= (1 - redistribute_pct)

        st.success(f"Moved {redistribute_pct*100:.0f}% of income from {col_from} → {col_to}")

    # ------------------------------------------------------
    # 3. COST INFLATION
    # ------------------------------------------------------
    if scenario_type == "Cost Inflation":

        df_stressed.loc[cost_rows, year_cols] *= (1 + inflation_rate)

        st.success(f"Costs increased by {inflation_rate*100:.0f}%")

    # ------------------------------------------------------
    # CALCULATE CASH FLOW, NPV, IRR
    # ------------------------------------------------------
    st.subheader("📊 Stressed Cash Flow")

    # detect row named "Flujo de Caja del Fideicomiso"
    cashflow_row = df[df.iloc[:,0].str.contains("Flujo", case=False, na=False)].index[0]

    cashflow = df_stressed.loc[cashflow_row, year_cols].values.astype(float)

    st.write("**Stressed Cash Flow:**")
    st.bar_chart(cashflow)

    project_npv = npv(wacc, cashflow)
    project_irr = irr(cashflow)

    st.metric("NPV (USD)", f"{project_npv/fx_rate:,.2f}")
    st.metric("IRR", f"{project_irr*100:.2f}%")

    # Show final table
    st.subheader("📄 Final Stressed Table")
    st.dataframe(df_stressed)

else:
    st.info("Upload an Excel file to begin.")



























