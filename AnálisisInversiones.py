import streamlit as st
import pandas as pd
import numpy as np

import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
st.set_page_config(page_title="Real Estate Stress Test", layout="wide")


def npv(rate, cashflows):
    return np.sum(cashflows / (1 + rate) ** np.arange(len(cashflows)))

def irr(cashflows):
    return npf.irr(cashflows)


st.title("Simulador de Escenarios de Estrés")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

# Sidebar parameters
st.sidebar.header("Parameters")

wacc = st.sidebar.number_input("WACC (%)", value=19.48, step=0.1) / 100
fx_rate = st.sidebar.number_input("DOP/USD Exchange Rate", value=63.0, step=0.5)

st.sidebar.write("---")
st.sidebar.header("Stress Scenarios (Select any combination)")

# -------------------------
# Scenario Switches
# -------------------------
use_income_reduction = st.sidebar.checkbox("Reducción de Ventas")
use_income_redistribution = st.sidebar.checkbox("Redistribución de las Ventas")
use_cost_inflation = st.sidebar.checkbox("Inflación o Aumento de Costos")

# -------------------------
# Scenario Parameters
# -------------------------
if use_income_reduction:
    income_reduction_pct = st.sidebar.slider(
        "Reduce projected income by (%)", 0, 80, 20
    ) / 100
else:
    income_reduction_pct = 0

if use_income_redistribution:
    redistribute_year = st.sidebar.number_input(
        "Redistribute FROM", min_value=0
    )
    redistribute_pct = st.sidebar.slider(
        "Redistribución (%)", 0, 100, 50
    ) / 100
else:
    redistribute_year = None
    redistribute_pct = 0

if use_cost_inflation:
    inflation_rate = st.sidebar.slider(
        "Costos incrementados en un (%)", 0, 40, 10
    ) / 100
else:
    inflation_rate = 0


# --------------------------------------
# PROCESSING
# --------------------------------------
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Raw Data")
    st.dataframe(df)

    # Copy for stressed scenario
    df_stressed = df.copy()

    # Assume: first column is category, next ones are years
    year_cols = df.columns[1:]

    # Detect rows
    income_rows = df[df.iloc[:,0].str.contains(
        "Ventas|Colocación|Ingresos|Revenue",
        case=False, na=False
    )].index

    cost_rows = df[df.iloc[:,0].str.contains(
        "Cost|Costo|Permisos|Etapa|Vigilancia|Mantenimiento",
        case=False, na=False
    )].index

    # --------------------------------------
    # 1) INCOME REDUCTION
    # --------------------------------------
    if use_income_reduction:
        df_stressed.loc[income_rows, year_cols] *= (1 - income_reduction_pct)
        st.success(f"Ingresos reducidos en un {income_reduction_pct*100:.0f}%")

    # --------------------------------------
    # 2) INCOME REDISTRIBUTION
    # --------------------------------------
    if use_income_redistribution:

        # Ensure valid redistribution target
        if redistribute_year < len(year_cols) - 1:
            col_from = year_cols[redistribute_year]
            col_to = year_cols[redistribute_year + 1]

            df_stressed.loc[income_rows, col_to] += (
                df_stressed.loc[income_rows, col_from] * redistribute_pct
            )
            df_stressed.loc[income_rows, col_from] *= (1 - redistribute_pct)

            st.success(
                f"Moved {redistribute_pct*100:.0f}% of income from {col_from} → {col_to}"
            )
        else:
            st.warning("Redistribution out of range.")

    # --------------------------------------
    # 3) COST INFLATION
    # --------------------------------------
    if use_cost_inflation:
        df_stressed.loc[cost_rows, year_cols] *= (1 + inflation_rate)
        st.success(f"Costos incrementados en un {inflation_rate*100:.0f}%")

    # --------------------------------------
    # CREATE CASHFLOW VECTOR
    # --------------------------------------
    cf_row = df[df.iloc[:,0].str.contains("Flujo", case=False, na=False)].index[0]
    cashflow = df_stressed.loc[cf_row, year_cols].values.astype(float)

    st.subheader("📊 Variación Cash Flow")
    st.bar_chart(cashflow)

    # --------------------------------------
    # NPV, IRR
    # --------------------------------------
    project_npv = npv(wacc, cashflow)
    project_irr = irr(cashflow)

    c1, c2 = st.columns(2)
    c1.metric("NPV (USD)", f"{project_npv/fx_rate:,.2f}")
    c2.metric("IRR", f"{project_irr*100:.2f}%")

    # Show final table
    st.subheader("Flujos Simulados Estresados")
    st.dataframe(df_stressed)

else:
    st.info("Upload an Excel file to begin.")





