import streamlit as st
import pandas as pd
import numpy as np

import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
st.set_page_config(page_title="Stress Analysis", layout="wide")


def npv(rate, cashflows):
    return npf.npv(rate, cashflows)

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
    redistribution_method = st.sidebar.selectbox(
        "Método de Absorción",
        [
            "Shock Aleatorio",
            "S-Curve",
            "Aleatorio Suavizado (Dirichlet)"
        ]
    )

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

    all_cost_rows  = df[df.iloc[:,0].str.contains(
        "Permisos|Promoción|Gastos de Administración|Legales, Seguros y Otros|Project Management|Management Fee|Fideicomiso|Imprevistos Soft Costs|Reembolso de CAPEX|Etapa I de Desarrollo Urbano|Etapa 2 de Desarrollo Urbano|Mantenimiento de Áreas Verdes|Vigilancia Diurna|Vigilancia Nocturna|Vigilancia Móvil",
        case=False, na=False
    )].index
    
    cost_rows = df[df.iloc[:,0].str.contains(
        "Etapa|Vigilancia|Mantenimiento|CAPEX",
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

        income = df_stressed.loc[income_rows, year_cols].values.astype(float)
        n_years = len(year_cols)
    
        # ------------------------------------------------
        # METHOD 1: RANDOM SHOCK REDISTRIBUTION
        # ------------------------------------------------
        if redistribution_method == "Shock Aleatorio":
    
            # Example: shocks from -30% to +20%
            shocks = np.random.uniform(-0.30, 0.20, size=n_years)
    
            for i in range(n_years):
    
                # Negative shock: move income to next year
                if shocks[i] < 0 and i < n_years - 1:
                    lost = -income[:, i] * shocks[i]
                    income[:, i] += income[:, i] * shocks[i]
                    income[:, i + 1] += lost
    
                # Positive shock: advance income from next year
                elif shocks[i] > 0 and i > 0:
                    advance = income[:, i] * shocks[i]
                    income[:, i] -= advance
                    income[:, i - 1] += advance
    
            df_stressed.loc[income_rows, year_cols] = income
            st.success(f"Ventas redistribuidas usando {redistribution_method}")
    
        # ------------------------------------------------
        # METHOD 2: S-CURVE REDISTRIBUTION
        # ------------------------------------------------
        elif redistribution_method == "S-Curve":
    
            # Common S-curve for absorption
            curve = np.linspace(0.1, 0.9, n_years)
            curve = np.sin(curve * np.pi)  # bell shape
            curve = curve / curve.sum()
    
            total_income = income.sum()
            redistributed = total_income * curve
    
            # Spread proportionally across all income accounts
            for i, col in enumerate(year_cols):
                df_stressed.loc[income_rows, col] = redistributed[i] * (
                    df_stressed.loc[income_rows, col] /
                    df_stressed.loc[income_rows, year_cols].sum(axis=1)
                ).fillna(1)
    
            st.success(f"Ventas redistribuidas usando {redistribution_method}")
    
        # ------------------------------------------------
        # METHOD 3: RANDOM SMOOTH (DIRICHLET)
        # ------------------------------------------------
        elif redistribution_method == "Aleatorio Suavizado (Dirichlet)":
    
            # Dirichlet alpha=2 produces smooth but random shape
            curve = np.random.dirichlet(alpha=np.ones(n_years) * 2)
    
            total_income = income.sum()
            redistributed = total_income * curve
    
            for i, col in enumerate(year_cols):
                df_stressed.loc[income_rows, col] = redistributed[i] * (
                    df_stressed.loc[income_rows, col] /
                    df_stressed.loc[income_rows, year_cols].sum(axis=1)
                ).fillna(1)
    
            st.success(f"Ventas redistribuidas usando {redistribution_method}")
    # --------------------------------------
    # 3) COST INFLATION
    # --------------------------------------
    if use_cost_inflation:
        df_stressed.loc[cost_rows, year_cols] *= (1 + inflation_rate)
        st.success(f"Costos incrementados en un {inflation_rate*100:.0f}%")

    # --------------------------------------
    # CREATE CASHFLOW VECTOR
    # --------------------------------------
    df_stressed[df_stressed.iloc[:,0].str.contains("Flujo", case=False, na=False)] = df_stressed[df_stressed.loc[income_rows, year_cols].sum()] - df_stressed[df_stressed.loc[all_cost_rows, year_cols].sum()]
    cashflow =  df_stressed[df_stressed.iloc[:,0].str.contains("Flujo", case=False, na=False)]
    st.write(cashflow)

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
















