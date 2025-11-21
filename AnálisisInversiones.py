import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px

st.set_page_config(page_title="Análisis Estrés", layout="wide")


def npv(rate, cashflows):
    return npf.npv(rate, cashflows)

def irr(cashflows):
    return npf.irr(cashflows)

def run_montecarlo(cashflow_base, n_sim=5000,
                   income_shock_std=0.10,
                   redistribution_std=0.10,
                   cost_inflation_mean=0.04,
                   cost_inflation_std=0.02,
                   wacc_mean=0.10,
                   wacc_std=0.015):

    npvs = []

    for _ in range(n_sim):

        cf = cashflow_base.copy().astype(float)

        # 1) Shock aleatorio a ingresos (años futuros)
        income_shock = np.random.normal(0, income_shock_std)
        cf[1:] *= (1 + income_shock)

        # 2) Redistribución aleatoria (no cambia el total)
        redist = np.random.normal(0, redistribution_std, size=len(cf)-1)
        redist -= redist.mean()
        cf[1:] *= (1 + redist)

        # 3) Inflación de costos si hay costos negativos en CF
        cost_inf = np.random.normal(cost_inflation_mean, cost_inflation_std)
        cf[cf < 0] *= (1 + cost_inf)

        # 4) WACC aleatorio
        wacc = np.random.normal(wacc_mean, wacc_std)

        # 5) Calcular NPV
        npv_get = npv(wacc,cf)
        npvs.append(npv_get)

    return np.array(npvs)



st.title("Simulador de Escenarios de Estrés")

uploaded_file = st.file_uploader("Subir Excel", type=["xlsx"])

# Sidebar parameters
st.sidebar.header("Parámetros")

wacc = st.sidebar.number_input("WACC (%)", value=19.48, step=0.1) / 100
fx_rate = st.sidebar.number_input("Tasa de Cambio DOP/USD", value=63.0, step=0.5)

st.sidebar.write("---")
st.sidebar.header("Escenarios de Estrés")

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
        "Ingresos reducidos en un (%)", 0, 100, 10
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
        "Costos incrementados en un (%)", 0, 50, 5
    ) / 100
else:
    inflation_rate = 0


# --------------------------------------
# PROCESSING
# --------------------------------------
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    try:
        df = df[~np.logical_and(df['Cash Flow del Fideicomiso'].str.contains("Total"), df['Cash Flow del Fideicomiso'].str.contains("Cost"))]
    except:
        pass
    st.dataframe(df)

    # Copy for stressed scenario
    df_stressed = df.copy()

    # Assume: first column is category, next ones are years
    year_cols = df.columns[1:]
    future_year_cols = year_cols[1:] 
    # Detect rows
    income_rows = df[df.iloc[:,0].str.contains(
        "Ventas|Colocación|Ingresos|Aporte",
        case=False, na=False
    )].index

    all_cost_rows  = df[np.logical_or(df.iloc[:,0].str.contains(
        "Permisos|Promoción|Gastos|Administración|Legales|Management|Imprevistos|Reembolso|CAPEX|Etapa|Etapa|Mantenimiento|Vigilancia",
        case=False, na=False
    ), df.iloc[:,0]=="Fideicomiso")].index
    
    cost_rows = df[df.iloc[:,0].str.contains(
        "Etapa|Vigilancia|Mantenimiento|CAPEX",
        case=False, na=False
    )].index

    # --------------------------------------
    # 1) INCOME REDUCTION
    # --------------------------------------
    if use_income_reduction:
        df_stressed.loc[income_rows, future_year_cols] *= (1 - income_reduction_pct)
        st.success(f"Ingresos reducidos en un {income_reduction_pct*100:.0f}%")

    # --------------------------------------
    # 2) INCOME REDISTRIBUTION
    # --------------------------------------
    if use_income_redistribution:

        income = df_stressed.loc[income_rows, future_year_cols].values.astype(float)
        n_years = len(future_year_cols)
    
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
    
            df_stressed.loc[income_rows, future_year_cols] = income
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
            for i, col in enumerate(future_year_cols):
                df_stressed.loc[income_rows, col] = redistributed[i] * (
                    df_stressed.loc[income_rows, col] /
                    df_stressed.loc[income_rows, future_year_cols].sum(axis=1)
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
    
            for i, col in enumerate(future_year_cols):
                df_stressed.loc[income_rows, col] = redistributed[i] * (
                    df_stressed.loc[income_rows, col] /
                    df_stressed.loc[income_rows, future_year_cols].sum(axis=1)
                ).fillna(1)
    
            st.success(f"Ventas redistribuidas usando {redistribution_method}")
    # --------------------------------------
    # 3) COST INFLATION
    # --------------------------------------
    if use_cost_inflation:
        df_stressed.loc[cost_rows, future_year_cols] *= (1 + inflation_rate)
        st.success(f"Costos incrementados en un {inflation_rate*100:.0f}%")


    target_row = df_stressed[df_stressed.iloc[:,0].str.contains("Flujo", case=False, na=False)].index[0]
    

    income_sum = df_stressed.loc[income_rows, year_cols].sum()
    cost_sum   = df_stressed.loc[all_cost_rows, year_cols].sum()
    # 3. Assign the result to the target row
    df_stressed.loc[target_row, year_cols] = income_sum + cost_sum

    cashflow =  df_stressed.loc[target_row, year_cols]
    

    st.subheader("📊 Variación Cash Flow")
    st.bar_chart(cashflow)


    retencion = st.number_input("% Retención", value=10.0, step=0.5) / 100
    fideico = st.number_input("% Fideicomitente", value=83.07, step=0.1) / 100
    ofp = 1 - fideico 
    st.write(f"Fideicomitente: {round(fideico*100,2)}%")
    st.write(f"Público General: {round(ofp*100,2)}%")

    
    project_npv = npv(wacc, cashflow)
    project_irr = irr(cashflow)
    
    st.write("Flujo de Caja del Proyecto")
    st.dataframe(pd.DataFrame(cashflow).T, hide_index=True)

    reserva_liquidez = pd.Series(np.zeros(len(cashflow)), index=cashflow.index)
    reserva_liquidez.iloc[1] = -350.00
    reserva_liquidez.iloc[-1] = 350.00

    reserva_liquidez2 = pd.Series(np.zeros(len(cashflow)), index=cashflow.index)
    reserva_liquidez2.iloc[-1] = 350.00
    
    st.write("Reserva de Liquidez")
    st.dataframe(pd.DataFrame(reserva_liquidez).T, hide_index=True)

    st.write("Flujo Acumulado")
    st.dataframe(pd.DataFrame((reserva_liquidez+cashflow).cumsum()).T, hide_index=True)
    
    st.write("Dividendos Netos")
  #  st.dataframe(pd.DataFrame((reserva_liquidez+cashflow).cumsum()*(1-retencion)).T, hide_index=True)
   # net_dividends = (reserva_liquidez+cashflow).cumsum()*(1-retencion)
   # net_dividends_tir = ((reserva_liquidez+cashflow)*(1-retencion)).clip(lower=0)

    cf = (reserva_liquidez+cashflow).astype(float)
    
    dividends = []
    cumsum = 0
    
    for value in cf:
        # 1. add this year's CF
        cumsum += value
    
        if cumsum > 0:
            # 2. pay out all excess cash as dividend
            dividend = cumsum
            cumsum = 0    # reset because we paid everything
        else:
            # 3. cannot pay dividend
            dividend = 0
    
        dividends.append(dividend)
    
    dividends = pd.Series(dividends, index=cf.index) * (1-retencion)

    st.dataframe(pd.DataFrame((dividends)).T, hide_index=True)


    new_cf = net_dividends.clip(lower=0)
    c1, c2 = st.columns(2)
    
    
    

    st.write("Flujo de Caja del Proyecto")

    
    cashflow_fideico = (new_cf*fideico)
    aporte_inicial_fideico = st.number_input("Aporte Inicial Fideicomitente", value=-2163.3, step=0.1)
    cashflow_fideico.iloc[0] = aporte_inicial_fideico
    net_dividends_tir_fideico = net_dividends_tir*fideico
    net_dividends_tir.iloc[0] = aporte_inicial_fideico
    net_dividends_tir_fideico.iloc[0] = aporte_inicial_fideico

    st.write("Flujo Fideicomitente")
    st.dataframe(pd.DataFrame((net_dividends_tir_fideico)).T, hide_index=True)
    
    cashflow_opv = (new_cf*ofp)
    aporte_inicial_opv = st.number_input("Aporte Inicial Publico General", value=-441.00, step=0.1)
    cashflow_opv.iloc[1] = aporte_inicial_opv
    net_dividends_tir_opv = ((reserva_liquidez+cashflow)*(1-retencion)).clip(lower=0)*ofp
    net_dividends_tir_opv.iloc[1] = aporte_inicial_opv

    st.write("Flujo Publico General")
    st.dataframe(pd.DataFrame((net_dividends_tir_opv)).T, hide_index=True)
    


    fideico_irr = irr(net_dividends_tir)
    opv_irr = irr(net_dividends_tir_opv)

    c1.metric("NPV (USD)", f"{project_npv/fx_rate:,.2f}")
    c1.metric("NPV of Dividenddos Netos (USD)", f"{npv(wacc, new_cf)/fx_rate:,.2f}")
    
    c2.metric("IRR Fideicomiso", f"{fideico_irr*100:.2f}%")
    c2.metric("IRR Público General", f"{opv_irr*100:.2f}%")

    
    
    # Show final table
    st.subheader("Flujos Simulados Estresados")
    st.dataframe(df_stressed)
    
    # Histograma
    st.subheader("Montecarlo NPV")
    npvs = run_montecarlo(cashflow, wacc_mean=wacc)/fx_rate
    npv_df = pd.DataFrame({"NPV": npvs})
    fig = px.histogram(
    npv_df,
    x="NPV",
    nbins=60,
    title="Distribución Montecarlo del NPV",
    opacity=0.7
    )
    
    fig.update_layout(
        bargap=0.05,
        template="plotly_white"
    )

    p1  = np.percentile(npvs, 1)
    p5  = np.percentile(npvs, 5)
    p10 = np.percentile(npvs, 10)

    fig.add_vline(x=p1,  line_dash="dash", line_width=2, annotation_text=f"P1={round(p1,2)}")
    fig.add_vline(x=p5,  line_dash="dash", line_width=2, annotation_text=f"P5={round(p5,2)}")
    fig.add_vline(x=p10, line_dash="dash", line_width=2, annotation_text=f"P10={round(p10,2)}")

    st.plotly_chart(fig, use_container_width=True)
        
    

else:
    st.info("Suba un Excel")



































































