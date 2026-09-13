from pathlib import Path

import pandas as pd
import streamlit as st
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


st.set_page_config(
    page_title="VAR | Ice cream & heater",
    page_icon="📈",
    layout="wide",
)


DATA_PATH = Path(__file__).parent / "ice_cream_vs_heater.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["Month"])
    data = data.set_index("Month").sort_index()
    data.columns = [column.strip() for column in data.columns]
    return data.apply(pd.to_numeric, errors="coerce").dropna()


@st.cache_data
def fit_var(data: pd.DataFrame, split_ratio: float, lag: int):
    split = max(lag + 2, int(len(data) * split_ratio))
    train_levels = data.iloc[:split]
    test_levels = data.iloc[split:]
    differences = data.diff().dropna()
    train_diff = differences.iloc[: max(0, split - 1)]
    test_diff = differences.iloc[max(0, split - 1) :]

    model = VAR(train_diff)
    fitted = model.fit(lag)
    test_forecast = fitted.forecast(train_diff.values[-fitted.k_ar :], steps=len(test_diff))
    test_forecast = pd.DataFrame(test_forecast, index=test_diff.index, columns=data.columns)
    level_forecast = test_forecast.cumsum().add(train_levels.iloc[-1], axis="columns")
    return fitted, train_levels, test_levels, train_diff, test_diff, level_forecast


def adf_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in data.columns:
        statistic, p_value, *_ = adfuller(data[column].dropna())
        rows.append({"Série": column, "Statistique ADF": statistic, "p-value": p_value})
    return pd.DataFrame(rows).set_index("Série")


st.markdown("# VAR Forecast Studio")
st.caption("Prévision multivariée mensuelle pour les séries heater et ice cream")

if not DATA_PATH.exists():
    st.error(f"Fichier introuvable : {DATA_PATH.name}")
    st.stop()

data = load_data(str(DATA_PATH))
if len(data.columns) < 2 or len(data) < 20:
    st.error("Le fichier doit contenir au moins deux séries et vingt observations.")
    st.stop()

with st.sidebar:
    st.header("Paramètres")
    split_ratio = st.slider("Part entraînement", 0.60, 0.95, 0.90, 0.05)
    max_lag = max(1, min(12, (int(len(data) * split_ratio) - 1) // 4))
    lag = st.slider("Ordre VAR (p)", 1, max_lag, min(4, max_lag))
    horizon = st.slider("Horizon de prévision", 1, 24, min(12, len(data) - 1))
    st.divider()
    st.metric("Observations", len(data))
    st.metric("Période", f"{data.index.min():%m/%Y} → {data.index.max():%m/%Y}")

fitted, train_levels, test_levels, train_diff, test_diff, heldout_forecast = fit_var(
    data, split_ratio, lag
)

future_diff = fitted.forecast(train_diff.values[-fitted.k_ar :], steps=horizon)
future_index = pd.date_range(data.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
future_forecast = pd.DataFrame(future_diff, index=future_index, columns=data.columns)
future_levels = future_forecast.cumsum().add(data.iloc[-1], axis="columns")

metrics = st.columns(3)
metrics[0].metric("Lag sélectionné", lag)
metrics[1].metric("Train", len(train_levels))
metrics[2].metric("Test", len(test_levels))

tab_forecast, tab_diagnostics, tab_data = st.tabs(["Prévisions", "Diagnostics", "Données"])

with tab_forecast:
    st.subheader("Historique et projection")
    history = data.copy()
    history.columns = [f"Réel · {column}" for column in history.columns]
    projection = future_levels.copy()
    projection.columns = [f"Prévision · {column}" for column in projection.columns]
    st.line_chart(pd.concat([history, projection], axis=1), height=430)

    st.subheader("Valeurs futures")
    st.dataframe(future_levels.style.format("{:.2f}"), use_container_width=True)

    if len(test_levels):
        st.subheader("Évaluation sur la période test")
        test_display = pd.concat(
            [test_levels.add_suffix(" · réel"), heldout_forecast.add_suffix(" · prévision")], axis=1
        )
        st.line_chart(test_display, height=320)

with tab_diagnostics:
    st.subheader("Stationnarité")
    st.caption("Le VAR est ajusté sur les différences premières des séries.")
    st.dataframe(adf_table(data), use_container_width=True)
    st.dataframe(adf_table(data.diff().dropna()).style.format({"Statistique ADF": "{:.3f}", "p-value": "{:.4f}"}), use_container_width=True)

    st.subheader("Autocorrélation des résidus")
    diagnostic_rows = []
    for column in fitted.resid.columns:
        p_value = acorr_ljungbox(fitted.resid[column], lags=[min(12, len(fitted.resid) // 4)], return_df=True)["lb_pvalue"].iloc[0]
        diagnostic_rows.append({"Série": column, "p-value Ljung-Box": p_value})
    st.dataframe(pd.DataFrame(diagnostic_rows).set_index("Série").style.format("{:.4f}"), use_container_width=True)

with tab_data:
    st.subheader("Jeu de données chargé")
    st.dataframe(data, use_container_width=True)
    st.download_button(
        "Télécharger les prévisions CSV",
        future_levels.to_csv().encode("utf-8"),
        "var_forecast.csv",
        "text/csv",
    )