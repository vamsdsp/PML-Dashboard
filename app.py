from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

ST_PAGE_TITLE = "Fund Risk Dashboard"

FEATURE_COLS = [
    "has_3year_data",
    "has_5year_data",
    "has_1year_data",
    "fund_beta_3years_filled",
    "fund_sharpe_ratio_3years_filled",
    "fund_stdev_3years_filled",
    "fund_return_1year_filled",
    "fund_return_3years_filled",
    "fund_return_5years_filled",
    "fund_return_ytd",
    "total_net_assets",
    "current_vix",
    "current_market_return",
    "current_uncertainty",
    "current_geopolitical_risk",
    "current_interest_rate",
]


@st.cache_data
def load_data() -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data_transformed" / "df_transformed.csv"
    return pd.read_csv(data_path)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    age_dummies = pd.get_dummies(df["fund_age_category"], prefix="age", drop_first=False)
    X = pd.concat([df[FEATURE_COLS], age_dummies], axis=1).fillna(0)
    return X, age_dummies.columns.tolist()


def build_features_with_columns(
    df: pd.DataFrame, age_columns: list[str], feature_columns: list[str]
) -> pd.DataFrame:
    age_dummies = pd.get_dummies(df["fund_age_category"], prefix="age", drop_first=False)
    age_dummies = age_dummies.reindex(columns=age_columns, fill_value=0)
    X = pd.concat([df[FEATURE_COLS], age_dummies], axis=1).fillna(0)
    return X.reindex(columns=feature_columns, fill_value=0)


@st.cache_resource
def train_model(X: pd.DataFrame, y: pd.Series) -> DecisionTreeClassifier:
    param_grid = {
        "max_depth": [3, 4, 5, 6, 7],
        "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "criterion": ["gini", "entropy"],
    }
    search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
    search.fit(X, y)
    return search.best_estimator_


def plot_summary_bars(values: dict[str, float], title: str, ylabel: str) -> None:
    labels = list(values.keys())
    data = list(values.values())

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(labels, data, color="#1f77b4")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig, clear_figure=True)


st.set_page_config(page_title=ST_PAGE_TITLE, layout="wide")

st.title(ST_PAGE_TITLE)

with st.spinner("Loading data..."):
    df = load_data()

X, age_columns = build_features(df)
feature_columns = X.columns.tolist()

y = df["fund_risk_class"]

with st.spinner("Training decision tree..."):
    model = train_model(X, y)

fund_symbols = sorted(df["fund_symbol"].dropna().unique())

selected_symbol = st.selectbox("Select a fund symbol", fund_symbols)
selected_rows = df[df["fund_symbol"] == selected_symbol]

if selected_rows.empty:
    st.warning("No rows found for the selected fund symbol.")
    st.stop()

if len(selected_rows) > 1:
    selected_index = st.selectbox(
        "Select a specific record",
        selected_rows.index,
        format_func=lambda idx: f"Record {idx}",
    )
    selected_row = selected_rows.loc[[selected_index]]
else:
    selected_row = selected_rows.iloc[[0]]

selected_features = build_features_with_columns(selected_row, age_columns, feature_columns)

prediction = model.predict(selected_features)[0]

col1, col2, col3 = st.columns(3)
col1.metric("Fund Symbol", selected_symbol)
col1.metric("Fund Category", selected_row["fund_category"].iloc[0])
col2.metric("Actual Risk Class", selected_row["fund_risk_class"].iloc[0])
col2.metric("Predicted Risk Class", prediction)
col3.metric("Total Net Assets", f"{selected_row['total_net_assets'].iloc[0]:,.0f}")
col3.metric("Fund Age Category", selected_row["fund_age_category"].iloc[0])

st.subheader("Summary Metrics")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    return_values = {
        "YTD": selected_row["fund_return_ytd"].iloc[0],
        "1Y": selected_row["fund_return_1year_filled"].iloc[0],
        "3Y": selected_row["fund_return_3years_filled"].iloc[0],
        "5Y": selected_row["fund_return_5years_filled"].iloc[0],
    }
    plot_summary_bars(return_values, "Return Profile", "Return")

with summary_col2:
    risk_values = {
        "Beta": selected_row["fund_beta_3years_filled"].iloc[0],
        "Sharpe": selected_row["fund_sharpe_ratio_3years_filled"].iloc[0],
        "Stdev": selected_row["fund_stdev_3years_filled"].iloc[0],
    }
    plot_summary_bars(risk_values, "Risk Metrics", "Value")

st.subheader("Selected Record")
st.dataframe(selected_row, use_container_width=True)
