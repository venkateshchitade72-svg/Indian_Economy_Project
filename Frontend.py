from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).resolve().parent / "Indian_Economy_Sectorwise_Dataset.csv"
REQUIRED_COLUMNS = {
    "Year",
    "Quarter",
    "Sector",
    "GDP_Lakh_Crore",
    "Growth_%",
    "Employment_Million",
    "Exports_Crore",
    "Imports_Crore",
    "FDI_Crore",
    "Inflation_%",
    "CPI",
    "IIP",
    "Tax_Revenue_Crore",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "The economy dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    return data


st.set_page_config(
    page_title="Indian Economy Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Indian Economy Dashboard")
st.caption(
    "Explore quarterly GDP, sector performance, employment, trade, and "
    "macroeconomic indicators."
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH.name}. Place it beside Frontend.py.")
    st.stop()

economy = load_data()
years = sorted(int(year) for year in economy["Year"].unique())
sectors = sorted(economy["Sector"].dropna().unique().tolist())

with st.sidebar:
    st.header("Filters")
    year_range = st.slider(
        "Year range",
        min_value=years[0],
        max_value=years[-1],
        value=(years[0], years[-1]),
        step=1,
    )
    selected_sectors = st.multiselect(
        "Sectors",
        options=sectors,
        default=sectors,
    )

filtered = economy.loc[
    economy["Year"].between(year_range[0], year_range[1])
    & economy["Sector"].isin(selected_sectors)
].copy()
filtered["Period"] = (
    filtered["Year"].astype(str) + " " + filtered["Quarter"].astype(str)
)

if filtered.empty:
    st.info("Choose at least one sector to display the dashboard.")
    st.stop()

total_gdp = filtered["GDP_Lakh_Crore"].sum()
average_growth = filtered["Growth_%"].mean()
total_employment = filtered["Employment_Million"].sum()
total_exports = filtered["Exports_Crore"].sum()

metric_cards = st.columns(4)
metric_cards[0].metric("GDP in selection", f"₹{total_gdp:,.2f} lakh crore")
metric_cards[1].metric("Average growth", f"{average_growth:.2f}%")
metric_cards[2].metric("Employment", f"{total_employment:,.1f} million")
metric_cards[3].metric("Exports", f"₹{total_exports:,.0f} crore")

st.divider()

gdp_by_period = (
    filtered.groupby(["Year", "Quarter"], as_index=False)["GDP_Lakh_Crore"]
    .sum()
    .sort_values(["Year", "Quarter"])
)
gdp_by_period["Period"] = (
    gdp_by_period["Year"].astype(str) + " " + gdp_by_period["Quarter"]
)
gdp_by_sector = (
    filtered.groupby("Sector", as_index=False)["GDP_Lakh_Crore"]
    .sum()
    .sort_values("GDP_Lakh_Crore", ascending=False)
)

trend_column, sector_column = st.columns([1.6, 1])
with trend_column:
    st.subheader("Quarterly GDP trend")
    st.line_chart(
        gdp_by_period,
        x="Period",
        y="GDP_Lakh_Crore",
        y_label="GDP (lakh crore)",
        x_label="Quarter",
    )

with sector_column:
    st.subheader("GDP by sector")
    st.bar_chart(
        gdp_by_sector,
        x="Sector",
        y="GDP_Lakh_Crore",
        y_label="GDP (lakh crore)",
    )

st.subheader("Explore another indicator")
indicator_options = {
    "Average growth (%)": ("Growth_%", "mean"),
    "Employment (million)": ("Employment_Million", "sum"),
    "Exports (crore)": ("Exports_Crore", "sum"),
    "Imports (crore)": ("Imports_Crore", "sum"),
    "FDI (crore)": ("FDI_Crore", "sum"),
    "Inflation (%)": ("Inflation_%", "mean"),
    "Consumer price index": ("CPI", "mean"),
    "Industrial production index": ("IIP", "mean"),
    "Tax revenue (crore)": ("Tax_Revenue_Crore", "sum"),
}
indicator_label = st.selectbox("Indicator", list(indicator_options))
indicator_column, aggregation = indicator_options[indicator_label]
indicator_by_period = (
    filtered.groupby(["Year", "Quarter"], as_index=False)[indicator_column]
    .agg(aggregation)
    .sort_values(["Year", "Quarter"])
)
indicator_by_period["Period"] = (
    indicator_by_period["Year"].astype(str) + " " + indicator_by_period["Quarter"]
)
st.area_chart(
    indicator_by_period,
    x="Period",
    y=indicator_column,
    y_label=indicator_label,
    x_label="Quarter",
)

with st.expander("View filtered data"):
    st.dataframe(
        filtered.drop(columns="Period"),
        use_container_width=True,
        hide_index=True,
    )
