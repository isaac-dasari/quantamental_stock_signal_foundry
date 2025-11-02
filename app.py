import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as plx

# from quant_pipeline import build_feature_matrix, train_and_backtest

from alpha_pipeline import build_feature_matrix, train_and_backtest
st.set_page_config(page_title="Quantamental App", layout="wide")

st.title("📈 Quantamental Signal Foundry")
st.caption("News + Macro + Technical features → ML model → Backtest & Signals")

st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker", value="AAPL")
start = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-08-01")).strftime("%Y-%m-%d")
end = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-08-30")).strftime("%Y-%m-%d")
warmup_days = st.sidebar.number_input("Warmup Days (feature spin-up)", min_value=0, max_value=365, value=200, step=10)
news_hours = st.sidebar.slider("News window (hours around each day, UTC)", min_value=6, max_value=48, value=12, step=6)

label_kind = st.sidebar.selectbox("Label Type", ["barrier", "nextday"])
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    barrier_up = st.number_input("Barrier Up (σ×)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
with col2:
    barrier_dn = st.number_input("Barrier Down (σ×)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
with col3:
    barrier_hold = st.number_input("Max Holding (days)", min_value=2, max_value=30, value=7, step=1)

tune_threshold = st.sidebar.checkbox("Tune threshold for Sharpe", value=True)
threshold = st.sidebar.slider("Initial threshold", min_value=0.3, max_value=0.7, value=0.55, step=0.01)
trade_excess = st.sidebar.checkbox("Trade excess vs SPY", value=True)

run_btn = st.sidebar.button("🚀 Run Backtest")

out_dir = st.sidebar.text_input("Save outputs to (local folder)", value="outputs")

# --- Helpers to load artifacts no matter parquet/csv ---
def _load_table(base_name: str, out_dir: str = ".") -> pd.DataFrame:
    pq = os.path.join(out_dir, f"{base_name}.parquet")
    cs = os.path.join(out_dir, f"{base_name}.csv")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    if os.path.exists(cs):
        return pd.read_csv(cs, index_col=0, parse_dates=True)
    raise FileNotFoundError(f"Couldn't find {base_name}.parquet or {base_name}.csv in {out_dir}")

def _as_series(obj: pd.Series | pd.DataFrame, preferred_col: str | None = None) -> pd.Series:
    """Return a 1-D Series from a Series or 1-col DataFrame (or by column name)."""
    if isinstance(obj, pd.Series):
        return obj
    if preferred_col and preferred_col in obj.columns:
        s = obj[preferred_col]
    elif obj.shape[1] == 1:
        s = obj.iloc[:, 0]
    else:
        # fallback: first numeric column
        num = obj.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            raise ValueError("No numeric column available to coerce into a Series.")
        s = num.iloc[:, 0]
    # ensure 1-D
    s = pd.Series(np.asarray(s).reshape(-1), index=obj.index, name=s.name)
    return s

if run_btn:
    with st.spinner("Building features... (prices, macro, news; computing technicals)"):
        # feats, px = build_feature_matrix(ticker, start, end, warmup_days=warmup_days, news_hours=news_hours, out_dir=out_dir)
        feats, px = build_feature_matrix(ticker, start, end, news_hours=news_hours, out_dir=out_dir)

    st.success(f"Features built: {feats.shape[0]} rows × {feats.shape[1]} features")

    with st.spinner("Training, CV, tuning threshold, and backtesting..."):
        metrics, trades = train_and_backtest(
            feats=feats,
            px = px,
            label_kind=label_kind,
            barrier_up=barrier_up,
            barrier_dn=barrier_dn,
            barrier_holding=barrier_hold,
            tune_threshold=tune_threshold,
            threshold=threshold,
            trade_excess=trade_excess,
            out_dir=out_dir
        )


    st.subheader("📊 Metrics")
    st.json(metrics)

    st.subheader("📉 Price & Strategy")
    # prices = px["AdjClose"] if "AdjClose" in px.columns else px["Close"]
    # df_plot = pd.DataFrame({
    #     "price": prices,
    #     "signal": trades["signal"].reindex(prices.index).fillna(0.0)
    # })
    # strat_curve = np.exp(trades["strat"].cumsum())
    # df_curve = pd.DataFrame({"strategy": strat_curve}).reindex(prices.index).ffill()

    # st.plotly_chart(px.line(df_plot[["price"]], title=f"{ticker} Price"), use_container_width=True)
    # st.plotly_chart(px.line(df_curve, title="Strategy Cumulative (exp of cum log returns)"), use_container_width=True)

    # st.subheader("🔁 Signals Over Time")
    # st.line_chart(trades["proba"], height=200)
    # st.bar_chart(trades["signal"], height=200)

    # --- Load saved outputs ---
    OUT_DIR = out_dir if 'out_dir' in globals() and out_dir else "."  # or set explicitly
    prices_df  = _load_table("prices",   OUT_DIR)
    trades_df  = _load_table("trades",   OUT_DIR)
    features_df = _load_table("features", OUT_DIR)  # optional, but often handy

    # --- Pick a clean 1-D price series (AdjClose -> Close -> first numeric) ---
    if "AdjClose" in prices_df.columns:
        price = _as_series(prices_df["AdjClose"])
    elif "Adj Close" in prices_df.columns:
        price = _as_series(prices_df["Adj Close"])
    elif "Close" in prices_df.columns:
        price = _as_series(prices_df["Close"])
    else:
        price = _as_series(prices_df)  # first numeric fallback
    price = pd.to_numeric(price, errors="coerce")
    price.name = "price"

    # --- Coerce trades columns to 1-D Series, aligned to price index ---
    signal = _as_series(trades_df, "signal").reindex(price.index).fillna(0.0)
    proba  = _as_series(trades_df, "proba").reindex(price.index)
    strat  = _as_series(trades_df, "strat").reindex(price.index)

    # --- Build plot DataFrames (pure 1-D inside) ---
    df_plot  = pd.concat([price, signal.rename("signal")], axis=1)
    strat_curve = np.exp(strat.cumsum()).rename("strategy").ffill()
    df_curve = strat_curve.to_frame()

    # --- Charts ---
    st.plotly_chart(plx.line(df_plot[["price"]], title=f"{ticker} Price"), use_container_width=True)
    st.plotly_chart(plx.line(df_curve,           title="Strategy Cumulative (exp of cum log returns)"),
                    use_container_width=True)

    st.subheader("🔁 Signals Over Time")
    st.line_chart(proba.fillna(0.0), height=200)
    st.bar_chart(signal, height=200)

    st.subheader("📥 Downloads")
    for fname in ["features.csv", "prices.csv", "labels.csv", "trades.csv", "metrics.json"]:
        p = os.path.join(out_dir, fname)
        if os.path.exists(p):
            with open(p, "rb") as f:
                st.download_button(f"Download {fname}", f, file_name=fname)
else:
    st.info("Configure parameters and click **Run Backtest** to execute the full pipeline.")
