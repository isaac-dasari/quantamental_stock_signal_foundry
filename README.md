# Quantamental Signal Foundry
This Streamlit project pulls together pricing, macro and news data, builds quantamental features, labels them, trains an ML classifier with embargoed time-series CV, tunes thresholds, backtests, and finally surfaces charts and downloads.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Adjust parameters on the left sidebar, click **Run Backtest**.

## Architecture
```mermaid
flowchart TD
    subgraph External Data
        YF[YFinance pricing]
        GDELT[GDELT news API]
        Macro[Macro benchmarks (SPY, QQQ, DXY, VIX)]
    end
    Ingest[Fetchers\n(prices, macro, actions, news)]
    Features[Feature engineering\n(technicals, momentum, sentiment, macro context)]
    Labels[Labeling & ML\n(triple-barrier or next-day; TS CV)]
    Backtest[Threshold tuning & backtest]
    App[Streamlit UI]
    Store[(Outputs\nfeatures/prices/trades/metrics)]

    External Data --> Ingest --> Features --> Labels --> Backtest --> App
    Backtest --> Store --> App
```

## Generate Data Artifacts
You can pre-compute the feature matrix, labels, and trades outside the UI. The snippet below writes results to `outputs/` (which is ignored by git).

```bash
python - <<'PY'
from alpha_pipeline import build_feature_matrix, train_and_backtest

ticker = "AAPL"
start = "2020-01-01"
end = "2024-12-31"

features, prices = build_feature_matrix(
    ticker=ticker,
    start=start,
    end=end,
    news_hours=12,
    out_dir="outputs"
)

metrics, trades = train_and_backtest(
    feats=features,
    px=prices,
    label_kind="barrier",
    tune_threshold=True,
    out_dir="outputs"
)

print(metrics)
PY
```
Artifacts saved:
- `outputs/features.csv` and `outputs/prices.csv`
- `outputs/trades.csv` and `outputs/labels.csv`
- `outputs/metrics.json`

These are the same files the Streamlit app loads after you click **Run Backtest**.

## Operational Notes
- No API keys or credentials are required; the project relies on public endpoints.
- Generated data, cached responses, and notebooks are excluded from version control via `.gitignore`.
- Suggested repository name for GitHub: `quantamental-signal-foundry`.
- For new research ideas, inspect `quant_pipeline.py` (slimmer runtime) or `alpha_pipeline.py` (research-focused, used by the app).
