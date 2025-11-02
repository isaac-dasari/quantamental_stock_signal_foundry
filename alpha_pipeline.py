# alpha_pipeline.py — Research-Grade Quantamental Pipeline (AAPL example)
# See usage examples above.

from __future__ import annotations
import os, sys, json
import typing as T
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

# 3rd party
import requests
from dateutil import parser as dateparser
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

# -----------------------------
# Paths / constants
# -----------------------------
HERE = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
CACHE_DIR = os.path.join(HERE, "cache")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

# -----------------------------
# Utilities
# -----------------------------
def write_parquet_or_csv(df: pd.DataFrame, base_name: str, index: bool = True) -> str:
    """Try to write Parquet; if no engine, write CSV instead."""
    pq_path = os.path.join(OUT_DIR, f"{base_name}.parquet")
    csv_path = os.path.join(OUT_DIR, f"{base_name}.csv")
    try:
        try:
            import pyarrow  # noqa: F401
            engine = "pyarrow"
        except Exception:
            engine = None
        if engine:
            df.to_parquet(pq_path, index=index, engine=engine)
        else:
            df.to_parquet(pq_path, index=index)  # may use fastparquet if present
        return pq_path
    except Exception:
        df.to_csv(csv_path, index=index)
        return csv_path

def to_utc(dt: T.Union[str, datetime]) -> datetime:
    if isinstance(dt, str):
        dt = dateparser.parse(dt)
    if dt.tzinfo is None:
        dt = dt.astimezone()  # local
    return dt.astimezone(pd.Timestamp.utcnow().tzinfo)

# -----------------------------
# Data: Prices & Macro
# -----------------------------
def get_prices(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No price data for {ticker}")
    df.index = pd.to_datetime(df.index)
    df.rename(columns={"Adj Close": "AdjClose"}, inplace=True)
    return df

def get_actions_flags(ticker: str, start: str, end: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    acts = t.actions
    s = pd.to_datetime(start).normalize()
    e = pd.to_datetime(end).normalize()
    out = pd.DataFrame(index=pd.date_range(s, e, freq="D"))
    out["dividend"] = 0
    out["split"] = 0
    if acts is None or acts.empty:
        return out
    acts = acts.copy()
    idx = pd.to_datetime(acts.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    acts.index = idx.normalize()
    acts = acts.loc[(acts.index >= s) & (acts.index <= e)]
    if acts.empty:
        return out
    if "Dividends" in acts.columns:
        div_days = acts.index[acts["Dividends"] > 0].unique()
        out.loc[out.index.isin(div_days), "dividend"] = 1
    if "Stock Splits" in acts.columns:
        split_days = acts.index[acts["Stock Splits"] > 0].unique()
        out.loc[out.index.isin(split_days), "split"] = 1
    return out

def get_macro(start: str, end: str) -> pd.DataFrame:
    frames = {}
    for tk in ["SPY", "QQQ", "^VIX"]:
        try:
            df = get_prices(tk, start, end)
            frames[tk] = df[["AdjClose"]].rename(columns={"AdjClose": f"{tk}_AdjClose"})
        except Exception:
            pass
    for dxy_tk in ["DX=F", "UUP", "^DXY"]:  # robust DXY
        try:
            df = get_prices(dxy_tk, start, end)
            frames["DXY"] = df[["AdjClose"]].rename(columns={"AdjClose": "DXY_AdjClose"})
            break
        except Exception:
            continue
    if pdr is not None:
        try:
            dgs10 = pdr.DataReader("DGS10", "fred", start, end)
            frames["DGS10"] = dgs10.rename(columns={"DGS10": "DGS10"})
        except Exception:
            pass
    macro = None
    for _, df in frames.items():
        macro = df if macro is None else macro.join(df, how="outer")
    if macro is None:
        raise ValueError("No macro series fetched")
    return macro

# -----------------------------
# Technical features
# -----------------------------
def logret(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1))

def overnight_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ret_overnight"] = np.log(df["Open"] / df["Close"].shift(1))
    out["ret_intraday"] = np.log(df["Close"] / df["Open"])
    return out

def garman_klass(df: pd.DataFrame) -> pd.Series:
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])
    return 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2

def parkinson(df: pd.DataFrame) -> pd.Series:
    return (1.0 / (4 * np.log(2))) * (np.log(df["High"] / df["Low"])) ** 2

def compute_rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def momentum_features(adj: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=adj.index)
    out["ret_1d"] = logret(adj)
    out["ret_5d"] = np.log(adj / adj.shift(5))
    out["ret_21d"] = np.log(adj / adj.shift(21))
    out["mom_20_100"] = adj / adj.rolling(100).mean() - adj / adj.rolling(20).mean()
    out["rsi_14"] = compute_rsi(adj, 14)
    return out

def rolling_beta(asset_ret: pd.Series, market_ret: pd.Series, window: int = 63) -> pd.Series:
    df = pd.concat([asset_ret, market_ret], axis=1).dropna()
    betas = pd.Series(index=asset_ret.index, dtype=float)
    for i in range(window, len(df) + 1):
        sub = df.iloc[i - window : i]
        y = sub.iloc[:, 0].values
        x = sm.add_constant(sub.iloc[:, 1].values)
        try:
            beta = sm.OLS(y, x).fit().params[1]
        except Exception:
            beta = np.nan
        betas.loc[sub.index[-1]] = beta
    return betas

# -----------------------------
# News & Sentiment (GDELT)
# -----------------------------
@dataclass
class Article:
    seendate: datetime
    title: str
    url: str
    domain: str
    language: str
    sourcecountry: str

def ticker_to_query(ticker: str) -> str:
    def or_group(terms: list[str]) -> str:
        q = " OR ".join(f'"{t}"' if not (t.startswith('"') and t.endswith('"')) else t for t in terms)
        return f"({q})"
    t = ticker.upper()
    mapping = {
        "AAPL": or_group(["Apple", "Apple Inc", "iPhone", "Tim Cook"]),
        "MSFT": or_group(["Microsoft", "Microsoft Corp", "Windows", "Satya Nadella"]),
        "AMZN": or_group(["Amazon", "Amazon.com", "AWS", "Andy Jassy"]),
        "GOOGL": or_group(["Google", "Alphabet", "Sundar Pichai"]),
        "TSLA": or_group(["Tesla", "Elon Musk", "Model 3"]),
    }
    return mapping.get(t, t)

def gdelt_fetch(query: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp, max_records: int = 200, timeout: int = 20) -> pd.DataFrame:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max(1, min(int(max_records), 200)),
        "sort": "DateDesc",
        "startdatetime": pd.Timestamp(start_utc).strftime("%Y%m%d%H%M%S"),
        "enddatetime": pd.Timestamp(end_utc).strftime("%Y%m%d%H%M%S"),
    }
    headers = {"User-Agent": "alpha-pipeline/1.0 (+contact: you@example.com)"}
    try:
        r = requests.get(GDELT_DOC_ENDPOINT, params=params, timeout=timeout, headers=headers)
    except requests.RequestException:
        return pd.DataFrame()
    if r.status_code != 200:
        return pd.DataFrame()
    try:
        data = r.json()
    except ValueError:
        return pd.DataFrame()
    arts = data.get("articles", []) if isinstance(data, dict) else []
    rows = [{
        "seendate": pd.to_datetime(a.get("seendate"), utc=True, errors="coerce"),
        "title": a.get("title"),
        "url": a.get("url"),
        "domain": a.get("domain"),
        "language": a.get("language"),
        "sourcecountry": a.get("sourcecountry"),
    } for a in arts]
    return pd.DataFrame(rows)

_vader = SentimentIntensityAnalyzer()

def vader_score(text: str) -> float:
    if not isinstance(text, str) or not text:
        return np.nan
    return _vader.polarity_scores(text)["compound"]

# Optional FinBERT (lazy)
_finbert_available = False
_finbert_model = _finbert_tokenizer = None

def maybe_init_finbert() -> None:
    global _finbert_available, _finbert_model, _finbert_tokenizer
    if _finbert_available: return
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch  # noqa: F401
        model_name = "ProsusAI/finbert"
        _finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _finbert_available = True
    except Exception:
        _finbert_available = False

def finbert_score_batch(texts: list[str]) -> np.ndarray:
    import torch
    enc = _finbert_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = _finbert_model(**enc).logits
        probs = torch.softmax(logits, dim=1).numpy()
    return probs[:, 2] - probs[:, 0]

def aggregate_news_sentiment(ticker: str, start: str, end: str, hours: int = 24, use_finbert: bool = False) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    all_days = pd.date_range(start_dt, end_dt, freq="D", tz="UTC")
    rows = []
    query = ticker_to_query(ticker)
    for day in all_days:
        st = (day - pd.Timedelta(hours=hours / 2))
        en = (day + pd.Timedelta(hours=hours / 2))
        df = gdelt_fetch(query, st, en)
        if df.empty:
            rows.append({"date": day.date(), "news_count": 0, "sent_mean": np.nan, "sent_min": np.nan, "sent_max": np.nan, "sent_std": np.nan})
            continue
        titles = df["title"].fillna("")
        if use_finbert:
            if not _finbert_available:
                maybe_init_finbert()
            if _finbert_available:
                scores = finbert_score_batch(titles.tolist())
            else:
                scores = titles.apply(vader_score).values
        else:
            scores = titles.apply(vader_score).values
        rows.append({
            "date": day.date(),
            "news_count": len(scores),
            "sent_mean": np.nanmean(scores),
            "sent_min": np.nanmin(scores),
            "sent_max": np.nanmax(scores),
            "sent_std": np.nanstd(scores),
        })
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    out.set_index("date", inplace=True)
    return out

# -----------------------------
# Events & Labels
# -----------------------------
def earnings_flags(ticker: str, start: str, end: str, window: int = 1) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    try:
        ed = t.get_earnings_dates(limit=60)
        ed = ed[(ed.index >= start) & (ed.index <= end)]
    except Exception:
        ed = pd.DataFrame()
    idx = pd.date_range(start, end)
    flags = pd.DataFrame(index=idx, data={"earnings_day": 0})
    for d in ed.index.normalize():
        for k in range(-window, window + 1):
            day = (pd.Timestamp(d) + pd.Timedelta(days=k)).normalize()
            if day in flags.index:
                flags.loc[day, "earnings_day"] = 1
    return flags

def triple_barrier_labels(
    prices: pd.Series | pd.DataFrame,
    vol: pd.Series | pd.DataFrame,
    up_mult: float = 2.0,
    dn_mult: float = 2.0,
    max_holding: int = 5,
) -> pd.Series:
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError("`prices` must be a Series or single-column DataFrame.")
        prices = prices.iloc[:, 0]
    if isinstance(vol, pd.DataFrame):
        if vol.shape[1] != 1:
            raise ValueError("`vol` must be a Series or a single-column DataFrame.")
        vol = vol.iloc[:, 0]
    prices = pd.to_numeric(prices.squeeze(), errors="coerce")
    vol = pd.to_numeric(vol.reindex(prices.index).squeeze(), errors="coerce").ffill().bfill()
    n = len(prices)
    labels = pd.Series(index=prices.index, dtype="float64")
    for i in range(n - 1):
        p0 = float(prices.iat[i]) if pd.notna(prices.iat[i]) else np.nan
        sigma = float(vol.iat[i]) if pd.notna(vol.iat[i]) else np.nan
        if not (np.isfinite(p0) and np.isfinite(sigma)):
            labels.iat[i] = np.nan
            continue
        up = p0 * (1 + sigma * up_mult)
        dn = p0 * (1 - sigma * dn_mult)
        j_end = min(i + max_holding, n - 1)
        path = prices.iloc[i + 1 : j_end + 1]
        lab = 0
        if (path >= up).any():
            lab = 1
        elif (path <= dn).any():
            lab = -1
        labels.iat[i] = lab
    return labels

# -----------------------------
# Modeling & Backtest
# -----------------------------
def train_and_backtest(
    

    feats: pd.DataFrame,
    px: pd.Series | pd.DataFrame,
    label_kind: str = "barrier",
    barrier_up: float = 1.5,
    barrier_dn: float = 1.5,
    barrier_holding: int = 7,
    tune_threshold: bool = True,
    threshold: float = 0.55,
    trade_excess: bool = True,
    out_dir: Optional[str] = None
) -> tuple[dict, pd.Series, pd.Series]:

    prices=px["AdjClose"]
     # Build training labels based on --target
    if label_kind == "barrier":
        daily_vol = np.sqrt(feats["gk_var"]).rolling(21).mean()
        labels = triple_barrier_labels(prices, daily_vol, up_mult=barrier_up, dn_mult=barrier_dn, max_holding=barrier_holding)
        y = (labels == 1).astype(int)
        label_path = write_parquet_or_csv(y.to_frame("label"), "labels")
        print(f"Saved: {label_path}")
    else:  # nextday up/down
        y = (prices.pct_change().shift(-1) > 0).astype(int).reindex(feats.index)
        label_path = write_parquet_or_csv(y.to_frame("label"), "labels")
        print(f"Saved: {label_path}")

    # Train + backtest
    mkt_ret = feats["mkt_ret"] if "mkt_ret" in feats.columns else None
    # --- Coerce prices to 1-D numeric Series ---
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError("`prices` must be a Series or single-column DataFrame.")
        prices = prices.iloc[:, 0]
    P = pd.to_numeric(prices, errors="coerce")

    # --- NUMERIC-ONLY features, clean & de-constant ---
    X = (
        feats.select_dtypes(include=[np.number])
           .replace([np.inf, -np.inf], np.nan)
           .ffill().bfill()
    )
    X = X.loc[:, X.nunique(dropna=True) > 1]  # drop constant columns
    # Drop columns that are entirely NaN or constant
    valid_cols = [c for c in X.columns if X[c].notna().any() and X[c].nunique(dropna=True) > 1]
    X = X[valid_cols]

    # --- Align indices & labels ---
    y = y.astype(int)
    idx = X.index.intersection(y.index).intersection(P.index)
    X = X.loc[idx]
    y = y.loc[idx]
    P = P.loc[idx]

    n_samples = len(X)
    if n_samples < 5 or X.shape[1] == 0:
        # Not enough data or no usable features: return neutral outputs
        preds = pd.Series(0.5, index=idx)
        ret = np.log(P / P.shift(1)).fillna(0)
        if trade_excess and (mkt_ret is not None):
            ret = ret - mkt_ret.reindex(ret.index).fillna(0)
        signal = (preds > threshold).astype(float)
        strat = signal.shift(1).fillna(0) * ret
        ann_ret = np.exp(np.nansum(strat) * (252.0 / max(1, strat.dropna().shape[0]))) - 1.0 if strat.any() else np.nan
        ann_vol = float(np.nanstd(strat) * np.sqrt(252.0)) if strat.any() else np.nan
        sharpe = float((np.nanmean(strat) / (np.nanstd(strat) + 1e-12)) * np.sqrt(252.0)) if strat.any() else np.nan
        metrics = {
            "threshold": threshold,
            "accuracy": np.nan,
            "roc_auc": np.nan,
            "precision_long": np.nan,
            "ann_return_%": ann_ret * 100 if np.isfinite(ann_ret) else np.nan,
            "ann_vol_%": ann_vol * 100 if np.isfinite(ann_vol) else np.nan,
            "sharpe": sharpe if np.isfinite(sharpe) else np.nan,
        }
        pd.DataFrame({"proba": preds, "signal": signal, "ret": ret, "strat": strat}).to_csv(os.path.join(OUT_DIR, "trades.csv"))
        trades = pd.DataFrame({"proba": oof_proba, "signal": signal, "ret": ret, "strat": strat})

        with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics, trades

    # --- Pipeline ---
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", GradientBoostingClassifier(random_state=42)),
    ])

    # Choose a safe number of splits for small samples
    # e.g., >= ~20 pts -> up to 5 splits; otherwise 2 splits
    n_splits = 5 if n_samples >= 50 else 2
    tscv = TimeSeriesSplit(n_splits=n_splits)

    preds = pd.Series(index=X.index, dtype=float)
    for train_idx, test_idx in tscv.split(X):
        Xt, Xs = X.iloc[train_idx], X.iloc[test_idx]
        yt, ys = y.iloc[train_idx], y.iloc[test_idx]
        pipe.fit(Xt, yt)
        proba = pipe.predict_proba(Xs)[:, 1]
        preds.iloc[test_idx] = proba

    # Optional threshold tuning by Sharpe (OOF) — enable by passing tune_threshold=True
    if tune_threshold:
        ret_oof = np.log(P / P.shift(1)).fillna(0)
        if trade_excess and (mkt_ret is not None):
            ret_oof = ret_oof - mkt_ret.reindex(ret_oof.index).fillna(0)
        candidates = [0.40, 0.45, 0.50, 0.55, 0.60]
        best_t, best_sharpe = threshold, -np.inf
        for t in candidates:
            sig = (preds > t).astype(float)
            strat_oof = sig.shift(1).fillna(0) * ret_oof
            mu, sd = np.nanmean(strat_oof), np.nanstd(strat_oof)
            sharpe_oof = (mu / (sd + 1e-12)) * np.sqrt(252.0) if sd > 0 else -np.inf
            if sharpe_oof > best_sharpe:
                best_sharpe, best_t = sharpe_oof, t
        threshold = best_t

    # --- Strategy & metrics on full OOF preds ---
    ret = np.log(P / P.shift(1)).fillna(0)
    if trade_excess and (mkt_ret is not None):
        ret = ret - mkt_ret.reindex(ret.index).fillna(0)
    signal = (preds > threshold).astype(float)
    strat = signal.shift(1).fillna(0) * ret

    n = int(strat.dropna().shape[0])
    if n == 0:
        ann_ret = np.nan; ann_vol = np.nan; sharpe = np.nan
    else:
        cum_log = float(np.nansum(strat))
        ann_ret = float(np.exp(cum_log * (252.0 / n)) - 1.0)
        ann_vol = float(np.nanstd(strat) * np.sqrt(252.0))
        sharpe = float((np.nanmean(strat) / (np.nanstd(strat) + 1e-12)) * np.sqrt(252.0))

    y_true = y.reindex(preds.index).fillna(0).astype(int)
    y_hat  = (preds > threshold).astype(int)

    try:
        acc = float(accuracy_score(y_true, y_hat))
    except Exception:
        acc = np.nan
    try:
        roc = float(roc_auc_score(y_true, preds.fillna(0)))
    except Exception:
        roc = np.nan
    try:
        prec_long = float(precision_score(y_true, y_hat, zero_division=0))
    except Exception:
        prec_long = np.nan

    trades = pd.DataFrame({"proba": preds, "signal": signal, "ret": ret, "strat": strat})
    trades.to_csv(os.path.join(OUT_DIR, "trades.csv"))

    metrics = {
        "threshold": threshold,
        "accuracy": acc,
        "roc_auc": roc,
        "precision_long": prec_long,
        "ann_return_%": ann_ret * 100 if np.isfinite(ann_ret) else np.nan,
        "ann_vol_%": ann_vol * 100 if np.isfinite(ann_vol) else np.nan,
        "sharpe": sharpe if np.isfinite(sharpe) else np.nan,
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics, trades


# -----------------------------
# Build Feature Matrix
# -----------------------------
def build_feature_matrix(
    ticker: str,
    start: str,
    end: str,
    news_hours: int = 24,
    use_finbert: bool = False,
    out_dir: Optional[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    px = get_prices(ticker, start, end)
    macro = get_macro(start, end)
    acts = get_actions_flags(ticker, start, end)

    feats = pd.DataFrame(index=px.index)
    feats = feats.join(overnight_intraday_features(px))
    feats["gk_var"] = garman_klass(px)
    feats["parkinson_var"] = parkinson(px)
    feats = feats.join(momentum_features(px["AdjClose"]))

    # Macro context
    if "SPY_AdjClose" in macro.columns:
        feats["mkt_ret"] = logret(macro["SPY_AdjClose"])
    if "QQQ_AdjClose" in macro.columns:
        feats["qqq_ret"] = logret(macro["QQQ_AdjClose"])
    if "^VIX_AdjClose" in macro.columns:
        feats["vix_chg"] = logret(macro["^VIX_AdjClose"])
    if "DXY_AdjClose" in macro.columns:
        feats["dxy_chg"] = logret(macro["DXY_AdjClose"])
    if "DGS10" in macro.columns:
        feats["dgs10_chg"] = macro["DGS10"].pct_change()

    # Rolling beta vs SPY
    feats["ret_1d_asset"] = logret(px["AdjClose"])
    if "mkt_ret" in feats.columns:
        feats["beta_63"] = rolling_beta(feats["ret_1d_asset"], feats["mkt_ret"])

    # Events
    feats = feats.join(earnings_flags(ticker, start, end, window=1))
    feats = feats.join(acts)

    # News sentiment (daily UTC), join by date
    try:
        news = aggregate_news_sentiment(ticker, start, end, hours=news_hours, use_finbert=use_finbert)
        news.columns = [f"news_{c}" for c in news.columns]
        feats = feats.join(news, on=pd.to_datetime(feats.index.date))
    except Exception:
        pass

    # Clean
    feats = feats.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # Save artifacts
    features_path = write_parquet_or_csv(feats, "features")
    prices_path   = write_parquet_or_csv(px, "prices")
    print(f"Saved: {features_path}\nSaved: {prices_path}")

    return feats, px

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Quantamental pipeline with sentiment + macro + technicals")
    p.add_argument("--ticker", type=str, default="AAPL")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--hours", type=int, default=24, help="news window hours around each day (UTC)")
    p.add_argument("--finbert", action="store_true", help="use FinBERT for headlines (requires torch/transformers)")
    p.add_argument("--target", type=str, default="barrier", choices=["barrier", "nextday"], help="training target")
    p.add_argument("--barrier_up", type=float, default=2.0)
    p.add_argument("--barrier_dn", type=float, default=2.0)
    p.add_argument("--barrier_holding", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--tune_threshold", action="store_true", help="grid-search threshold on OOF preds (F1)")
    p.add_argument("--excess", action="store_true", help="trade excess returns vs SPY")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    feats, px = build_feature_matrix(args.ticker, args.start, args.end, news_hours=args.hours, use_finbert=args.finbert)

    # Build training labels based on --target
    if args.target == "barrier":
        daily_vol = np.sqrt(feats["gk_var"]).rolling(21).mean()
        labels = triple_barrier_labels(px["AdjClose"], daily_vol, up_mult=args.barrier_up, dn_mult=args.barrier_dn, max_holding=args.barrier_holding)
        y_bin = (labels == 1).astype(int)
        label_path = write_parquet_or_csv(y_bin.to_frame("label"), "labels")
        print(f"Saved: {label_path}")
    else:  # nextday up/down
        y_bin = (px["AdjClose"].pct_change().shift(-1) > 0).astype(int).reindex(feats.index)
        label_path = write_parquet_or_csv(y_bin.to_frame("label"), "labels")
        print(f"Saved: {label_path}")

    # Train + backtest
    mkt_ret = feats["mkt_ret"] if "mkt_ret" in feats.columns else None
    metrics, preds, strat = train_and_backtest(
        feats=feats,
        px=px,
    )
    print(json.dumps(metrics, indent=2))

    # Plot (optional)
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            trades = pd.read_csv(os.path.join(OUT_DIR, "trades.csv"), parse_dates=True, index_col=0)
            (trades["strat"].cumsum().apply(np.exp)).plot(title="Strategy CumReturn (approx)")
            plt.show()
        except Exception:
            pass
