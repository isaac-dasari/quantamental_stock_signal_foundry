# quant_pipeline.py — alpha-compatible outputs with robust internals
from __future__ import annotations
import os, json, time
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score

# =====================================
# Helpers: time, IO, dtype-safe ops
# =====================================
def _get_adj_series(px: pd.DataFrame, ticker: str | None = None) -> pd.Series:
    """
    Always return a 1-D numeric adjusted/close series from px, robust to MultiIndex.
    Prefers 'Adj Close' from a flattened OHLC; falls back to 'Close' or first numeric col.
    """
    # Always flatten first — never rely on px["AdjClose"] directly
    px_flat = _flatten_ohlc(px, ticker=ticker)

    if "Adj Close" in px_flat.columns:
        s = px_flat["Adj Close"]
    elif "Close" in px_flat.columns:
        s = px_flat["Close"]
    else:
        # last resort: take the first numeric column
        nc = px_flat.select_dtypes(include=[np.number])
        if nc.shape[1] == 0:
            raise ValueError("No numeric price column found after flattening OHLC.")
        s = nc.iloc[:, 0]

    return pd.to_numeric(s, errors="coerce")

def _naive_ts(x):
    """Return tz-naive pandas Timestamp (UTC -> naive) for safe comparisons."""
    return pd.to_datetime(x, utc=True).tz_convert(None)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def logret(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return np.log(s / s.shift(1))

def compute_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

# =====================================
# OHLCV normalization (handles MultiIndex)
# =====================================

_FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "AdjClose", "Volume"}

def _standardize_cols(cols):
    """Map common alt names to canonical ones (keep 'Adj Close' w/ space)."""
    rename_map = {
        "Adj_Close": "Adj Close", "adjclose": "Adj Close", "AdjClose": "Adj Close",
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
    }
    return [rename_map.get(str(c), str(c)) for c in cols]

def _flatten_ohlc(px: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """
    Return a 1-level OHLCV DataFrame with columns:
    Open, High, Low, Close, Adj Close, Volume (subset depending on availability).
    Works for single-level columns or MultiIndex in either order (field,symbol) / (symbol,field).
    """
    if isinstance(px, pd.Series):
        px = px.to_frame()

    df = px.copy()
    if df.empty:
        return pd.DataFrame(columns=["Open","High","Low","Close","Adj Close","Volume"])

    # Single-level columns
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = _standardize_cols(df.columns)
        keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
        df = df[keep]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    # MultiIndex: detect which level is fields vs symbols
    lvl0_vals = set(map(str, df.columns.get_level_values(0)))
    lvl1_vals = set(map(str, df.columns.get_level_values(1)))
    if _FIELDS & lvl0_vals:
        field_level, symbol_level = 0, 1
    elif _FIELDS & lvl1_vals:
        field_level, symbol_level = 1, 0
    else:
        # Fallback: best-effort pick any columns containing field names
        cols_flat = ["|".join(map(str, t)) for t in df.columns]
        mask = [any(f in cf for f in _FIELDS) for cf in cols_flat]
        df = df.loc[:, mask]
        df.columns = [str(c) for c in df.columns]
        out = pd.DataFrame(index=df.index)
        for f in _FIELDS:
            candidates = [c for c in df.columns if c.endswith("|"+f) or c.startswith(f+"|") or c == f]
            if candidates:
                out[f] = pd.to_numeric(df[candidates[0]], errors="coerce")
        out.columns = _standardize_cols(out.columns)
        keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in out.columns]
        return out[keep]

    # Select symbol slice
    symbols = df.columns.get_level_values(symbol_level).unique().tolist()
    if ticker is not None and str(ticker) in set(map(str, symbols)):
        sym_to_use = ticker
    elif len(symbols) == 1:
        sym_to_use = symbols[0]
    else:
        sym_to_use = symbols[0]  # deterministic fallback

    df_sel = df.xs(key=sym_to_use, level=symbol_level, axis=1)
    df_sel.columns = _standardize_cols(df_sel.columns)
    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df_sel.columns]
    df_sel = df_sel[keep]
    for c in df_sel.columns:
        df_sel[c] = pd.to_numeric(df_sel[c], errors="coerce")
    return df_sel

# =====================================
# Data fetchers (alpha-compatible)
# =====================================

def get_prices(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No price data for {ticker}")
    df.index = pd.to_datetime(df.index)
    # alpha expects 'AdjClose' (no space) — keep that convention for the UI
    if "Adj Close" in df.columns and "AdjClose" not in df.columns:
        df = df.rename(columns={"Adj Close": "AdjClose"})
    return df

def get_macro(start: str, end: str) -> pd.DataFrame:
    frames = {}
    for tk in ["SPY","QQQ","^VIX"]:
        try:
            df = get_prices(tk, start, end)
            col = "AdjClose" if "AdjClose" in df.columns else "Close"
            frames[tk] = df[[col]].rename(columns={col: f"{tk}_AdjClose"})
        except Exception:
            pass
    for dxy_tk in ["DX=F","UUP","^DXY"]:
        try:
            df = get_prices(dxy_tk, start, end)
            col = "AdjClose" if "AdjClose" in df.columns else "Close"
            frames["DXY"] = df[[col]].rename(columns={col: "DXY_AdjClose"})
            break
        except Exception:
            continue
    macro = None
    for _, d in frames.items():
        macro = d if macro is None else macro.join(d, how="outer")
    if macro is None:
        raise ValueError("No macro series fetched")
    return macro

def get_actions_flags(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Return alpha-compatible columns:
      - 'dividend' (0/1), 'split' (0/1), indexed by daily dates.
    """
    s = _naive_ts(start).normalize()
    e = _naive_ts(end).normalize()
    idx = pd.date_range(s, e, freq="D")
    out = pd.DataFrame(index=idx, data={"dividend": 0, "split": 0})

    try:
        t = yf.Ticker(ticker)
        acts = getattr(t, "actions", None)
        if acts is None or acts.empty:
            return out
        acts = acts.copy()
        acts.index = pd.to_datetime(acts.index, utc=True).tz_convert(None).normalize()
        acts = acts.loc[(acts.index >= s) & (acts.index <= e)]
        if acts.empty:
            return out
        if "Dividends" in acts.columns:
            div_days = acts.index[acts["Dividends"].fillna(0) > 0].unique()
            out.loc[out.index.isin(div_days), "dividend"] = 1
        if "Stock Splits" in acts.columns:
            split_days = acts.index[acts["Stock Splits"].fillna(0) > 0].unique()
            out.loc[out.index.isin(split_days), "split"] = 1
        return out
    except Exception:
        return out

# =====================================
# Features (alpha-compatible signals)
# =====================================

def overnight_intraday_features(px: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """
    Return ret_overnight, ret_intraday using robust OHLC selection.
    """
    if px is None or len(px) == 0:
        return pd.DataFrame()
    df = _flatten_ohlc(px, ticker=ticker).sort_index()
    out = pd.DataFrame(index=df.index)
    if {"Open","Close"}.issubset(df.columns):
        out["ret_overnight"] = np.log(df["Open"] / df["Close"].shift(1))
        out["ret_intraday"]  = np.log(df["Close"] / df["Open"])
    else:
        out["ret_overnight"] = np.nan
        out["ret_intraday"]  = np.nan
    return out

def garman_klass(px: pd.DataFrame) -> pd.Series:
    df = _flatten_ohlc(px)
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])
    return 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2

def parkinson(px: pd.DataFrame) -> pd.Series:
    df = _flatten_ohlc(px)
    return (1.0 / (4*np.log(2))) * (np.log(df["High"] / df["Low"])) ** 2

def momentum_features(adj_like: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """
    Accept Series or DataFrame; looks for 'AdjClose'/'Adj Close'/'Close'.
    If none found, falls back to the first numeric column.
    """
    if isinstance(adj_like, pd.DataFrame):
        if "AdjClose" in adj_like.columns:
            adj = adj_like["AdjClose"]
        elif "Adj Close" in adj_like.columns:
            adj = adj_like["Adj Close"]
        elif "Close" in adj_like.columns:
            adj = adj_like["Close"]
        else:
            nc = adj_like.select_dtypes(include=[np.number])
            if nc.shape[1] == 0:
                # final fallback: try flatten then pick first numeric
                df = _flatten_ohlc(adj_like)
                nc = df.select_dtypes(include=[np.number])
            if nc.shape[1] == 0:
                # return empty frame rather than crash
                return pd.DataFrame(index=adj_like.index)
            adj = nc.iloc[:, 0]
    else:
        adj = adj_like

    adj = pd.to_numeric(adj, errors="coerce")
    out = pd.DataFrame(index=adj.index)
    out["ret_1d"] = logret(adj)
    out["ret_5d"] = np.log(adj / adj.shift(5))
    out["ret_21d"] = np.log(adj / adj.shift(21))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["mom_20_100"] = adj / adj.rolling(100).mean() - adj / adj.rolling(20).mean()
    out["rsi_14"] = compute_rsi(adj, 14)
    return out

def rolling_beta(asset_ret: pd.Series, mkt_ret: pd.Series, window: int = 63) -> pd.Series:
    asset_ret = pd.to_numeric(asset_ret, errors="coerce")
    mkt_ret = pd.to_numeric(mkt_ret.reindex(asset_ret.index), errors="coerce")
    cov = asset_ret.rolling(window).cov(mkt_ret)
    var = mkt_ret.rolling(window).var()
    return cov / (var.replace(0, np.nan))

# =====================================
# News (GDELT) + sentiment
# =====================================

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
_vader = SentimentIntensityAnalyzer()

def ticker_to_query(ticker: str) -> str:
    t = ticker.upper()
    mapping = {
        "AAPL": '(Apple OR "Apple Inc" OR iPhone OR "Tim Cook")',
        "MSFT": '(Microsoft OR "Microsoft Corp" OR Windows OR "Satya Nadella")',
        "AMZN": '(Amazon OR "Amazon.com" OR AWS OR "Andy Jassy")',
        "GOOGL": '(Google OR Alphabet OR "Sundar Pichai")',
        "TSLA": '(Tesla OR "Elon Musk" OR "Model 3")',
    }
    return mapping.get(t, t)

def vader_score(text: str) -> float:
    if not isinstance(text, str) or not text:
        return np.nan
    return _vader.polarity_scores(text)["compound"]

def gdelt_fetch(query: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp, max_records: int = 50, timeout: int = 20) -> pd.DataFrame:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max(1, min(int(max_records), 50)),
        "sort": "DateDesc",
        "startdatetime": pd.Timestamp(start_utc).strftime("%Y%m%d%H%M%S"),
        "enddatetime":  pd.Timestamp(end_utc).strftime("%Y%m%d%H%M%S"),
    }
    headers = {"User-Agent": "quant-pipeline/1.0 (+contact: you@example.com)"}
    try:
        r = requests.get(GDELT_DOC_ENDPOINT, params=params, timeout=timeout, headers=headers)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
    except Exception:
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

def aggregate_news_sentiment(ticker: str, start: str, end: str, hours: int = 12) -> pd.DataFrame:
    start_dt = _naive_ts(start).tz_localize("UTC")
    end_dt   = _naive_ts(end).tz_localize("UTC")
    all_days = pd.date_range(start_dt.normalize(), end_dt.normalize(), freq="D", tz="UTC")
    rows = []
    query = ticker_to_query(ticker)
    for day in all_days:
        st = day - pd.Timedelta(hours=hours/2)
        en = day + pd.Timedelta(hours=hours/2)
        df = gdelt_fetch(query, st, en, max_records=50)
        if df.empty:
            rows.append({"date": day.date(), "news_count": 0, "news_unique_titles": 0, "news_unique_domains": 0,
                         "sent_mean": np.nan, "sent_min": np.nan, "sent_max": np.nan, "sent_std": np.nan})
            continue
        titles = df["title"].fillna("")
        df["title_norm"] = titles.str.lower().str.strip()
        scores = titles.apply(vader_score).values
        rows.append({
            "date": day.date(),
            "news_count": int(len(scores)),
            "news_unique_titles": int(df["title_norm"].nunique()),
            "news_unique_domains": int(df["domain"].nunique()),
            "sent_mean": float(np.nanmean(scores)),
            "sent_min": float(np.nanmin(scores)),
            "sent_max": float(np.nanmax(scores)),
            "sent_std": float(np.nanstd(scores)),
        })
        time.sleep(0.15)
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    out.set_index("date", inplace=True)
    return out

def earnings_flags(ticker: str, start: str, end: str, window: int = 1) -> pd.DataFrame:
    s = _naive_ts(start); e = _naive_ts(end)
    t = yf.Ticker(ticker)
    try:
        ed = t.get_earnings_dates(limit=60)
        ed.index = pd.to_datetime(ed.index, utc=True).tz_convert(None)
        ed = ed[(ed.index >= s) & (e >= ed.index)]
    except Exception:
        ed = pd.DataFrame()
    idx = pd.date_range(s, e, freq="D")
    flags = pd.DataFrame(index=idx, data={"earnings_day": 0})
    for d in ed.index.normalize():
        for k in range(-window, window + 1):
            day = (pd.Timestamp(d) + pd.Timedelta(days=k)).normalize()
            if day in flags.index:
                flags.loc[day, "earnings_day"] = 1
    return flags

# =====================================
# Labels + training (unchanged API)
# =====================================

def triple_barrier_labels(prices: pd.Series, vol: pd.Series, up_mult: float = 1.5, dn_mult: float = 1.5, max_holding: int = 7) -> pd.Series:
    prices = pd.to_numeric(prices, errors="coerce")
    vol = pd.to_numeric(vol.reindex(prices.index), errors="coerce").ffill().bfill()
    n = len(prices)
    labels = pd.Series(index=prices.index, dtype="float64")
    for i in range(n - 1):
        p0 = prices.iat[i]; sigma = vol.iat[i]
        if not (np.isfinite(p0) and np.isfinite(sigma)):
            labels.iat[i] = np.nan; continue
        up = p0 * (1 + sigma * up_mult)
        dn = p0 * (1 - sigma * dn_mult)
        j_end = min(i + max_holding, n - 1)
        path = prices.iloc[i + 1: j_end + 1]
        lab = 0
        if (path >= up).any(): lab = 1
        elif (path <= dn).any(): lab = -1
        labels.iat[i] = lab
    return labels

def _metrics_from_preds(y: pd.Series, proba: pd.Series, signal: pd.Series, strat: pd.Series) -> dict:
    y_true = y.reindex(proba.index).fillna(0).astype(int)
    y_hat = (proba > 0.5).astype(int)
    try: acc = float(accuracy_score(y_true, y_hat))
    except Exception: acc = float("nan")
    try: roc = float(roc_auc_score(y_true, proba.fillna(0)))
    except Exception: roc = float("nan")
    try: prec = float(precision_score(y_true, y_hat, zero_division=0))
    except Exception: prec = float("nan")
    n = int(strat.dropna().shape[0])
    if n == 0:
        ann_ret = ann_vol = sharpe = float("nan")
    else:
        cum_log = float(np.nansum(strat))
        ann_ret = float(np.exp(cum_log * (252.0 / n)) - 1.0)
        ann_vol = float(np.nanstd(strat) * np.sqrt(252.0))
        sharpe = float((np.nanmean(strat) / (np.nanstd(strat) + 1e-12)) * np.sqrt(252.0))
    return {
        "accuracy": acc, "roc_auc": roc, "precision_long": prec,
        "ann_return_%": ann_ret*100 if np.isfinite(ann_ret) else float("nan"),
        "ann_vol_%": ann_vol*100 if np.isfinite(ann_vol) else float("nan"),
        "sharpe": sharpe if np.isfinite(sharpe) else float("nan"),
    }

def train_and_backtest(
    feats: pd.DataFrame,
    prices: pd.Series | pd.DataFrame,
    label_kind: str = "barrier",
    barrier_up: float = 1.5,
    barrier_dn: float = 1.5,
    barrier_holding: int = 7,
    tune_threshold: bool = True,
    threshold: float = 0.55,
    trade_excess: bool = True,
    out_dir: Optional[str] = None
):
    if isinstance(prices, pd.DataFrame):
        col = "AdjClose" if "AdjClose" in prices.columns else "Close"
        P = pd.to_numeric(prices[col], errors="coerce")
    else:
        P = pd.to_numeric(prices, errors="coerce")

    if label_kind == "nextday":
        y = (P.pct_change().shift(-1) > 0).astype(int)
    else:
        daily_vol = np.sqrt(feats["gk_var"]).ewm(span=21, adjust=False).mean()
        tb = triple_barrier_labels(P, daily_vol, up_mult=barrier_up, dn_mult=barrier_dn, max_holding=barrier_holding)
        y = (tb == 1).astype(int)

    X = feats.select_dtypes(include=[np.number]).copy()
    X = X.loc[:, X.nunique(dropna=True) > 1]
    idx = X.index.intersection(y.index).intersection(P.index)
    X, y, P = X.loc[idx], y.loc[idx], P.loc[idx]

    ret = np.log(P / P.shift(1)).fillna(0.0)
    if trade_excess and ("mkt_ret" in X.columns):
        ret = ret - X["mkt_ret"].fillna(0.0)

    if len(X) < 30 or X.shape[1] == 0 or y.nunique() < 2:
        proba = pd.Series(0.5, index=X.index)
        signal = (proba > threshold).astype(float)
        strat = signal.shift(1).fillna(0) * ret
        metrics = _metrics_from_preds(y, proba, signal, strat)
        trades = pd.DataFrame({"proba": proba, "signal": signal, "ret": ret, "strat": strat})
        if out_dir:
            _ensure_dir(out_dir)
            trades.to_csv(os.path.join(out_dir, "trades.csv"))
            y.to_frame("label").to_csv(os.path.join(out_dir, "labels.csv"))
            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
        return metrics, trades

    n_splits = 5 if len(X) >= 250 else 3
    tscv = TimeSeriesSplit(n_splits=n_splits)
    embargo = 5
    oof_proba = pd.Series(index=X.index, dtype=float)

    for train_idx, test_idx in tscv.split(X):
        if len(train_idx) > embargo:
            train_idx = train_idx[:-embargo]
        Xt, Xs = X.iloc[train_idx], X.iloc[test_idx]
        yt, ys = y.iloc[train_idx], y.iloc[test_idx]

        clf = GradientBoostingClassifier(random_state=42)
        try:
            clf.fit(Xt, yt)
            proba = clf.predict_proba(Xs)[:, 1]
        except Exception:
            logit = LogisticRegression(max_iter=300, class_weight="balanced")
            logit.fit(Xt, yt)
            proba = logit.predict_proba(Xs)[:, 1]
        oof_proba.iloc[test_idx] = proba

    if tune_threshold:
        candidates = [0.40, 0.45, 0.50, 0.55, 0.60]
        best_t, best_sh = threshold, -1e9
        for t in candidates:
            sig = (oof_proba > t).astype(float)
            strat_oof = sig.shift(1).fillna(0) * ret
            mu, sd = np.nanmean(strat_oof), np.nanstd(strat_oof)
            sh = (mu / (sd + 1e-12)) * np.sqrt(252.0) if sd > 0 else -1e9
            if sh > best_sh:
                best_sh, best_t = sh, t
        threshold = best_t

    signal = (oof_proba > threshold).astype(float)
    signal.name = threshold
    strat = signal.shift(1).fillna(0) * ret

    metrics = _metrics_from_preds(y, oof_proba, signal, strat)
    trades = pd.DataFrame({"proba": oof_proba, "signal": signal, "ret": ret, "strat": strat})

    if out_dir:
        _ensure_dir(out_dir)
        trades.to_csv(os.path.join(out_dir, "trades.csv"))
        y.to_frame("label").to_csv(os.path.join(out_dir, "labels.csv"))
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics, trades

# =====================================
# Build feature matrix (alpha schema)
# =====================================

def build_feature_matrix(
    ticker: str,
    start: str,
    end: str,
    warmup_days: int = 200,
    news_hours: int = 12,
    out_dir: Optional[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Normalize windows
    start = _naive_ts(start)
    end   = _naive_ts(end)
    true_start = start
    warmup_start = start - pd.Timedelta(days=warmup_days)

    # Fetch with warmup to stabilize rolling features
    px_full = get_prices(ticker, warmup_start, end)
    macro   = get_macro(warmup_start, end)
    acts    = get_actions_flags(ticker, warmup_start, end)  # columns: dividend, split

    feats = pd.DataFrame(index=px_full.index)

    # OHLC-based features (robust even if yfinance returns MultiIndex)
    feats = feats.join(overnight_intraday_features(px_full, ticker=ticker))
    feats["gk_var"] = garman_klass(px_full)
    feats["parkinson_var"] = parkinson(px_full)

    # Momentum/RSI from alpha's adjusted naming
    px_flat = _flatten_ohlc(px_full, ticker=ticker)

    # if "AdjClose" in px_full.columns:
    #     # alpha-style column from get_prices
    #     adj_series = pd.to_numeric(px_full["AdjClose"], errors="coerce")
    # elif "Adj Close" in px_flat.columns:
    #     adj_series = pd.to_numeric(px_flat["Adj Close"], errors="coerce")
    # elif "Close" in px_flat.columns:
    #     adj_series = pd.to_numeric(px_flat["Close"], errors="coerce")
    # else:
    #     # last-resort: first numeric column from either px_full or px_flat
    #     cand = None
    #     if isinstance(px_full, pd.DataFrame) and not px_full.empty:
    #         nc = px_full.select_dtypes(include=[np.number])
    #         if nc.shape[1] > 0:
    #             cand = nc.iloc[:, 0]
    #     if cand is None and isinstance(px_flat, pd.DataFrame) and not px_flat.empty:
    #         nc = px_flat.select_dtypes(include=[np.number])
    #         if nc.shape[1] > 0:
    #             cand = nc.iloc[:, 0]
    #     if cand is None:
    #         raise ValueError("No numeric price column found in px_full/px_flat")
        # adj_series = pd.to_numeric(cand, errors="coerce")
    adj_series = _get_adj_series(px_full, ticker=ticker)

    feats = feats.join(momentum_features(adj_series))
    feats["ret_1d_asset"] = logret(adj_series)

    # Macro context (alpha-like column names)
    if "SPY_AdjClose" in macro.columns:
        feats["mkt_ret"] = logret(macro["SPY_AdjClose"])
    if "QQQ_AdjClose" in macro.columns:
        feats["qqq_ret"] = logret(macro["QQQ_AdjClose"])
    if "^VIX_AdjClose" in macro.columns:
        feats["vix_chg"] = logret(macro["^VIX_AdjClose"])
    if "DXY_AdjClose" in macro.columns:
        feats["dxy_chg"] = logret(macro["DXY_AdjClose"])

    # Asset daily returns & beta
    feats["ret_1d_asset"] = logret(adj_series)
    if "mkt_ret" in feats.columns:
        feats["beta_63"] = rolling_beta(feats["ret_1d_asset"], feats["mkt_ret"], window=63)

    # Earnings (daily) & actions (daily) — align by normalized date
    earn = earnings_flags(ticker, warmup_start, end, window=1)
    norm_idx = feats.index.normalize()
    feats["earnings_day"] = earn.reindex(norm_idx).fillna(0).astype(int)
    feats["dividend"] = acts["dividend"].reindex(norm_idx).fillna(0).astype(int)
    feats["split"]    = acts["split"].reindex(norm_idx).fillna(0).astype(int)

    # News sentiment only over requested [start, end] (not warmup)
    try:
        news = aggregate_news_sentiment(ticker, start, end, hours=news_hours)
        news.columns = [f"news_{c}" for c in news.columns]
        for c in news.columns:
            feats[c] = news[c].reindex(norm_idx)
    except Exception:
        pass

    # Clean & clip back to requested window
    feats = feats.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    feats = feats.loc[feats.index >= true_start]
    px = px_full.loc[px_full.index >= true_start]  # what UI expects to plot

    if out_dir:
        _ensure_dir(out_dir)
        feats.to_csv(os.path.join(out_dir, "features.csv"))
        px.to_csv(os.path.join(out_dir, "prices.csv"))

    return feats, px
