from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=300)
def get_expiries(ticker: str) -> List[str]:
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return []
    try:
        tk = yf.Ticker(symbol)
        return list(tk.options or [])
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_spot_price(ticker: str) -> Optional[float]:
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return None

    tk = yf.Ticker(symbol)
    price = None

    try:
        fast_info = getattr(tk, "fast_info", None)
        if fast_info is not None:
            price = fast_info.get("lastPrice")
    except Exception:
        price = None

    if price is None:
        try:
            hist = tk.history(period="1d", interval="1m")
            if not hist.empty:
                close = hist.get("Close", pd.Series(dtype=float)).dropna()
                if not close.empty:
                    price = float(close.iloc[-1])
        except Exception:
            price = None

    return float(price) if price is not None else None


@st.cache_data(ttl=300)
def get_option_chain(ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    symbol = (ticker or "").strip().upper()
    exp = (expiry or "").strip()
    empty = pd.DataFrame()

    if not symbol or not exp:
        return empty, empty

    try:
        tk = yf.Ticker(symbol)
        chain = tk.option_chain(exp)
        calls = chain.calls.copy() if chain.calls is not None else pd.DataFrame()
        puts = chain.puts.copy() if chain.puts is not None else pd.DataFrame()
        return calls, puts
    except Exception:
        return empty, empty


def compute_dte(expiry: str) -> Optional[int]:
    try:
        exp = datetime.strptime(expiry, "%Y-%m-%d").date()
    except Exception:
        return None

    today = datetime.now().date()
    dte = (exp - today).days
    return dte if dte > 0 else None


def parse_expiry_date(expiry: str):
    try:
        return datetime.strptime(expiry, "%Y-%m-%d").date()
    except Exception:
        return None


def _to_datetime_value(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime()
    except Exception:
        return None
    return None


@st.cache_data(ttl=3600)
def get_ticker_profile(ticker: str) -> dict:
    """
    Return profile dict:
    {
      "sector": str,
      "earnings_date": date|None,
      "earnings_status": "available" | "unavailable"
    }
    """
    symbol = (ticker or "").strip().upper()
    result = {
        "sector": "Unknown",
        "earnings_date": None,
        "earnings_status": "unavailable",
    }
    if not symbol:
        return result

    try:
        tk = yf.Ticker(symbol)
    except Exception:
        return result

    # Sector
    try:
        info = getattr(tk, "info", None)
        if isinstance(info, dict):
            sector = str(info.get("sector") or "").strip()
            if sector:
                result["sector"] = sector
    except Exception:
        pass

    now = datetime.now()

    # Earnings date source 1: calendar
    try:
        cal = tk.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty and "Earnings Date" in cal.index:
            row = cal.loc["Earnings Date"]
            dt = None
            if isinstance(row, pd.Series):
                for item in row.tolist():
                    dt = _to_datetime_value(item)
                    if dt is not None:
                        break
            else:
                dt = _to_datetime_value(row)
            if dt is not None:
                if dt.tzinfo is not None:
                    dt = dt.astimezone().replace(tzinfo=None)
                result["earnings_date"] = dt.date()
                result["earnings_status"] = "available"
                return result
    except Exception:
        pass

    # Earnings date source 2: earnings_dates
    try:
        edf = tk.get_earnings_dates(limit=12)
        if isinstance(edf, pd.DataFrame) and not edf.empty:
            idx = pd.to_datetime(edf.index, errors="coerce")
            idx = idx[~pd.isna(idx)]
            if len(idx) > 0:
                candidates = []
                for ts in idx:
                    dt = ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else None
                    if dt is None:
                        continue
                    if dt.tzinfo is not None:
                        dt = dt.astimezone().replace(tzinfo=None)
                    candidates.append(dt)
                if candidates:
                    future = [d for d in candidates if d >= now]
                    pick = min(future) if future else max(candidates)
                    result["earnings_date"] = pick.date()
                    result["earnings_status"] = "available"
    except Exception:
        pass

    return result


@st.cache_data(ttl=300)
def get_vix_history(cache_buster: int = 0) -> pd.DataFrame:
    """
    Return cleaned VIX daily close history for ~1 year.
    cache_buster is only used to force-refresh cache.
    """
    _ = cache_buster
    try:
        tk = yf.Ticker("^VIX")
        hist = tk.history(period="1y", interval="1d")
        if hist is None or hist.empty:
            return pd.DataFrame(columns=["Close"])

        out = hist.copy()
        if "Close" not in out.columns:
            return pd.DataFrame(columns=["Close"])

        out = out[["Close"]].copy()
        out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
        out = out.dropna(subset=["Close"]).sort_index()
        return out
    except Exception:
        return pd.DataFrame(columns=["Close"])


@st.cache_data(ttl=300)
def get_vix_snapshot(cache_buster: int = 0) -> dict:
    """
    Return VIX data dict:
    {
      "price": float|None,
      "prev_close": float|None,
      "change": float|None,
      "change_pct": float|None,
      "status": str,
    }
    """
    result = {
        "price": None,
        "prev_close": None,
        "change": None,
        "change_pct": None,
        "status": "N/A",
    }

    try:
        hist = get_vix_history(cache_buster=cache_buster)
        close = hist.get("Close", pd.Series(dtype=float)).dropna()
        if close.empty:
            return result

        price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else None

        change = None
        change_pct = None
        if prev_close is not None and prev_close != 0:
            change = price - prev_close
            change_pct = change / prev_close

        if price < 15:
            status = "低波动（偏平静）"
        elif price <= 25:
            status = "中性波动"
        else:
            status = "高波动（风险偏高）"

        result.update(
            {
                "price": price,
                "prev_close": prev_close,
                "change": change,
                "change_pct": change_pct,
                "status": status,
            }
        )
        return result
    except Exception:
        return result
