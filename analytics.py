from __future__ import annotations

from datetime import timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from data import compute_dte, get_expiries, get_option_chain, get_spot_price, get_ticker_profile, parse_expiry_date


def calculate_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> Optional[float]:
    try:
        if S is None or K is None or T is None or sigma is None:
            return None
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return None
        d1 = (np.log(S / K) + (r + (sigma**2) / 2.0) * T) / (sigma * np.sqrt(T))
        return float(norm.cdf(d1) - 1.0)
    except Exception:
        return None


def calculate_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> Optional[float]:
    try:
        if S is None or K is None or T is None or sigma is None:
            return None
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return None
        d1 = (np.log(S / K) + (r + (sigma**2) / 2.0) * T) / (sigma * np.sqrt(T))
        return float(norm.cdf(d1))
    except Exception:
        return None


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "contractSymbol",
        "strike",
        "bid",
        "ask",
        "impliedVolatility",
        "delta",
        "openInterest",
        "volume",
    ]
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan

    for col in ["strike", "bid", "ask", "impliedVolatility", "delta", "openInterest", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _fill_missing_delta(
    df: pd.DataFrame,
    option_type: str,
    spot_price: Optional[float],
    dte: int,
    risk_free_rate: float,
) -> pd.DataFrame:
    out = df.copy()
    if out.empty or spot_price is None or spot_price <= 0 or dte <= 0:
        return out

    T = dte / 365.0
    if T <= 0:
        return out

    missing_mask = out["delta"].isna()
    if not missing_mask.any():
        return out

    for idx in out.index[missing_mask]:
        K = out.at[idx, "strike"]
        sigma = out.at[idx, "impliedVolatility"]
        if pd.isna(K) or pd.isna(sigma):
            continue

        if option_type == "put":
            est = calculate_put_delta(float(spot_price), float(K), T, risk_free_rate, float(sigma))
        else:
            est = calculate_call_delta(float(spot_price), float(K), T, risk_free_rate, float(sigma))

        if est is not None:
            out.at[idx, "delta"] = est

    return out


def _liquidity_mask(df: pd.DataFrame) -> pd.Series:
    bid_ok = df["bid"].fillna(0).gt(0)

    has_oi = "openInterest" in df.columns and df["openInterest"].notna().any()
    has_vol = "volume" in df.columns and df["volume"].notna().any()

    if not has_oi and not has_vol:
        return bid_ok

    oi_ok = df["openInterest"].fillna(0).gt(0) if has_oi else pd.Series(False, index=df.index)
    vol_ok = df["volume"].fillna(0).gt(0) if has_vol else pd.Series(False, index=df.index)
    return bid_ok & (oi_ok | vol_ok)


def _delta_fit(abs_delta: pd.Series, target_delta: float) -> pd.Series:
    td = max(float(target_delta), 1e-9)
    fit = 1.0 - (abs_delta - td).abs() / td
    return fit.clip(lower=0, upper=1)


def _otm_fit(otm_pct: pd.Series, target_pct: float) -> pd.Series:
    t = max(float(target_pct), 1e-9)
    fit = 1.0 - (otm_pct - t).abs() / t
    return fit.clip(lower=0, upper=1)


def _above_cost_fit(above_cost_pct: pd.Series, target_pct: float) -> pd.Series:
    t = max(float(target_pct), 1e-9)
    fit = 1.0 - (above_cost_pct - t).abs() / t
    return fit.clip(lower=0, upper=1)


def _in_earnings_window(expiry_date, earnings_date, days_before: int, days_after: int) -> bool:
    if expiry_date is None or earnings_date is None:
        return False
    left = earnings_date - timedelta(days=int(days_before))
    right = earnings_date + timedelta(days=int(days_after))
    return left <= expiry_date <= right


def _prepare_put_candidates(
    df: pd.DataFrame,
    ticker: str,
    sector: str,
    expiry: str,
    dte: int,
    spot_price: Optional[float],
    target_delta: float,
    min_delta: float,
    max_delta: float,
    min_annualized: float,
    require_liquid: bool,
    risk_free_rate: float,
    put_otm_min_pct: float,
    put_otm_max_pct: float,
    put_otm_target_pct: float,
    earnings_flag: str,
    earnings_filter_mode: str,
    earnings_penalty: float,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if spot_price is None or spot_price <= 0:
        return pd.DataFrame()

    out = _ensure_columns(df)
    out = _fill_missing_delta(out, "put", spot_price, dte, risk_free_rate)

    out["ticker"] = ticker
    out["type"] = "Put"
    out["sector"] = sector or "Unknown"
    out["expiry"] = expiry
    out["dte"] = dte
    out["abs_delta"] = out["delta"].abs()
    out["annualized"] = np.where(
        (out["strike"] > 0) & (out["bid"] >= 0) & (dte > 0),
        (out["bid"] / out["strike"]) * (365.0 / float(dte)),
        np.nan,
    )
    out["liquidity_ok"] = _liquidity_mask(out)

    # Put OTM preference: require K < S and OTM% in range.
    out = out[out["strike"] < float(spot_price)].copy()
    if out.empty:
        return out

    out["otm_pct"] = (float(spot_price) - out["strike"]) / float(spot_price)

    mask = out["abs_delta"].between(min_delta, max_delta, inclusive="both")
    mask = mask & out["annualized"].ge(min_annualized)
    mask = mask & out["otm_pct"].between(float(put_otm_min_pct), float(put_otm_max_pct), inclusive="both")
    if require_liquid:
        mask = mask & out["liquidity_ok"]

    out = out[mask].copy()
    if out.empty:
        return out

    out["DeltaFit"] = _delta_fit(out["abs_delta"], target_delta)
    out["OTMFit"] = _otm_fit(out["otm_pct"], put_otm_target_pct)
    out["score"] = out["annualized"] * 0.55 + out["DeltaFit"] * 0.25 + out["OTMFit"] * 0.20

    out["above_cost_pct"] = np.nan
    out["earnings_flag"] = earnings_flag
    if earnings_filter_mode == "soft" and earnings_flag == "Risk":
        out["score"] = out["score"] * float(earnings_penalty)

    out = out.sort_values(by=["score", "annualized"], ascending=[False, False], na_position="last")
    return out.reset_index(drop=True)


def _prepare_call_candidates(
    df: pd.DataFrame,
    ticker: str,
    sector: str,
    expiry: str,
    dte: int,
    spot_price: Optional[float],
    cost_basis: Optional[float],
    target_delta: float,
    min_delta: float,
    max_delta: float,
    min_annualized: float,
    require_liquid: bool,
    risk_free_rate: float,
    call_min_above_cost_pct: float,
    call_target_above_cost_pct: float,
    earnings_flag: str,
    earnings_filter_mode: str,
    earnings_penalty: float,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if cost_basis is None or cost_basis <= 0:
        return pd.DataFrame()

    out = _ensure_columns(df)
    out = _fill_missing_delta(out, "call", spot_price, dte, risk_free_rate)

    out["ticker"] = ticker
    out["type"] = "Call"
    out["sector"] = sector or "Unknown"
    out["expiry"] = expiry
    out["dte"] = dte
    out["abs_delta"] = out["delta"].abs()
    out["annualized"] = np.where(
        (out["strike"] > 0) & (out["bid"] >= 0) & (dte > 0),
        (out["bid"] / out["strike"]) * (365.0 / float(dte)),
        np.nan,
    )
    out["liquidity_ok"] = _liquidity_mask(out)

    min_strike = float(cost_basis) * (1.0 + float(call_min_above_cost_pct))
    out = out[out["strike"] >= min_strike].copy()
    if out.empty:
        return out

    out["above_cost_pct"] = (out["strike"] - float(cost_basis)) / float(cost_basis)

    mask = out["abs_delta"].between(min_delta, max_delta, inclusive="both")
    mask = mask & out["annualized"].ge(min_annualized)
    if require_liquid:
        mask = mask & out["liquidity_ok"]

    out = out[mask].copy()
    if out.empty:
        return out

    out["DeltaFit"] = _delta_fit(out["abs_delta"], target_delta)
    out["AboveCostFit"] = _above_cost_fit(out["above_cost_pct"], call_target_above_cost_pct)
    out["score"] = out["annualized"] * 0.55 + out["DeltaFit"] * 0.25 + out["AboveCostFit"] * 0.20

    out["otm_pct"] = np.nan
    out["earnings_flag"] = earnings_flag
    if earnings_filter_mode == "soft" and earnings_flag == "Risk":
        out["score"] = out["score"] * float(earnings_penalty)

    out = out.sort_values(by=["score", "annualized"], ascending=[False, False], na_position="last")
    return out.reset_index(drop=True)


def _apply_sector_quota(sorted_df: pd.DataFrame, top_n: int, max_per_sector: int):
    if sorted_df is None or sorted_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    keep_rows = []
    skip_rows = []
    sector_count: Dict[str, int] = {}

    for _, row in sorted_df.iterrows():
        sector = row.get("sector", "Unknown")
        if pd.isna(sector) or not str(sector).strip():
            sector = "Unknown"
        sector = str(sector)

        cnt = sector_count.get(sector, 0)
        if cnt >= int(max_per_sector):
            skipped = row.copy()
            skipped["filtered_reason"] = f"sector_cap_exceeded({sector})"
            skip_rows.append(skipped)
            continue

        keep_rows.append(row)
        sector_count[sector] = cnt + 1

        if len(keep_rows) >= int(top_n):
            break

    keep_df = pd.DataFrame(keep_rows).reset_index(drop=True) if keep_rows else pd.DataFrame()
    skip_df = pd.DataFrame(skip_rows).reset_index(drop=True) if skip_rows else pd.DataFrame()
    return keep_df, skip_df


def build_recommendations(
    tickers: list[str],
    include_put: bool,
    include_call: bool,
    target_delta: float = 0.20,
    min_delta: float = 0.15,
    max_delta: float = 0.25,
    min_annualized: float = 0.15,
    min_dte: int = 14,
    max_dte: int = 60,
    require_liquid: bool = True,
    risk_free_rate: float = 0.04,
    per_ticker_top_k: int = 3,
    global_top_n: int = 10,
    put_otm_min_pct: float = 0.02,
    put_otm_max_pct: float = 0.12,
    put_otm_target_pct: float = 0.05,
    cost_basis_map: Optional[dict] = None,
    call_min_above_cost_pct: float = 0.02,
    call_target_above_cost_pct: float = 0.06,
    earnings_block_days_before: int = 7,
    earnings_block_days_after: int = 1,
    earnings_filter_mode: str = "strict",
    earnings_penalty: float = 0.6,
    max_per_sector: int = 2,
) -> Dict[str, object]:
    cost_basis_map = cost_basis_map or {}
    normalized_cost = {}
    for k, v in cost_basis_map.items():
        key = str(k).strip().upper()
        try:
            val = float(v)
        except Exception:
            val = np.nan
        normalized_cost[key] = val

    per_ticker: Dict[str, Dict[str, object]] = {}
    all_frames = []

    for raw in tickers:
        ticker = (raw or "").strip().upper()
        if not ticker:
            continue

        spot = get_spot_price(ticker)
        expiries = get_expiries(ticker)

        profile = get_ticker_profile(ticker)
        sector = profile.get("sector", "Unknown") or "Unknown"
        earnings_date = profile.get("earnings_date")
        earnings_status = profile.get("earnings_status", "unavailable")

        ticker_puts = []
        ticker_calls = []

        cost_basis = normalized_cost.get(ticker)
        if pd.isna(cost_basis):
            cost_basis = None

        for expiry in expiries:
            dte = compute_dte(expiry)
            if dte is None:
                continue
            if dte < int(min_dte) or dte > int(max_dte):
                continue

            expiry_date = parse_expiry_date(expiry)
            earnings_hit = _in_earnings_window(
                expiry_date=expiry_date,
                earnings_date=earnings_date,
                days_before=earnings_block_days_before,
                days_after=earnings_block_days_after,
            )

            if earnings_filter_mode == "strict" and earnings_hit:
                continue

            if earnings_status != "available":
                earnings_flag = "Unavailable"
            else:
                earnings_flag = "Risk" if earnings_hit else "OK"

            calls_df, puts_df = get_option_chain(ticker, expiry)

            if include_put:
                puts_candidates = _prepare_put_candidates(
                    df=puts_df,
                    ticker=ticker,
                    sector=sector,
                    expiry=expiry,
                    dte=dte,
                    spot_price=spot,
                    target_delta=target_delta,
                    min_delta=min_delta,
                    max_delta=max_delta,
                    min_annualized=min_annualized,
                    require_liquid=require_liquid,
                    risk_free_rate=risk_free_rate,
                    put_otm_min_pct=put_otm_min_pct,
                    put_otm_max_pct=put_otm_max_pct,
                    put_otm_target_pct=put_otm_target_pct,
                    earnings_flag=earnings_flag,
                    earnings_filter_mode=earnings_filter_mode,
                    earnings_penalty=earnings_penalty,
                )
                if not puts_candidates.empty:
                    ticker_puts.append(puts_candidates)

            if include_call:
                calls_candidates = _prepare_call_candidates(
                    df=calls_df,
                    ticker=ticker,
                    sector=sector,
                    expiry=expiry,
                    dte=dte,
                    spot_price=spot,
                    cost_basis=cost_basis,
                    target_delta=target_delta,
                    min_delta=min_delta,
                    max_delta=max_delta,
                    min_annualized=min_annualized,
                    require_liquid=require_liquid,
                    risk_free_rate=risk_free_rate,
                    call_min_above_cost_pct=call_min_above_cost_pct,
                    call_target_above_cost_pct=call_target_above_cost_pct,
                    earnings_flag=earnings_flag,
                    earnings_filter_mode=earnings_filter_mode,
                    earnings_penalty=earnings_penalty,
                )
                if not calls_candidates.empty:
                    ticker_calls.append(calls_candidates)

        puts_table = pd.concat(ticker_puts, ignore_index=True) if ticker_puts else pd.DataFrame()
        calls_table = pd.concat(ticker_calls, ignore_index=True) if ticker_calls else pd.DataFrame()

        if not puts_table.empty:
            puts_table = puts_table.sort_values(by=["score", "annualized"], ascending=[False, False]).reset_index(drop=True)
            all_frames.append(puts_table)
        if not calls_table.empty:
            calls_table = calls_table.sort_values(by=["score", "annualized"], ascending=[False, False]).reset_index(drop=True)
            all_frames.append(calls_table)

        per_ticker[ticker] = {
            "put": puts_table.head(per_ticker_top_k) if not puts_table.empty else pd.DataFrame(),
            "call": calls_table.head(per_ticker_top_k) if not calls_table.empty else pd.DataFrame(),
            "spot": spot,
            "sector": sector,
            "earnings_date": earnings_date,
            "earnings_status": earnings_status,
            "cost_basis": cost_basis,
        }

    global_all = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    if not global_all.empty:
        global_all = global_all.sort_values(by=["score", "annualized"], ascending=[False, False], na_position="last").reset_index(drop=True)

    global_top, global_skipped = _apply_sector_quota(global_all, global_top_n, max_per_sector)

    return {
        "per_ticker": per_ticker,
        "global_top": global_top,
        "global_all": global_all,
        "global_skipped": global_skipped,
    }
