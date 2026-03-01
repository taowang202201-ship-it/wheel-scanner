from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from analytics import build_recommendations
from config import load_config, save_config
from data import get_vix_history, get_vix_snapshot


st.set_page_config(page_title="Daily Multi-Ticker Options Recommender", layout="wide")
st.title("Daily Multi-Ticker Options Recommender")


def parse_watchlist(raw: str) -> list[str]:
    symbols = []
    for line in (raw or "").splitlines():
        chunk = line.replace("，", ",")
        parts = [p.strip().upper() for p in chunk.split(",")]
        for p in parts:
            if p and p not in symbols:
                symbols.append(p)
    return symbols


def normalize_cost_basis_map(raw: dict) -> dict:
    out = {}
    for k, v in (raw or {}).items():
        ticker = str(k).strip().upper()
        if not ticker:
            continue
        try:
            value = float(v)
        except Exception:
            continue
        if value > 0:
            out[ticker] = value
    return out


def build_cost_basis_rows(watchlist: list[str], cfg_cost_map: dict) -> pd.DataFrame:
    rows = []
    seen = set()
    for ticker in watchlist:
        t = ticker.strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        rows.append({"ticker": t, "cost_basis": cfg_cost_map.get(t)})

    for ticker, basis in cfg_cost_map.items():
        t = str(ticker).strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        rows.append({"ticker": t, "cost_basis": basis})

    return pd.DataFrame(rows, columns=["ticker", "cost_basis"])


def cost_basis_from_editor(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    out = {}
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        value = row.get("cost_basis")
        try:
            value = float(value)
        except Exception:
            continue
        if value > 0:
            out[ticker] = value
    return out


def cost_basis_from_upload(uploaded_file) -> dict:
    if uploaded_file is None:
        return {}
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        return {}

    if df.empty:
        return {}

    cols_lower = {c.lower().strip(): c for c in df.columns}
    ticker_col = cols_lower.get("ticker") or cols_lower.get("symbol")
    cost_col = cols_lower.get("cost_basis") or cols_lower.get("cost")

    if ticker_col is None or cost_col is None:
        if len(df.columns) >= 2:
            ticker_col = df.columns[0]
            cost_col = df.columns[1]
        else:
            return {}

    out = {}
    for _, row in df.iterrows():
        ticker = str(row.get(ticker_col, "")).strip().upper()
        if not ticker:
            continue
        try:
            value = float(row.get(cost_col))
        except Exception:
            continue
        if value > 0:
            out[ticker] = value
    return out


def format_reco_table(df: pd.DataFrame, include_reason: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    keep_cols = [
        "ticker",
        "type",
        "expiry",
        "dte",
        "strike",
        "bid",
        "ask",
        "impliedVolatility",
        "delta",
        "abs_delta",
        "annualized",
        "otm_pct",
        "above_cost_pct",
        "sector",
        "earnings_flag",
        "score",
        "openInterest",
        "volume",
        "contractSymbol",
    ]
    if include_reason:
        keep_cols.append("filtered_reason")

    existing = [c for c in keep_cols if c in df.columns]
    out = df[existing].copy()
    out = out.rename(
        columns={
            "ticker": "Ticker",
            "type": "Type",
            "expiry": "Expiry",
            "dte": "DTE",
            "strike": "Strike",
            "bid": "Bid",
            "ask": "Ask",
            "impliedVolatility": "IV",
            "delta": "Delta",
            "abs_delta": "|Delta|",
            "annualized": "Annualized",
            "otm_pct": "otm_pct",
            "above_cost_pct": "above_cost_pct",
            "sector": "Sector",
            "earnings_flag": "EarningsFlag",
            "score": "Score",
            "openInterest": "OpenInterest",
            "volume": "Volume",
            "contractSymbol": "Contract",
            "filtered_reason": "FilteredReason",
        }
    )
    return out


def trailing_change_pct(series: pd.Series, days: int):
    need_points = int(days) + 1
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < need_points:
        return None
    latest = float(clean.iloc[-1])
    base = float(clean.iloc[-need_points])
    if base == 0:
        return None
    return (latest / base) - 1.0


cfg = load_config()
default_watchlist_text = "\n".join(cfg.get("watchlist", ["NVDA", "SOFI", "TSLA"]))
default_cost_map = normalize_cost_basis_map(cfg.get("cost_basis_map", {}))

with st.sidebar:
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        if "vix_refresh_token" in st.session_state:
            st.session_state["vix_refresh_token"] = 0
        st.rerun()

    st.header("Watchlist")
    watchlist_text = st.text_area(
        "每行一个 ticker（支持逗号分隔）",
        value=default_watchlist_text,
        height=120,
    )
    watchlist = parse_watchlist(watchlist_text)

    st.header("策略")
    include_put = st.checkbox("Sell Put (CSP)", value=bool(cfg.get("include_put", True)))
    include_call = st.checkbox("Sell Covered Call", value=bool(cfg.get("include_call", True)))

    st.header("Put OTM 偏好")
    put_otm_min_pct = st.number_input("put_otm_min_pct", min_value=0.0, max_value=0.99, value=float(cfg.get("put_otm_min_pct", 0.02)), step=0.01, format="%.2f")
    put_otm_max_pct = st.number_input("put_otm_max_pct", min_value=0.0, max_value=1.50, value=float(cfg.get("put_otm_max_pct", 0.12)), step=0.01, format="%.2f")
    put_otm_target_pct = st.number_input("put_otm_target_pct", min_value=0.001, max_value=1.50, value=float(cfg.get("put_otm_target_pct", 0.05)), step=0.01, format="%.2f")

    st.header("Covered Call 成本价")
    cost_editor_df = build_cost_basis_rows(watchlist, default_cost_map)
    edited_cost_df = st.data_editor(
        cost_editor_df,
        use_container_width=True,
        num_rows="dynamic",
        key="cost_basis_editor",
        column_config={
            "ticker": st.column_config.TextColumn("ticker"),
            "cost_basis": st.column_config.NumberColumn("cost_basis", min_value=0.0, step=0.01, format="%.4f"),
        },
    )
    uploaded_cost_csv = st.file_uploader("上传成本价 CSV（ticker,cost_basis）", type=["csv"], accept_multiple_files=False)
    call_min_above_cost_pct = st.number_input("call_min_above_cost_pct", min_value=0.0, max_value=1.0, value=float(cfg.get("call_min_above_cost_pct", 0.02)), step=0.01, format="%.2f")
    call_target_above_cost_pct = st.number_input("call_target_above_cost_pct", min_value=0.001, max_value=1.0, value=float(cfg.get("call_target_above_cost_pct", 0.06)), step=0.01, format="%.2f")

    st.header("筛选参数")
    target_delta = st.number_input("target_delta", min_value=0.01, max_value=0.99, value=float(cfg.get("target_delta", 0.20)), step=0.01)
    delta_range = st.slider(
        "delta 区间（按 abs(delta)）",
        min_value=0.01,
        max_value=0.99,
        value=(float(cfg.get("min_delta", 0.15)), float(cfg.get("max_delta", 0.25))),
        step=0.01,
    )
    min_annualized = st.number_input(
        "最低年化收益率",
        min_value=0.0,
        max_value=10.0,
        value=float(cfg.get("min_annualized", 0.15)),
        step=0.01,
        format="%.2f",
    )
    dte_range = st.slider(
        "DTE 范围",
        min_value=1,
        max_value=365,
        value=(int(cfg.get("min_dte", 14)), int(cfg.get("max_dte", 60))),
        step=1,
    )
    require_liquid = st.checkbox("流动性过滤（bid>0 且 OI>0 或 volume>0）", value=bool(cfg.get("require_liquid", True)))

    st.header("财报过滤")
    earnings_block_days_before = st.number_input("earnings_block_days_before", min_value=0, max_value=30, value=int(cfg.get("earnings_block_days_before", 7)), step=1)
    earnings_block_days_after = st.number_input("earnings_block_days_after", min_value=0, max_value=30, value=int(cfg.get("earnings_block_days_after", 1)), step=1)
    earnings_filter_mode = st.selectbox(
        "earnings_filter_mode",
        options=["strict", "soft"],
        index=0 if str(cfg.get("earnings_filter_mode", "strict")) == "strict" else 1,
    )

    st.header("组合风险控制")
    max_per_sector = st.number_input("max_per_sector", min_value=1, max_value=20, value=int(cfg.get("max_per_sector", 2)), step=1)

    st.header("输出")
    per_ticker_top_k = st.number_input("每个 ticker TopK", min_value=1, max_value=10, value=int(cfg.get("per_ticker_top_k", 3)), step=1)
    global_top_n = st.number_input("全局 TopN", min_value=1, max_value=200, value=int(cfg.get("global_top_n", 10)), step=1)

    cost_map_editor = cost_basis_from_editor(edited_cost_df)
    cost_map_upload = cost_basis_from_upload(uploaded_cost_csv)
    cost_basis_map = {**cost_map_editor, **cost_map_upload}

    if st.button("保存当前配置到 config.json", use_container_width=True):
        save_payload = {
            "watchlist": watchlist or ["NVDA", "SOFI", "TSLA"],
            "include_put": include_put,
            "include_call": include_call,
            "target_delta": float(target_delta),
            "min_delta": float(delta_range[0]),
            "max_delta": float(delta_range[1]),
            "min_annualized": float(min_annualized),
            "min_dte": int(dte_range[0]),
            "max_dte": int(dte_range[1]),
            "require_liquid": require_liquid,
            "global_top_n": int(global_top_n),
            "per_ticker_top_k": int(per_ticker_top_k),
            "put_otm_min_pct": float(put_otm_min_pct),
            "put_otm_max_pct": float(put_otm_max_pct),
            "put_otm_target_pct": float(put_otm_target_pct),
            "cost_basis_map": cost_basis_map,
            "call_min_above_cost_pct": float(call_min_above_cost_pct),
            "call_target_above_cost_pct": float(call_target_above_cost_pct),
            "earnings_block_days_before": int(earnings_block_days_before),
            "earnings_block_days_after": int(earnings_block_days_after),
            "earnings_filter_mode": str(earnings_filter_mode),
            "max_per_sector": int(max_per_sector),
        }
        save_config(save_payload)
        st.success("已保存到 config.json")

if not watchlist:
    st.warning("watchlist 为空，请至少输入一个 ticker。")
    st.stop()

if not include_put and not include_call:
    st.warning("请至少勾选一个策略（Sell Put 或 Sell Covered Call）。")
    st.stop()

if "vix_refresh_token" not in st.session_state:
    st.session_state["vix_refresh_token"] = 0

vix = get_vix_snapshot(cache_buster=int(st.session_state["vix_refresh_token"]))
vix_hist = get_vix_history(cache_buster=int(st.session_state["vix_refresh_token"]))
col1, col2, col3 = st.columns([1.2, 1.2, 2.0])
with col1:
    if vix.get("price") is None:
        st.metric("VIX", "—", delta="—")
    else:
        delta_text = "—"
        if vix.get("change") is not None:
            delta_text = f"{vix['change']:+.2f}"
        st.metric("VIX", f"{vix['price']:.2f}", delta=delta_text)
with col2:
    st.metric("更新时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
with col3:
    st.info(f"VIX 状态：{vix.get('status', 'N/A')}")

st.markdown("#### VIX 趋势")
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.2, 1, 1, 1.2])
with ctrl1:
    trend_window = st.selectbox("区间", options=["30D", "90D", "180D"], index=1)
with ctrl2:
    show_ma5 = st.checkbox("MA5", value=True)
with ctrl3:
    show_ma20 = st.checkbox("MA20", value=True)
with ctrl4:
    if st.button("Refresh VIX", use_container_width=True):
        st.session_state["vix_refresh_token"] = int(st.session_state["vix_refresh_token"]) + 1
        st.rerun()

if vix_hist is None or vix_hist.empty:
    st.caption("VIX 趋势数据不可用。")
else:
    window_map = {"30D": 30, "90D": 90, "180D": 180}
    n = window_map.get(trend_window, 90)
    vix_plot = vix_hist.tail(n).copy()
    vix_plot = vix_plot.rename(columns={"Close": "close"})
    vix_plot["ma5"] = vix_plot["close"].rolling(5).mean()
    vix_plot["ma20"] = vix_plot["close"].rolling(20).mean()

    plot_cols = ["close"]
    if show_ma5:
        plot_cols.append("ma5")
    if show_ma20:
        plot_cols.append("ma20")

    st.line_chart(vix_plot[plot_cols], use_container_width=True)

    chg7 = trailing_change_pct(vix_hist["Close"], 7)
    chg30 = trailing_change_pct(vix_hist["Close"], 30)
    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"7D change %: {'—' if chg7 is None else f'{chg7:+.2%}'}")
    with c2:
        st.caption(f"30D change %: {'—' if chg30 is None else f'{chg30:+.2%}'}")

with st.spinner("正在拉取多标的数据并计算推荐..."):
    rec = build_recommendations(
        tickers=watchlist,
        include_put=include_put,
        include_call=include_call,
        target_delta=float(target_delta),
        min_delta=float(delta_range[0]),
        max_delta=float(delta_range[1]),
        min_annualized=float(min_annualized),
        min_dte=int(dte_range[0]),
        max_dte=int(dte_range[1]),
        require_liquid=bool(require_liquid),
        risk_free_rate=0.04,
        per_ticker_top_k=int(per_ticker_top_k),
        global_top_n=int(global_top_n),
        put_otm_min_pct=float(put_otm_min_pct),
        put_otm_max_pct=float(put_otm_max_pct),
        put_otm_target_pct=float(put_otm_target_pct),
        cost_basis_map=cost_basis_map,
        call_min_above_cost_pct=float(call_min_above_cost_pct),
        call_target_above_cost_pct=float(call_target_above_cost_pct),
        earnings_block_days_before=int(earnings_block_days_before),
        earnings_block_days_after=int(earnings_block_days_after),
        earnings_filter_mode=str(earnings_filter_mode),
        max_per_sector=int(max_per_sector),
    )

per_ticker = rec.get("per_ticker", {})
global_top = rec.get("global_top", pd.DataFrame())
global_all = rec.get("global_all", pd.DataFrame())
global_skipped = rec.get("global_skipped", pd.DataFrame())

st.subheader("按 Ticker 推荐")
for ticker in watchlist:
    entry = per_ticker.get(ticker, {})
    spot = entry.get("spot")
    sector = entry.get("sector", "Unknown")
    earnings_date = entry.get("earnings_date")
    earnings_status = entry.get("earnings_status", "unavailable")
    cost_basis = entry.get("cost_basis")

    title = f"{ticker} | Sector: {sector}"
    if spot is not None:
        title += f" | 现价: {spot:.2f}"
    st.markdown(f"### {title}")

    if earnings_status == "available":
        st.caption(f"财报日期：{earnings_date} | 过滤模式：{earnings_filter_mode}")
    else:
        st.caption("财报日期：unavailable（已跳过财报过滤）")

    if include_put:
        st.markdown("**Best Put（Top 1-3）**")
        put_df = format_reco_table(entry.get("put", pd.DataFrame()))
        if put_df.empty:
            st.write("无符合条件的 Put")
        else:
            st.dataframe(put_df, use_container_width=True, hide_index=True)

    if include_call:
        st.markdown("**Best Call（Top 1-3）**")
        if cost_basis is None:
            st.write("未提供有效 cost_basis，Call 推荐已跳过。")
        call_df = format_reco_table(entry.get("call", pd.DataFrame()))
        if call_df.empty:
            st.write("无符合条件的 Call")
        else:
            st.dataframe(call_df, use_container_width=True, hide_index=True)

st.subheader(f"全局 Top {int(global_top_n)} 推荐（行业上限 max_per_sector={int(max_per_sector)}）")
show_global = format_reco_table(global_top)
if show_global.empty:
    st.info("当前筛选条件下没有全局推荐结果。")
else:
    st.dataframe(show_global, use_container_width=True, hide_index=True)

if global_skipped is not None and not global_skipped.empty:
    st.markdown("**被行业配额过滤的候选（原因）**")
    st.dataframe(format_reco_table(global_skipped, include_reason=True), use_container_width=True, hide_index=True)

st.caption(
    "Put评分：annualized*0.55 + DeltaFit*0.25 + OTMFit*0.20；"
    "Call评分：annualized*0.55 + DeltaFit*0.25 + AboveCostFit*0.20；"
    "soft财报模式下 Risk 合约会乘以 penalty=0.6"
)

st.subheader("导出")
if global_top is None or global_top.empty:
    st.write("当前无可导出的推荐数据。")
else:
    date_tag = datetime.now().strftime("%Y%m%d")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"recommendations_{date_tag}.csv"

    export_df = format_reco_table(global_top)

    if st.button("导出当日推荐 CSV（保存到 outputs/）"):
        export_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        st.success(f"已导出：{output_path}")

    csv_bytes = export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="下载 CSV",
        data=csv_bytes,
        file_name=output_path.name,
        mime="text/csv",
    )

if global_all is not None and not global_all.empty:
    st.caption(f"总候选数：{len(global_all)}，全局入选：{len(global_top)}")
