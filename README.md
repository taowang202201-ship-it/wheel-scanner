# Daily Multi-Ticker Options Recommender

一个基于 `Streamlit + yfinance` 的多股票期权推荐工具，支持 `Sell Put (CSP)` 与 `Sell Covered Call`，并按你的交易偏好进行评分与筛选。

## 核心能力

- 多股票 watchlist（textarea，每行一个 ticker）
- 一键保存配置到本地 `config.json`
- 双策略勾选：
  - `Sell Put (CSP)`
  - `Sell Covered Call`
- 拉取多到期日 option chain（受 DTE 范围约束）
- 计算：
  - `DTE`
  - `annualized = (bid / strike) * (365 / DTE)`
- Delta 缺失自动回填（Black-Scholes）：
  - Put: `N(d1)-1`
  - Call: `N(d1)`
  - `d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))`
  - `r=0.04`, `T=DTE/365`, `sigma=impliedVolatility`

## 新增偏好逻辑

### A) Put OTM 偏好

- 参数：
  - `put_otm_min_pct`（默认 `0.02`）
  - `put_otm_max_pct`（默认 `0.12`）
  - `put_otm_target_pct`（默认 `0.05`）
- 定义：`otm_pct = (S - K)/S`
- 规则：
  - Put 仅保留 `K < S`
  - `otm_pct` 必须在 `[min,max]`
- 评分：
  - `OTMFit = max(0, 1 - abs(otm_pct - put_otm_target_pct)/put_otm_target_pct)`
  - `Score_put = annualized*0.55 + DeltaFit*0.25 + OTMFit*0.20`

### B) Covered Call 成本价约束

- 参数：
  - `cost_basis_map`（可编辑表格 + CSV 上传）
  - `call_min_above_cost_pct`（默认 `0.02`）
  - `call_target_above_cost_pct`（默认 `0.06`）
- 规则：
  - 需要有效 `cost_basis`
  - 过滤：`K >= cost_basis*(1+call_min_above_cost_pct)`
  - `above_cost_pct = (K-cost_basis)/cost_basis`
- 评分：
  - `AboveCostFit = max(0, 1 - abs(above_cost_pct - call_target_above_cost_pct)/call_target_above_cost_pct)`
  - `Score_call = annualized*0.55 + DeltaFit*0.25 + AboveCostFit*0.20`

### C) 财报过滤（earnings week）

- 参数：
  - `earnings_block_days_before`（默认 `7`）
  - `earnings_block_days_after`（默认 `1`）
  - `earnings_filter_mode`：`strict/soft`（默认 `strict`）
- 数据源：`yfinance`（若缺失显示 `unavailable` 并跳过过滤，不崩溃）
- 逻辑：
  - `strict`：财报窗口内到期合约剔除
  - `soft`：不剔除，`score * 0.6`，并标记 `earnings_flag=Risk`

### D) 行业配额（组合风险控制）

- 参数：`max_per_sector`（默认 `2`）
- sector 来源：`ticker.info['sector']`，缺失归类 `Unknown`
- 全局 TopN 按 Score 从高到低选取，超行业上限则跳过
- 页面会显示被配额过滤的候选及原因（`FilteredReason`）

## 页面输出

- 每个 ticker：
  - `Best Put`
  - `Best Call`
  - 表格包含：`otm_pct`（put）、`above_cost_pct`（call）、`Sector`、`EarningsFlag`
- 全局 TopN：应用行业上限后的最终推荐
- VIX 顶部指标：当前值、较前日变化、状态提示
- VIX 趋势：
  - metric 保留 VIX 数字与日变化
  - 趋势区间切换：`30D / 90D / 180D`（默认 90D）
  - 可切换均线：`MA5`、`MA20`
  - 趋势摘要：`7D change %`、`30D change %`
  - `Refresh VIX` 按钮可强制刷新缓存
- 导出：
  - 保存到 `outputs/recommendations_YYYYMMDD.csv`
  - 支持下载 CSV
- 数据刷新：
  - 侧边栏 `Refresh Data` 按钮会执行 `st.cache_data.clear()`，强制重新拉取 yfinance 数据

## 文件结构

- `app.py`：UI 与参数输入
- `data.py`：yfinance 拉数 + 缓存 + ticker profile（sector/earnings）
- `analytics.py`：筛选/评分/财报逻辑/行业配额
- `config.py`：`config.json` 读写

## 运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

默认访问：
- `http://localhost:8501`

iPhone 同局域网访问：

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
