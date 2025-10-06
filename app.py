# app.py
# -------------------------------------------------------------
# mSSRM-PGA Portfolio Backtester (Python + Streamlit)
# -------------------------------------------------------------
# Features
# 1) Fetches price data directly from Yahoo Finance (yfinance)
# 2) Converts to gross returns (1 + r)
# 3) Runs m-SSRM with Projected Gradient Algorithm (PGA) over a rolling window
# 4) Updates cumulative wealth S_t = S_{t-1} * (data[t] · w_t)
# 5) Computes Sharpe on tick2ret(CW) as mean/std (NOT annualized)
# 6) Simple web UI using Streamlit (single-file app)
# 7) NEW: Preset menus for common groups (incl. Vietnam presets like VN30 ETF & VN-Index)
# -------------------------------------------------------------
# Quickstart
#   pip install streamlit yfinance pandas numpy matplotlib
#   streamlit run app.py
# -------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import date, timedelta

# -------------------------------
# Presets (Yahoo Finance based)
# -------------------------------
# NOTE: Some markets/tickers on Yahoo require suffixes (e.g., .HK, .L, .HM).
# Vietnam examples below are commonly used, but availability can vary.
# - E1VFVN30.HM: SSIAM VN30 ETF (tracks VN30)
# - ^VNINDEX: Vietnam VN-Index (broad market)
# If a symbol is unavailable in your region, you can edit/replace it in the input.

YF_PRESETS: dict[str, list[str]] = {
    "US Megacap Tech": ["AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL"],
    "US Broad ETFs": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "Europe Large (examples)": ["EZU", "VGK"],
    "Hong Kong / China Tech": ["0700.HK", "3690.HK", "BABA", "9888.HK"],
    # Vietnam
    "Vietnam - VN30 ETF": ["E1VFVN30.HM"],
    "Vietnam - Broad Index": ["^VNINDEX"],
    # You can extend with your own curated baskets
}

# -------------------------------
# Utils
# -------------------------------

def tick2ret(cw: pd.Series | np.ndarray) -> pd.Series:
    """Convert cumulative wealth to simple returns: r_t = CW_t / CW_{t-1} - 1."""
    if isinstance(cw, np.ndarray):
        cw = pd.Series(cw)
    return cw.pct_change().dropna()


def spectral_largest_eigval_sym(A: np.ndarray) -> float:
    """Largest eigenvalue for symmetric PSD matrix.
    Use eigh for better numerical stability than SVD here.
    """
    vals = np.linalg.eigvalsh(A)
    return float(vals.max())


# -------------------------------
# Core: mSSRM-PGA in NumPy
# -------------------------------

def mssrm_pga(matR: np.ndarray,
              vecmu: np.ndarray,
              m: int,
              eps: float = 1e-3,
              iternum: int = 10_000,
              tol: float = 1e-5,
              alpha: float | None = None) -> tuple[np.ndarray, list[float]]:
    """Projected Gradient Algorithm for m-SSRM.

    Problem (nonnegativity + sparsity ≤ m):
        minimize  0.5 * w^T (Q^T Q + eps I) w  - vecmu^T w
        s.t.      w >= 0,  ||w||_0 <= m

    Where Q = (matR - mean(matR, axis=0)) / sqrt(T-1)

    Returns (w, RE_history)
    """
    T, N = matR.shape
    m_eff = min(int(m), N)

    # Demean & scale by sqrt(T-1) (guard for T=1)
    denom = np.sqrt(max(T - 1, 1))
    Q = (matR - matR.mean(axis=0, keepdims=True)) / denom

    # (Q'Q + eps I)
    QeI = Q.T @ Q + eps * np.eye(N)

    # Stepsize
    if alpha is None:
        L = spectral_largest_eigval_sym(QeI)  # Lipschitz constant of gradient
        # Safe step
        alpha = 0.999 / max(L, 1e-12)

    # Init w: nonnegative from vecmu, then normalize if positive sum
    w = np.maximum(vecmu, 0.0).astype(float)
    s = w.sum()
    if s > 0:
        w /= s
    else:
        w[:] = 0.0

    RE_hist: list[float] = []

    for _ in range(iternum):
        w_old = w.copy()
        # Gradient: (Q'Q + eps I) w - mu
        grad = QeI @ w - vecmu
        w_pre = w - alpha * grad
        # Project to nonnegative orthant
        np.maximum(w_pre, 0.0, out=w_pre)
        # Sparsity projection: keep top-m entries by value (nonnegative already)
        if m_eff < N:
            # argpartition for O(N)
            keep_idx = np.argpartition(w_pre, -m_eff)[-m_eff:]
            w = np.zeros_like(w_pre)
            w[keep_idx] = w_pre[keep_idx]
        else:
            w = w_pre
        # Relative error
        denom = max(np.linalg.norm(w_old, 2), 1e-12)
        RE = float(np.linalg.norm(w - w_old, 2) / denom)
        RE_hist.append(RE)
        if RE <= tol:
            break

    # Normalize to sum 1 if positive mass
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        w[:] = 0.0

    return w, RE_hist


# -------------------------------
# Data: Yahoo Finance → gross returns (1 + r)
# -------------------------------

@st.cache_data(show_spinner=False)
def fetch_gross_returns(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Fetch Adjusted Close from Yahoo, compute daily gross returns (1+r).
    Uses auto_adjust=True so 'Close' is adjusted; returns a DataFrame indexed by date
    with columns=tickers, entries are gross returns (1+r).
    """
    tick_list = [t.strip() for t in tickers if t and t.strip()]
    tick_str = " ".join([t.upper() for t in tick_list])
    if not tick_str:
        raise ValueError("Ticker list is empty.")

    raw = yf.download(
        tick_str,
        start=start,
        end=end + timedelta(days=1),  # include full end-date
        auto_adjust=True,
        progress=False,
        group_by='column'
    )

    # Handle single vs multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw['Close'].copy()
        # Keep only requested columns if Yahoo returned a superset
        cols_to_keep = [c for c in px.columns if c.upper() in {s.upper() for s in tick_list}]
        px = px[cols_to_keep]
    else:
        # Single ticker case
        px = raw[['Close']].copy()
        # When Yahoo normalizes names, use the original requested name
        px.columns = [tick_list[0]]

    # Drop columns that are completely NaN (unavailable tickers), warn in UI later
    px = px.dropna(axis=1, how='all')

    # If nothing left, raise
    if px.shape[1] == 0:
        raise ValueError("None of the provided tickers returned data. Please check symbols/suffixes.")

    # Forward/backward fill gaps then compute returns
    px = px.ffill().bfill()
    rets = px.pct_change().dropna(how='any')
    gross = 1.0 + rets

    # Remove infs/nans if any
    gross = gross.replace([np.inf, -np.inf], np.nan).dropna(how='any')

    return gross


# -------------------------------
# Rolling backtest runner
# -------------------------------

def run_backtest(
    gross: pd.DataFrame,
    winsize: int,
    m: int,
    eps: float = 1e-3,
    iternum: int = 10_000,
    tol: float = 1e-5,
    warmup: int = 5,
):
    """Run rolling-window mSSRM-PGA strategy.

    gross: DataFrame of gross returns (1+r), shape T x N
    Returns: cw (Series), weights (DataFrame T x N), sharpe (float)
    """
    if winsize <= 0:
        raise ValueError("winsize must be positive")
    tickers = list(gross.columns)
    T, N = gross.shape

    fullR = gross - 1.0  # matrix of simple returns

    cw = pd.Series(index=gross.index, dtype='float64')  # cumulative wealth
    all_w = pd.DataFrame(0.0, index=gross.index, columns=tickers)

    S = 1.0

    for t in range(T):
        if t >= warmup:
            win_start = max(0, t - winsize)
            win_end = t - 1
            if win_end >= win_start:
                matR = fullR.iloc[win_start:win_end + 1, :].to_numpy()
                vecmu = matR.mean(axis=0)
                w_t, _ = mssrm_pga(matR, vecmu, m=m, eps=eps, iternum=iternum, tol=tol)
                all_w.iloc[t, :] = w_t
                if w_t.sum() > 0:
                    # Update S with today's gross returns * weights
                    S *= float(np.dot(gross.iloc[t, :].to_numpy(), w_t))
        cw.iloc[t] = S

    # Compute Sharpe from tick2ret(cw)
    strat_rets = tick2ret(cw)
    sharpe = float(strat_rets.mean() / strat_rets.std()) if strat_rets.std() > 0 else np.nan

    return cw, all_w, sharpe


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(
    page_title="mSSRM-PGA Backtester (Yahoo Finance)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("mSSRM-PGA Backtester (Yahoo Finance)")
st.caption(
    "This demo fetches data from Yahoo Finance, transforms to gross returns (1+r), runs mSSRM with PGA, and reports non-annualized Sharpe."
)

with st.sidebar:
    st.header("Parameters")

    # Tickers input with session state so we can modify from presets
    if "tickers_text" not in st.session_state:
        st.session_state.tickers_text = "AAPL, MSFT, AMZN, NVDA, META, GOOGL"

    # Preset chooser
    st.subheader("Presets (Yahoo)")
    preset_name = st.selectbox("Choose a preset (optional)", options=["<None>"] + list(YF_PRESETS.keys()), index=0)
    if st.button("Add preset to tickers") and preset_name != "<None>":
        current = [t.strip() for t in st.session_state.tickers_text.split(',') if t.strip()]
        merged = list(dict.fromkeys(current + YF_PRESETS[preset_name]))  # unique, keep order
        st.session_state.tickers_text = ", ".join(merged)

    tickers_text = st.text_input("Tickers (comma-separated)", key="tickers_text")
    tickers = [t.strip() for t in tickers_text.split(',') if t.strip()]

    st.caption("Tip: Vietnam examples — VN30 ETF: E1VFVN30.HM; Broad index: ^VNINDEX. If a symbol fails, edit suffix.")

    today = date.today()
    start_date = st.date_input("Start date", value=today.replace(year=max(2005, today.year - 7)))
    end_date = st.date_input("End date", value=today)

    winsize = st.number_input("Rolling window (days)", min_value=5, max_value=2000, value=60, step=5)
    m = st.number_input("Sparsity m (number of assets)", min_value=1, max_value=200, value=5, step=1)
    warmup = st.number_input("Warmup periods (no trading)", min_value=1, max_value=100, value=5, step=1)

    eps = st.number_input("Ridge epsilon", min_value=1e-9, value=1e-3, format="%e")
    tol = st.number_input("Tolerance (RE)", min_value=1e-9, value=1e-5, format="%e")
    iternum = st.number_input("Max iterations", min_value=100, max_value=200000, value=10000, step=100)

    run_btn = st.button("Run Backtest", type="primary")

# Main content
if run_btn:
    try:
        with st.status("Fetching data from Yahoo Finance...", expanded=False):
            gross = fetch_gross_returns(tickers, start_date, end_date)
            # Warn about missing tickers
            returned_cols = set(map(str.upper, gross.columns))
            requested_cols = set(map(str.upper, tickers))
            missing = [t for t in tickers if t.upper() not in returned_cols]
            if missing:
                st.warning(f"No data returned for: {', '.join(missing)}. Please verify symbols/suffixes.")

        if m > len(gross.columns):
            st.warning(f"m ({m}) > number of available series ({len(gross.columns)}). Using m={len(gross.columns)} instead.")
            m_eff = len(gross.columns)
        else:
            m_eff = int(m)

        with st.status("Running mSSRM-PGA rolling backtest...", expanded=False):
            cw, weights, sharpe = run_backtest(
                gross, winsize=winsize, m=m_eff, eps=float(eps), iternum=int(iternum), tol=float(tol), warmup=int(warmup)
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Non-annualized Sharpe", f"{sharpe:.4f}" if not np.isnan(sharpe) else "NaN")
        with col2:
            st.metric("Final wealth (CW end)", f"{cw.iloc[-1]:.4f}")
        with col3:
            st.metric("Trading days", f"{len(cw):d}")

        st.subheader("Cumulative Wealth")
        st.line_chart(cw, height=300)

        st.subheader("Weights over time (stacked area)")
        # Only plot from first non-zero weights row
        first_idx = weights.replace(0.0, np.nan).dropna(how='all').index.min()
        w_plot = weights.copy()
        if first_idx is not None:
            w_plot = w_plot.loc[first_idx:]
        st.area_chart(w_plot, height=360)

        st.subheader("Sample of weights (last 10 rows)")
        st.dataframe(weights.tail(10).style.format("{:.4f}"))

        # Downloads
        cw_csv = cw.to_frame(name='CW').to_csv().encode('utf-8')
        st.download_button("Download CW (CSV)", data=cw_csv, file_name="cw.csv", mime="text/csv")

        weights_csv = weights.to_csv().encode('utf-8')
        st.download_button("Download Weights (CSV)", data=weights_csv, file_name="weights.csv", mime="text/csv")

        st.caption("Note: This is a research demo, not investment advice.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    st.info("Choose a preset or enter tickers, set parameters, then click **Run Backtest**.")
