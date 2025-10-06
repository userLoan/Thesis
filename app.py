import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================== mSSRM_PGA Algorithm ==================
def mSSRM_PGA(Param, matR, vecmu):
    """
    mSSRM-PGA Algorithm for portfolio optimization
    
    Parameters:
    - Param: dict with keys {eps, alpha, iternum, tol, m}
    - matR: T x N matrix of returns
    - vecmu: N x 1 vector of mean returns
    
    Returns:
    - w: N x 1 portfolio weights
    """
    T, N = matR.shape
    
    # Regularization
    eI = Param['eps'] * np.eye(N)
    p = vecmu
    
    # Compute Q matrix
    ones_T = np.ones((T, T))
    Q = (1 / np.sqrt(T - 1)) * (matR - (1 / T) * ones_T @ matR)
    QeI = Q.T @ Q + eI
    
    # Set alpha
    alpha = 0.999 / np.linalg.norm(QeI, 2)
    
    # Initialize
    w = vecmu.copy()
    k = 1
    RE = [100]  # Relative error
    
    while k <= Param['iternum'] and RE[-1] > Param['tol']:
        w1 = w.copy()
        
        # Proximal gradient step
        w_pre = w - alpha * (QeI @ w - p)
        
        # Project to non-negative orthant
        w_pre[w_pre < 0] = 0
        
        # Sparse projection: keep only top m assets
        sorted_idx = np.argsort(w_pre)[::-1]
        w = np.zeros(N)
        w[sorted_idx[:Param['m']]] = w_pre[sorted_idx[:Param['m']]]
        
        # Compute relative error
        if np.linalg.norm(w1, 2) > 0:
            RE.append(np.linalg.norm(w - w1, 2) / np.linalg.norm(w1, 2))
        else:
            RE.append(0)
        
        k += 1
    
    # Normalize weights
    if np.sum(w) == 0:
        w = np.zeros(N)
    else:
        w = w / np.sum(w)
    
    return w


def run_mSSRM_PGA(win_size, data, m, eps=1e-3, iternum=10000, tol=1e-5):
    """
    Run mSSRM-PGA portfolio optimization with rolling window
    
    Parameters:
    - win_size: window size for historical data
    - data: DataFrame of gross returns (1 + r)
    - m: sparsity level (number of assets to select)
    
    Returns:
    - CW: cumulative wealth
    - sharpe: Sharpe ratio
    - all_w: portfolio weights over time
    """
    Param = {
        'winsize': win_size,
        'm': m,
        'iternum': iternum,
        'tol': tol,
        'eps': eps
    }
    
    fullR = data.values - 1  # Convert to simple returns
    fullT, N = fullR.shape
    T_end = fullT
    
    all_w = np.ones((N, fullT)) / N
    CW = np.zeros(T_end)
    S = 1.0
    
    for t in range(T_end):
        if t > 5:
            # Define window
            if t <= Param['winsize']:
                win_start = 0
            else:
                win_start = t - Param['winsize']
            
            win_end = t - 1
            T = win_end - win_start + 1
            
            # Get historical returns
            matR = fullR[win_start:win_end+1, :]
            vecmu = np.mean(matR, axis=0)
            
            # Optimize portfolio
            w = mSSRM_PGA(Param, matR, vecmu)
            all_w[:, t] = w
            
            # Update cumulative wealth
            if np.sum(w) != 0:
                S = S * np.dot(data.values[t, :], w)
        
        CW[t] = S
    
    # Calculate Sharpe ratio
    returns = np.diff(CW) / CW[:-1]
    returns = returns[~np.isnan(returns)]
    
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0
    
    return CW, sharpe, all_w


# ================== Streamlit App ==================
st.set_page_config(page_title="mSSRM-PGA Portfolio Optimization", layout="wide")

st.title("📊 mSSRM-PGA Portfolio Optimization")
st.markdown("**Sparse Portfolio Selection using Proximal Gradient Algorithm**")

# Sidebar for parameters
st.sidebar.header("⚙️ Parameters")

# Stock selection
default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
tickers_input = st.sidebar.text_area(
    "Stock Tickers (comma-separated)",
    value=', '.join(default_tickers),
    help="Enter stock tickers separated by commas"
)
tickers = [t.strip().upper() for t in tickers_input.split(',')]

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=365*2),
        max_value=datetime.now()
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )

# Algorithm parameters
st.sidebar.subheader("Algorithm Parameters")
win_size = st.sidebar.slider("Window Size", 20, 252, 60, 10, help="Historical window for optimization")
m = st.sidebar.slider("Sparsity Level (m)", 1, len(tickers), min(5, len(tickers)), 1, 
                      help="Number of assets to select")
eps = st.sidebar.select_slider("Regularization (ε)", 
                               options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                               value=1e-3,
                               format_func=lambda x: f"{x:.0e}")

# Run button
run_button = st.sidebar.button("🚀 Run Optimization", type="primary")

# Main content
if run_button:
    with st.spinner("Downloading data from Yahoo Finance..."):
        try:
            # Download data
            data_dict = {}
            failed_tickers = []
            
            for ticker in tickers:
                try:
                    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not df.empty:
                        data_dict[ticker] = df['Adj Close']
                    else:
                        failed_tickers.append(ticker)
                except:
                    failed_tickers.append(ticker)
            
            if failed_tickers:
                st.warning(f"Failed to download: {', '.join(failed_tickers)}")
            
            if len(data_dict) == 0:
                st.error("No data downloaded. Please check tickers and try again.")
                st.stop()
            
            # Create price DataFrame
            prices = pd.DataFrame(data_dict)
            prices = prices.dropna()
            
            if len(prices) < 10:
                st.error("Insufficient data. Please select a longer date range.")
                st.stop()
            
            st.success(f"✅ Downloaded data for {len(prices.columns)} stocks with {len(prices)} trading days")
            
            # Convert to gross returns
            gross_returns = prices.pct_change().fillna(0) + 1
            gross_returns = gross_returns.iloc[1:]  # Remove first row
            
        except Exception as e:
            st.error(f"Error downloading data: {str(e)}")
            st.stop()
    
    with st.spinner("Running mSSRM-PGA optimization..."):
        try:
            # Run optimization
            CW, sharpe, all_w = run_mSSRM_PGA(win_size, gross_returns, m, eps=eps)
            
            # Display results
            st.header("📈 Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            final_wealth = CW[-1]
            total_return = (final_wealth - 1) * 100
            annualized_sharpe = sharpe * np.sqrt(252)  # Assuming daily data
            
            with col1:
                st.metric("Final Wealth", f"${final_wealth:.2f}")
            with col2:
                st.metric("Total Return", f"{total_return:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe:.4f}")
            with col4:
                st.metric("Annualized Sharpe", f"{annualized_sharpe:.4f}")
            
            # Plot cumulative wealth
            st.subheader("💰 Cumulative Wealth")
            fig_wealth = go.Figure()
            fig_wealth.add_trace(go.Scatter(
                x=prices.index[1:],
                y=CW,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2)
            ))
            fig_wealth.update_layout(
                xaxis_title="Date",
                yaxis_title="Cumulative Wealth",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_wealth, use_container_width=True)
            
            # Plot portfolio weights over time
            st.subheader("⚖️ Portfolio Weights Over Time")
            
            # Get average weights for display
            avg_weights = np.mean(all_w[:, 6:], axis=1)
            sorted_idx = np.argsort(avg_weights)[::-1]
            top_k = min(10, len(sorted_idx))
            
            fig_weights = go.Figure()
            for i in sorted_idx[:top_k]:
                fig_weights.add_trace(go.Scatter(
                    x=prices.index[1:],
                    y=all_w[i, 1:],
                    mode='lines',
                    name=prices.columns[i],
                    stackgroup='one'
                ))
            
            fig_weights.update_layout(
                xaxis_title="Date",
                yaxis_title="Weight",
                hovermode='x unified',
                height=400,
                yaxis=dict(tickformat='.0%')
            )
            st.plotly_chart(fig_weights, use_container_width=True)
            
            # Final portfolio composition
            st.subheader("🎯 Final Portfolio Composition")
            final_weights = all_w[:, -1]
            weight_df = pd.DataFrame({
                'Ticker': prices.columns,
                'Weight': final_weights
            }).sort_values('Weight', ascending=False)
            weight_df = weight_df[weight_df['Weight'] > 0]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(
                    weight_df.style.format({'Weight': '{:.2%}'}),
                    hide_index=True,
                    height=400
                )
            
            with col2:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=weight_df['Ticker'],
                    values=weight_df['Weight'],
                    hole=.3
                )])
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Download results
            st.subheader("💾 Download Results")
            
            results_df = pd.DataFrame({
                'Date': prices.index[1:],
                'Cumulative_Wealth': CW
            })
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"mSSRM_PGA_results_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error during optimization: {str(e)}")
            st.exception(e)

else:
    st.info("👈 Configure parameters in the sidebar and click 'Run Optimization' to start")
    
    # Show example
    st.markdown("---")
    st.subheader("📖 About mSSRM-PGA")
    st.markdown("""
    The **mSSRM-PGA** (modified Sparse Second-order Risk Minimization - Proximal Gradient Algorithm) 
    is a portfolio optimization method that:
    
    - **Minimizes risk** while targeting expected returns
    - **Enforces sparsity** by selecting only the top `m` assets
    - **Uses regularization** to improve stability
    - **Runs efficiently** with proximal gradient descent
    
    **Parameters:**
    - **Window Size**: Number of historical days used for optimization
    - **Sparsity Level (m)**: Maximum number of assets to hold
    - **Regularization (ε)**: Controls the strength of regularization (higher = more stable)
    
    **Algorithm Steps:**
    1. Download historical price data from Yahoo Finance
    2. Convert to returns (gross returns = 1 + simple return)
    3. For each day (after initial warmup):
       - Use a rolling window of past returns
       - Optimize portfolio weights using mSSRM-PGA
       - Rebalance portfolio and update cumulative wealth
    4. Calculate performance metrics (Sharpe ratio, total return, etc.)
    """)
