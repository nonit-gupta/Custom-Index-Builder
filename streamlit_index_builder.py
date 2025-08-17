# streamlit_index_builder.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime as dt

st.set_page_config(layout="wide", page_title="Simple Market-Cap / Equal / Custom Index Builder")

# --- Original helper functions (kept logic unchanged) ---
def risk_adjusted_ratios(close_prices, risk_free_rate=0.0, trading_days=252):
    close_prices = pd.Series(close_prices).dropna()
    returns = close_prices.pct_change().dropna()
    daily_rf = risk_free_rate / trading_days
    excess_returns = returns - daily_rf
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    sharpe_ratio = (mean_return / std_return) * np.sqrt(trading_days) if std_return != 0 else np.nan
    sortino_ratio = (mean_return / downside_std) * np.sqrt(trading_days) if downside_std != 0 else np.nan
    return sharpe_ratio, sortino_ratio

def max_drawdown(price_series):
    prices = pd.Series(price_series).dropna()
    cumulative_max = prices.cummax()
    drawdown = (prices / cumulative_max) - 1
    max_dd = drawdown.min()
    return max_dd*100

# --- Page inputs (moved from sidebar to main page) ---
st.title("Index Builder — Inputs")

col_left, col_right = st.columns([1,1])

with col_left:
    default_tickers = "TCS.NS,INFY.NS,RELIANCE.NS,BAJAJ-AUTO.NS"
    tickers_input = st.text_input("Tickers (comma-separated)", value=default_tickers)
    tickers_list = [t.strip() for t in tickers_input.split(",") if t.strip()]

    weighting_choice = st.selectbox("Weighting technique", options=["Market Cap", "Equal", "Custom"], index=0)

    risk_free_rate_pct = st.number_input("Risk-free rate (%)", value=0.0, format="%.4f")
    risk_free_rate = float(risk_free_rate_pct) / 100.0

    # Custom weights input field (only show when Custom selected)
    custom_weights_text = ""
    if weighting_choice == "Custom":
        custom_weights_text = st.text_input(
            "Custom weights (comma-separated). Will be normalized if they don't sum to 1.",
            value=",".join(["{:.4f}".format(1/len(tickers_list)) for _ in range(len(tickers_list))])
        )

with col_right:
    use_today_end = st.checkbox("Use current date as end date", value=True)
    today = dt.date.today()
    default_end = today if use_today_end else today - dt.timedelta(days=1)
    end_date = st.date_input("End date", value=default_end)

    st.markdown("Start date presets (tick boxes). If multiple selected, the largest period will be used.")
    preset_1y = st.checkbox("1 year ago", value=False)
    preset_2y = st.checkbox("2 years ago", value=False)
    preset_3y = st.checkbox("3 years ago", value=False)
    preset_5y = st.checkbox("5 years ago", value=False)
    preset_ytd = st.checkbox("Year-to-date (YTD)", value=False)

    # Choose start date based on presets:
    selected_years = []
    if preset_1y: selected_years.append(1)
    if preset_2y: selected_years.append(2)
    if preset_3y: selected_years.append(3)
    if preset_5y: selected_years.append(5)

    if selected_years:
        chosen_years = max(selected_years)
        start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=chosen_years)).date()
        st.markdown(f"Start date set to {chosen_years} year(s) before end date: **{start_date}**")
    elif preset_ytd:
        # YTD: start of the end_date's year
        start_date = dt.date(end_date.year, 1, 1)
        st.markdown(f"Start date set to Year-to-date (YTD): **{start_date}**")
    else:
        start_date = st.date_input("Start date", value=(pd.to_datetime(end_date) - pd.DateOffset(years=1)).date())

st.markdown("---")

# Build button (main page)
if st.button("Build index"):
    st.session_state['build_index'] = True

if 'build_index' not in st.session_state:
    st.session_state['build_index'] = False

# --- CACHED helpers (return only picklable types) ---
@st.cache_data(ttl=600)
def get_history_df(ticker_symbol, start, end):
    """Return DataFrame of historical prices for ticker_symbol between start and end."""
    t = yf.Ticker(ticker_symbol)
    df = t.history(start=start, end=end)
    df = df.reset_index()
    if 'Stock Splits' in df.columns:
        try:
            df = df.drop(columns=['Stock Splits'])
        except Exception:
            pass
    # ensure Date column exists and is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data(ttl=600)
def get_market_cap_value(ticker_symbol):
    """Return the current market cap (float) for the ticker, or np.nan if unavailable."""
    try:
        t = yf.Ticker(ticker_symbol)
        mc = t.fast_info.market_cap
        return float(mc) if mc is not None else float('nan')
    except Exception:
        return float('nan')

# Only run when user presses the button
if st.session_state['build_index']:
    if len(tickers_list) == 0:
        st.error("Please enter at least one ticker.")
        st.stop()

    dataframes = []
    returns = []
    drawdowns = []
    sharpes = []
    sortinos = []

    progress = st.progress(0)
    for idx, ticker_sym in enumerate(tickers_list):
        df = get_history_df(ticker_sym, start_date.isoformat(), end_date.isoformat())
        market_cap_val = get_market_cap_value(ticker_sym)

        # compute outstanding shares using market cap and last close (same logic as original)
        try:
            last_close = df.loc[df.shape[0]-1, 'Close']
            if np.isnan(market_cap_val):
                outstanding_shares = float('nan')
            else:
                outstanding_shares = market_cap_val / last_close
        except Exception:
            outstanding_shares = float('nan')

        df['Market Cap'] = outstanding_shares * df['Close']
        dataframes.append(df)

        # individual metrics as original
        try:
            r = (df.loc[df.shape[0]-1,'Close'] / df.loc[0,'Close'] - 1) * 100
        except Exception:
            r = np.nan
        returns.append(r)
        drawdowns.append(max_drawdown(df['Close']))
        sharpe, sortino = risk_adjusted_ratios(df['Close'], risk_free_rate=risk_free_rate)
        sharpes.append(sharpe)
        sortinos.append(sortino)

        progress.progress(int((idx+1)/len(tickers_list)*100))

    # Build merged_df as in original script (inner-join on Date)
    merged_df = pd.DataFrame({'Date': dataframes[0]['Date']})
    for i, df in enumerate(dataframes):
        df_subset = df[['Date', 'Open', 'High', 'Low', 'Close', 'Market Cap']].rename(
            columns={'Open':f'Open_{i}','High':f'High_{i}','Low':f'Low_{i}','Close':f'Close_{i}', 'Market Cap':f'MarketCap_{i}'}
        )
        merged_df = merged_df.merge(df_subset, on='Date', how='inner')

    market_cap_cols = [col for col in merged_df.columns if 'MarketCap' in col]
    merged_df['TotalMarketCap'] = merged_df[market_cap_cols].sum(axis=1)

    # Create Weight_i columns according to chosen weighting technique
    if weighting_choice == "Market Cap":
        for i in range(len(dataframes)):
            merged_df[f'Weight_{i}'] = merged_df[f'MarketCap_{i}'] / merged_df['TotalMarketCap']
    elif weighting_choice == "Equal":
        n = len(dataframes)
        for i in range(n):
            merged_df[f'Weight_{i}'] = 1.0 / n
    elif weighting_choice == "Custom":
        try:
            user_weights = [float(x.strip()) for x in custom_weights_text.split(",") if x.strip()!='']
        except Exception:
            user_weights = []
        n = len(dataframes)
        if len(user_weights) != n:
            if len(user_weights) < n:
                pad = [1.0/n] * (n - len(user_weights))
                user_weights = user_weights + pad
            else:
                user_weights = user_weights[:n]
        uw = np.array(user_weights, dtype=float)
        if np.nansum(uw) == 0:
            uw = np.array([1.0/n]*n)
        else:
            uw = uw / np.nansum(uw)
        for i in range(n):
            merged_df[f'Weight_{i}'] = float(uw[i])

    merged_df.drop(columns=[f'MarketCap_{i}' for i in range(0, len(tickers_list))], inplace=True, errors='ignore')
    merged_df.drop(columns=['TotalMarketCap'], inplace=True, errors='ignore')

    merged_df['Index Open'] = sum(merged_df[f'Weight_{i}'] * dataframes[i].loc[dataframes[i]['Date'].isin(merged_df['Date']), 'Open'].values for i in range(len(dataframes)))
    merged_df['Index High'] = sum(merged_df[f'Weight_{i}'] * dataframes[i].loc[dataframes[i]['Date'].isin(merged_df['Date']), 'High'].values for i in range(len(dataframes)))
    merged_df['Index Low']  = sum(merged_df[f'Weight_{i}'] * dataframes[i].loc[dataframes[i]['Date'].isin(merged_df['Date']), 'Low'].values for i in range(len(dataframes)))
    merged_df['Index Close']= sum(merged_df[f'Weight_{i}'] * dataframes[i].loc[dataframes[i]['Date'].isin(merged_df['Date']), 'Close'].values for i in range(len(dataframes)))

    merged_df.drop(columns=[f'Open_{i}' for i in range(0,len(tickers_list))], inplace=True, errors='ignore')
    merged_df.drop(columns=[f'High_{i}' for i in range(0,len(tickers_list))], inplace=True, errors='ignore')
    merged_df.drop(columns=[f'Low_{i}' for i in range(0,len(tickers_list))], inplace=True, errors='ignore')
    merged_df.drop(columns=[f'Close_{i}' for i in range(0,len(tickers_list))], inplace=True, errors='ignore')

    avg_weights = [merged_df[f'Weight_{i}'].mean() for i in range(len(tickers_list))]
    merged_df.drop(columns=[f'Weight_{i}' for i in range(0,len(tickers_list))], inplace=True, errors='ignore')

    individual_results = {
        "Return (%)": returns,
        "Maxmimum Drawdown (%)": drawdowns,
        "Sharpe Ratio": sharpes,
        "Sortino Ratio": sortinos,
        "Average Weights": avg_weights
    }

    # --- Display results ---
    st.header("Index Candlestick")
    left, right = st.columns([3,1])

    # Main candlestick
    fig = go.Figure(data=[
        go.Candlestick(
            x=merged_df['Date'],
            open=merged_df['Index Open'],
            high=merged_df['Index High'],
            low=merged_df['Index Low'],
            close=merged_df['Index Close']
        )
    ])
    fig.update_layout(
        title='Index Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600
    )
    left.plotly_chart(fig, use_container_width=True)

    # Index summary metrics including CAGR
    try:
        final_return = (merged_df.loc[merged_df.shape[0]-1,'Index Close'] / merged_df.loc[0,'Index Close'] - 1) * 100
    except Exception:
        final_return = np.nan

    final_maxdd = max_drawdown(merged_df['Index Close'])
    final_sharpe, final_sortino = risk_adjusted_ratios(merged_df['Index Close'], risk_free_rate=risk_free_rate)

    # Compute CAGR using actual index date range
    try:
        start_price = merged_df.loc[0, 'Index Close']
        end_price = merged_df.loc[merged_df.shape[0]-1, 'Index Close']
        days = (merged_df.loc[merged_df.shape[0]-1, 'Date'] - merged_df.loc[0, 'Date']).days
        years = days / 365.2425 if days > 0 else 0.0
        if years > 0 and start_price > 0:
            cagr = (end_price / start_price) ** (1.0/years) - 1.0
        else:
            cagr = np.nan
    except Exception:
        cagr = np.nan

    with right:
        # two small columns so Index Return and CAGR appear side-by-side
        c1, c2 = st.columns(2)
        c1.metric("Index Return (%)", f"{final_return:.2f}" if not pd.isna(final_return) else "N/A")
        c2.metric("Index CAGR (%)", f"{(cagr*100):.2f}" if not pd.isna(cagr) else "N/A")

        st.metric("Index Max Drawdown (%)", f"{final_maxdd:.2f}" if not pd.isna(final_maxdd) else "N/A")
        st.metric("Index Sharpe", f"{final_sharpe:.4f}" if not pd.isna(final_sharpe) else "N/A")
        st.metric("Index Sortino", f"{final_sortino:.4f}" if not pd.isna(final_sortino) else "N/A")
        st.markdown("---")
        st.write("Selected weighting:", weighting_choice)
        if weighting_choice == "Custom":
            st.write("Custom weights (normalized):", list(np.round(uw,6)))
        st.write("Risk-free rate (%)", f"{risk_free_rate_pct:.4f}")

    # -----------------------
    # Comparison: SENSEX & NIFTY50 (stacked layout) — aligned intersection logic
    # -----------------------
    st.markdown("### Comparison with SENSEX and NIFTY50")
    comp_df = merged_df[['Date', 'Index Close']].copy().rename(columns={'Index Close': 'My Index'})

    sensex_df = get_history_df("^BSESN", start_date.isoformat(), end_date.isoformat())
    nifty_df  = get_history_df("^NSEI", start_date.isoformat(), end_date.isoformat())

    sensex_subset = sensex_df[['Date', 'Close']].rename(columns={'Close': 'SENSEX'})
    nifty_subset = nifty_df[['Date', 'Close']].rename(columns={'Close': 'NIFTY50'})

    comp_df = comp_df.merge(sensex_subset, on='Date', how='left')
    comp_df = comp_df.merge(nifty_subset, on='Date', how='left')

    # Align to overlapping (intersection) dates where ALL three series have data
    aligned = comp_df.dropna(subset=['My Index', 'SENSEX', 'NIFTY50']).reset_index(drop=True)

    if aligned.shape[0] < 2:
        st.warning("SENSEX and/or NIFTY50 do not share enough overlapping dates with your index. "
                   "Comparison metrics will be computed per-series (may use slightly different start/end dates).")
        def compute_metrics_fallback(series):
            s = pd.Series(series).dropna()
            if s.shape[0] < 2:
                return np.nan, np.nan, np.nan, np.nan
            r = (s.iloc[-1] / s.iloc[0] - 1) * 100
            dd = max_drawdown(s)
            sh, so = risk_adjusted_ratios(s, risk_free_rate=risk_free_rate)
            return r, dd, sh, so

        my_metrics = compute_metrics_fallback(comp_df['My Index'])
        sensex_metrics = compute_metrics_fallback(comp_df['SENSEX'])
        nifty_metrics = compute_metrics_fallback(comp_df['NIFTY50'])
        norm_plot_df = comp_df.copy()
        st.markdown("**Metrics (fallback — per series)**")
    else:
        def compute_metrics_aligned(df, col):
            s = pd.Series(df[col]).dropna()
            r = (s.iloc[-1] / s.iloc[0] - 1) * 100
            dd = max_drawdown(s)
            sh, so = risk_adjusted_ratios(s, risk_free_rate=risk_free_rate)
            return r, dd, sh, so

        my_metrics = compute_metrics_aligned(aligned, 'My Index')
        sensex_metrics = compute_metrics_aligned(aligned, 'SENSEX')
        nifty_metrics = compute_metrics_aligned(aligned, 'NIFTY50')
        norm_plot_df = aligned.copy()
        st.markdown("**Metrics (same period — intersection of available dates for all three series)**")

    comparison_df = pd.DataFrame(
        [my_metrics, sensex_metrics, nifty_metrics],
        index=['My Index', 'SENSEX', 'NIFTY50'],
        columns=['Return (%)', 'Maxmimum Drawdown (%)', 'Sharpe Ratio', 'Sortino Ratio']
    )

    st.dataframe(comparison_df.style.format({
        "Return (%)": "{:.2f}",
        "Maxmimum Drawdown (%)": "{:.2f}",
        "Sharpe Ratio": "{:.4f}",
        "Sortino Ratio": "{:.4f}"
    }), height=220)

    # Normalized price series using the same frame used for metrics (norm_plot_df)
    norm_df = norm_plot_df.copy()
    for col in ['My Index', 'SENSEX', 'NIFTY50']:
        first_valid = norm_df[col].dropna()
        if first_valid.shape[0] > 0:
            base = first_valid.iloc[0]
            if base == 0 or pd.isna(base):
                norm_df[col] = np.nan
            else:
                norm_df[col] = norm_df[col] / base * 100.0
        else:
            norm_df[col] = np.nan

    st.markdown("**Normalized Close (base = 100 at first available date in the comparison period)**")
    fig2 = go.Figure()
    for col in ['My Index', 'SENSEX', 'NIFTY50']:
        if norm_df[col].dropna().shape[0] > 0:
            fig2.add_trace(go.Scatter(x=norm_df['Date'], y=norm_df[col], mode='lines', name=col))
    fig2.update_layout(
        xaxis_title="Date",
        yaxis_title="Normalized Price (100 = start)",
        height=320
    )
    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------
    # Individual results (per-stock) - kept below comparison
    # -----------------------
    results_df = pd.DataFrame(individual_results, index=tickers_list)
    st.markdown("### Individual results")
    st.dataframe(results_df.style.format({
        "Return (%)": "{:.2f}",
        "Maxmimum Drawdown (%)": "{:.2f}",
        "Sharpe Ratio": "{:.4f}",
        "Sortino Ratio": "{:.4f}",
        "Average Weights": "{:.6f}"
    }), height=300)

    st.success("Index built successfully.")
else:
    st.info("Fill inputs above and click **Build index** to run.")
