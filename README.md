# Streamlit Index Builder

**Build your own stock index** (Market-Cap Weighted, Equal Weighted, or Custom Weighted) and compare it with benchmark indices like **SENSEX** and **NIFTY50** — all in a simple **Streamlit web app**.

---

## Features
**Custom Index Creation**  
- Choose your own set of tickers (NSE symbols supported, e.g., `TCS.NS, INFY.NS`).  
- Weighting options:
  - **Market Cap Weighted**
  - **Equal Weighted**
  - **Custom Weights**

**Performance Metrics**
- Return (%)
- **CAGR (%)**
- Maximum Drawdown (%)
- **Sharpe Ratio**
- **Sortino Ratio**

**Visualizations**
- **Candlestick chart** for your custom index.
- Comparison chart vs **SENSEX** & **NIFTY50** (normalized performance).
- Individual stock performance metrics.

**Dynamic Date Selection**
- Presets: 1Y, 2Y, 3Y, 5Y, YTD.
- Custom start & end dates.

**Benchmark Comparison**
- Compare your index with **SENSEX** and **NIFTY50** (same period).

---

## Tech Stack
- **Python 3.9+**
- [Streamlit](https://streamlit.io/) (UI)
- [yfinance](https://pypi.org/project/yfinance/) (Market data)
- **pandas**, **numpy**
- **plotly** (Charts)

---
