# Optiver-Closing-Cross-Microstructure-Alpha


## Overview
Quantitative analysis of order book microstructure signals during the Nasdaq closing auction using data from the **Optiver - Trading at the Close** competition.

## Strategy Hypothesis
During the final 10 minutes of trading, aggressive order book imbalances often suffer from adverse selection as high-frequency liquidity providers lean against the flow. A naive momentum strategy decays rapidly, whereas a **mean-reverting contrarian strategy** captures short-term price reversals.

## Performance Metrics
- **Dataset Evaluated:** 250,000 Order Book Ticks
- **Information Coefficient (IC):** 0.1387
- **Mean Absolute Error (MAE):** 7.9571
- **Annualized Sharpe Ratio:** 17.38
- **Active Positions Taken:** 184,295

## Key Visualizations
![Cumulative PnL](pnl_plot.png)

## Strategy Script
The full strategy backtest code is contained in `main.py` using Pandas vectorized operations.
