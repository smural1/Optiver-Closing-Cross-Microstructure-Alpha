# Optiver-Closing-Cross-Microstructure-Alpha


## Overview
Quantitative analysis of order book microstructure signals during the Nasdaq closing auction using data from the **Optiver - Trading at the Close** competition.

## Strategy Hypothesis
During the final 10 minutes of trading, aggressive order book imbalances often suffer from adverse selection as high-frequency liquidity providers lean against the flow. A naive momentum strategy decays rapidly, whereas a **mean-reverting contrarian strategy** captures short-term price reversals.

## Performance Metrics
- **Dataset Evaluated:** 4.2 Million Order Book Ticks
- **Information Coefficient (IC):** -0.1030


## Key Visualizations
![Cumulative PnL](pnl_plot.png)

## Strategy Script
The full strategy backtest code is contained in `main.py` using Pandas vectorized operations.
