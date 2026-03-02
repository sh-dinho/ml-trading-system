from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.trading_metrics import TradingMetrics


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    transaction_cost: float = 0.0005  # per trade
    benchmark_col: str = "benchmark"  # optional


class Backtester:
    def __init__(self, config: BacktestConfig):
        self.cfg = config

    # --------------------------------------------------------
    # Single-ticker backtest
    # --------------------------------------------------------
    def run(
        self,
        df: pd.DataFrame,
        preds: pd.Series,
        ticker: str,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        df = df.copy()
        df["signal"] = np.sign(preds)
        df["return"] = df["close"].pct_change().fillna(0.0)
        df["strategy_return"] = df["signal"].shift(1).fillna(0.0) * df["return"]

        # transaction costs on signal changes
        trades = df["signal"].diff().abs().fillna(0.0)
        df["strategy_return"] -= trades * self.cfg.transaction_cost

        df["equity"] = (1 + df["strategy_return"]).cumprod() * self.cfg.initial_capital

        if self.cfg.benchmark_col in df.columns:
            df["benchmark_equity"] = (1 + df[self.cfg.benchmark_col]).cumprod() * self.cfg.initial_capital

        returns = df["strategy_return"].to_numpy()
        prices = df["equity"].to_numpy()

        metrics = {
            "sharpe": TradingMetrics.sharpe_ratio(returns),
            "max_drawdown": TradingMetrics.max_drawdown_from_returns(returns),
            "cumulative_return": TradingMetrics.cumulative_return(prices),
        }

        return df, metrics

    # --------------------------------------------------------
    # Portfolio backtest (equal-weight)
    # --------------------------------------------------------
    def run_portfolio(
        self,
        data: Dict[str, pd.DataFrame],
        preds: Dict[str, pd.Series],
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        aligned = []
        for ticker, df in data.items():
            d = df.copy()
            p = preds[ticker].reindex(d.index).fillna(0.0)
            d["signal"] = np.sign(p)
            d["return"] = d["close"].pct_change().fillna(0.0)
            d["strategy_return"] = d["signal"].shift(1).fillna(0.0) * d["return"]
            aligned.append(d[["strategy_return"]].rename(columns={"strategy_return": ticker}))

        port = pd.concat(aligned, axis=1).fillna(0.0)
        port["portfolio_return"] = port.mean(axis=1)

        port["equity"] = (1 + port["portfolio_return"]).cumprod() * self.cfg.initial_capital

        returns = port["portfolio_return"].to_numpy()
        prices = port["equity"].to_numpy()

        metrics = {
            "sharpe": TradingMetrics.sharpe_ratio(returns),
            "max_drawdown": TradingMetrics.max_drawdown_from_returns(returns),
            "cumulative_return": TradingMetrics.cumulative_return(prices),
        }

        return port, metrics