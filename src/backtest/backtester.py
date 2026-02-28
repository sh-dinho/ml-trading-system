from dataclasses import dataclass, asdict
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import mlflow


@dataclass
class BacktestConfig:
    threshold_long: float = 0.0
    threshold_short: float = 0.0
    cost_bps: float = 10.0
    max_leverage: float = 1.0


class Backtester:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def _generate_signals(self, preds: pd.Series) -> pd.Series:
        s = pd.Series(0, index=preds.index, dtype=int)
        s[preds > self.config.threshold_long] = 1
        s[preds < self.config.threshold_short] = -1
        return s

    def _compute_transaction_costs(self, positions: pd.Series) -> pd.Series:
        pos_change = positions.diff().abs().fillna(0)
        cost_rate = self.config.cost_bps / 10000.0
        return pos_change * cost_rate

    def run(self, df: pd.DataFrame, preds: pd.Series, ticker: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
        df = df.copy()
        df["pred"] = preds

        df["position"] = self._generate_signals(df["pred"])
        df["ret"] = df["target"].astype(float)
        df["strategy_gross"] = df["position"] * df["ret"]
        df["cost"] = self._compute_transaction_costs(df["position"])
        df["strategy_net"] = df["strategy_gross"] - df["cost"]
        df["equity"] = (1 + df["strategy_net"]).cumprod()

        metrics = self._compute_metrics(df)

        # MLflow logging
        with mlflow.start_run(run_name=f"backtest_{ticker}"):
            mlflow.log_params(asdict(self.config))
            mlflow.log_metrics(metrics)

            # Save artifacts
            equity_path = f"equity_{ticker}.csv"
            df[["equity"]].to_csv(equity_path)
            mlflow.log_artifact(equity_path)

            full_log_path = f"backtest_full_{ticker}.csv"
            df.to_csv(full_log_path)
            mlflow.log_artifact(full_log_path)

        return df, metrics

    def _compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        strat = df["strategy_net"]
        daily_ret = strat
        mean_daily = daily_ret.mean()
        std_daily = daily_ret.std(ddof=0)

        sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0.0

        equity = df["equity"]
        roll_max = equity.cummax()
        drawdown = equity / roll_max - 1.0
        max_dd = drawdown.min()

        total_ret = equity.iloc[-1] - 1.0

        wins = (strat > 0).sum()
        losses = (strat < 0).sum()
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

        return {
            "total_return": float(total_ret),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "avg_daily_return": float(mean_daily),
            "std_daily_return": float(std_daily),
        }