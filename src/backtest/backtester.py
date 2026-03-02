from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
import mlflow


# ============================================================
# CONFIG
# ============================================================

@dataclass
class BacktestConfig:
    threshold_long: float = 0.0
    threshold_short: float = 0.0
    cost_bps: float = 10.0

    # Position sizing
    sizing: str = "equal"          # equal | vol | risk_parity | signal
    vol_window: int = 20           # for volatility-based sizing


# ============================================================
# BACKTESTER
# ============================================================

class Backtester:
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    # --------------------------------------------------------
    # SIGNAL GENERATION
    # --------------------------------------------------------
    def _generate_signals(self, preds: pd.Series) -> pd.Series:
        s = pd.Series(0, index=preds.index, dtype=int)
        s[preds > self.config.threshold_long] = 1
        s[preds < self.config.threshold_short] = -1
        return s

    # --------------------------------------------------------
    # TRANSACTION COSTS
    # --------------------------------------------------------
    def _compute_costs(self, pos: pd.Series) -> pd.Series:
        return pos.diff().abs().fillna(0) * self.config.cost_bps / 10000

    # --------------------------------------------------------
    # POSITION SIZING (PORTFOLIO)
    # --------------------------------------------------------
    def _compute_weights(self, data: Dict[str, pd.DataFrame], preds: Dict[str, pd.Series]):
        sizing = self.config.sizing
        tickers = list(data.keys())

        # Equal weight
        if sizing == "equal":
            return {t: 1 / len(tickers) for t in tickers}

        # Volatility targeting
        if sizing == "vol":
            vols = {
                t: data[t]["target"].rolling(self.config.vol_window).std().iloc[-1]
                for t in tickers
            }
            inv_vol = {t: 1 / v if v > 0 else 0 for t, v in vols.items()}
            total = sum(inv_vol.values())
            return {t: w / total for t, w in inv_vol.items()}

        # Risk parity
        if sizing == "risk_parity":
            rets = pd.DataFrame({t: data[t]["target"] for t in tickers})
            cov = rets.cov()
            vols = np.sqrt(np.diag(cov))
            inv_risk = {t: 1 / vols[i] for i, t in enumerate(tickers)}
            total = sum(inv_risk.values())
            return {t: w / total for t, w in inv_risk.items()}

        # Signal-scaled
        if sizing == "signal":
            strengths = {t: preds[t].abs().mean() for t in tickers}
            total = sum(strengths.values())
            return {t: w / total for t, w in strengths.items()}

        raise ValueError(f"Unknown sizing method: {sizing}")

    # --------------------------------------------------------
    # SINGLE‑TICKER BACKTEST
    # --------------------------------------------------------
    def _run_single(self, df: pd.DataFrame, preds: pd.Series) -> pd.DataFrame:
        df = df.copy()
        df["pred"] = preds
        df["position"] = self._generate_signals(df["pred"])
        df["ret"] = df["target"].astype(float)

        df["strategy_gross"] = df["position"] * df["ret"]
        df["cost"] = self._compute_costs(df["position"])
        df["strategy_net"] = df["strategy_gross"] - df["cost"]

        df["equity"] = (1 + df["strategy_net"]).cumprod()

        # Benchmark (buy‑and‑hold)
        df["benchmark"] = (1 + df["ret"]).cumprod()

        return df

    # --------------------------------------------------------
    # TRADE‑LEVEL ANALYTICS
    # --------------------------------------------------------
    def _extract_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        pos = df["position"]
        strat = df["strategy_net"]
        eq = df["equity"]

        trades = []
        in_trade = False
        entry_idx = None
        entry_pos = 0
        entry_eq = 1.0

        for i in range(1, len(df)):
            prev_pos = pos.iloc[i - 1]
            curr_pos = pos.iloc[i]

            # Entry
            if not in_trade and curr_pos != 0:
                in_trade = True
                entry_idx = df.index[i]
                entry_pos = curr_pos
                entry_eq = eq.iloc[i - 1]

            # Exit
            if in_trade and (curr_pos == 0 or curr_pos != entry_pos):
                exit_idx = df.index[i]
                trade_slice = df.loc[entry_idx:exit_idx]

                pnl = trade_slice["strategy_net"].sum()
                ret = eq.loc[exit_idx] / entry_eq - 1.0
                duration = len(trade_slice)

                cum = (1 + trade_slice["strategy_net"]).cumprod()
                mae = (cum.cummax() - cum).max()
                mfe = (cum - cum.cummin()).max()

                trades.append({
                    "entry": entry_idx,
                    "exit": exit_idx,
                    "side": "long" if entry_pos == 1 else "short",
                    "pnl": pnl,
                    "return": ret,
                    "duration": duration,
                    "mae": mae,
                    "mfe": mfe,
                })

                in_trade = False

        return pd.DataFrame(trades)

    # --------------------------------------------------------
    # TRADE‑LEVEL METRICS
    # --------------------------------------------------------
    def _compute_trade_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        if trades.empty:
            return {
                "num_trades": 0,
                "win_rate_trades": 0.0,
                "avg_trade_return": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "avg_duration": 0.0,
            }

        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]

        gross_profit = wins["pnl"].sum()
        gross_loss = -losses["pnl"].sum()

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        expectancy = (
            wins["return"].mean() * len(wins) -
            abs(losses["return"].mean()) * len(losses)
        ) / len(trades)

        return {
            "num_trades": len(trades),
            "win_rate_trades": len(wins) / len(trades),
            "avg_trade_return": trades["return"].mean(),
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "avg_duration": trades["duration"].mean(),
        }

    # --------------------------------------------------------
    # DAILY METRICS
    # --------------------------------------------------------
    def _compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        strat = df["weighted_net"] if "weighted_net" in df else df["strategy_net"]

        daily = strat
        mean_daily = daily.mean()
        std_daily = daily.std(ddof=0)
        sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0.0

        eq = df["equity"]
        dd = (eq.cummax() - eq) / eq.cummax()
        max_dd = dd.max()

        total_ret = eq.iloc[-1] - 1.0

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

    # --------------------------------------------------------
    # PORTFOLIO BACKTEST
    # --------------------------------------------------------
    def run_portfolio(self, data: Dict[str, pd.DataFrame], preds: Dict[str, pd.Series]):
        weights = self._compute_weights(data, preds)

        results = []
        for t in data:
            df_single = self._run_single(data[t], preds[t])
            df_single["ticker"] = t
            df_single["weight"] = weights[t]
            results.append(df_single)

        combined = pd.concat(results).sort_index()

        combined["weighted_net"] = combined["strategy_net"] * combined["weight"]

        portfolio = combined.groupby(combined.index).agg({
            "weighted_net": "sum",
            "benchmark": "mean"
        })

        portfolio["equity"] = (1 + portfolio["weighted_net"]).cumprod()
        portfolio["benchmark_equity"] = (1 + portfolio["benchmark"]).cumprod()

        metrics = self._compute_metrics(portfolio)

        return portfolio, metrics

    # --------------------------------------------------------
    # SINGLE‑TICKER BACKTEST (MLflow)
    # --------------------------------------------------------
    def run(self, df: pd.DataFrame, preds: pd.Series, ticker: str):
        df = self._run_single(df, preds)
        trades = self._extract_trades(df)
        trade_metrics = self._compute_trade_metrics(trades)
        daily_metrics = self._compute_metrics(df)

        all_metrics = {**daily_metrics, **trade_metrics}

        with mlflow.start_run(run_name=f"backtest_{ticker}"):
            mlflow.log_params(asdict(self.config))
            mlflow.log_metrics(all_metrics)

        return df, all_metrics