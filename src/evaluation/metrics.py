import numpy as np
from typing import Dict
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score
)


class TradingMetrics:
    """Robust trading performance metrics."""

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------
    @staticmethod
    def log_returns(prices: np.ndarray) -> np.ndarray:
        """Compute log returns."""
        return np.diff(np.log(prices))

    @staticmethod
    def simple_returns(prices: np.ndarray) -> np.ndarray:
        """Compute simple returns."""
        return prices[1:] / prices[0:-1] - 1

    # --------------------------------------------------------
    # Sharpe Ratio
    # --------------------------------------------------------
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Compute annualized Sharpe ratio."""
        if returns.std() == 0:
            return 0.0

        excess = returns - risk_free_rate / 252
        return excess.mean() / excess.std() * np.sqrt(252)

    # --------------------------------------------------------
    # Max Drawdown
    # --------------------------------------------------------
    @staticmethod
    def max_drawdown_from_returns(returns: np.ndarray) -> float:
        """Compute max drawdown from returns."""
        equity = np.exp(np.cumsum(returns))
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return drawdown.min()

    @staticmethod
    def max_drawdown_from_prices(prices: np.ndarray) -> float:
        """Compute max drawdown from price series."""
        returns = TradingMetrics.log_returns(prices)
        return TradingMetrics.max_drawdown_from_returns(returns)

    # --------------------------------------------------------
    # Win Rate
    # --------------------------------------------------------
    @staticmethod
    def win_rate(pred_direction: np.ndarray, actual_returns: np.ndarray) -> float:
        """Percentage of correct directional predictions."""
        correct = np.sign(pred_direction) == np.sign(actual_returns)
        return correct.mean()

    # --------------------------------------------------------
    # Cumulative Return
    # --------------------------------------------------------
    @staticmethod
    def cumulative_return(prices: np.ndarray) -> float:
        """Compute cumulative return from price series."""
        return prices[-1] / prices[0] - 1

    @staticmethod
    def cumulative_return_from_returns(returns: np.ndarray) -> float:
        """Compute cumulative return from returns."""
        return np.exp(returns.sum()) - 1

    # --------------------------------------------------------
    # Classification Metrics
    # --------------------------------------------------------
    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute classification metrics."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }