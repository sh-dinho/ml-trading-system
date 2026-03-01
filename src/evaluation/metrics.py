import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class TradingMetrics:
    """Calculate trading performance metrics."""
    
    @staticmethod
    def calculate_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate log returns."""
        return np.diff(np.log(prices))
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio."""
        excess_returns = returns - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def max_drawdown(prices: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + TradingMetrics.calculate_returns(prices))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    @staticmethod
    def win_rate(predictions: np.ndarray, actual_returns: np.ndarray) -> float:
        """Calculate win rate of predictions."""
        correct_direction = np.sign(predictions) == np.sign(actual_returns)
        return np.sum(correct_direction) / len(actual_returns)
    
    @staticmethod
    def cumulative_return(prices: np.ndarray) -> float:
        """Calculate cumulative return."""
        return (prices[-1] - prices[0]) / prices[0]
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }