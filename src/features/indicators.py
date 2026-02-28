import pandas as pd
import numpy as np

# ---------------- Utility ----------------
def _close(df: pd.DataFrame) -> pd.Series:
    if "Adj Close" in df.columns:
        return pd.to_numeric(df["Adj Close"], errors="coerce")
    return pd.to_numeric(df["Close"], errors="coerce")


# ---------------- Trend Indicators ----------------
def sma(df: pd.DataFrame, window: int) -> pd.Series:
    return _close(df).rolling(window=window).mean().rename(f"SMA_{window}")

def ema(df: pd.DataFrame, window: int) -> pd.Series:
    return _close(df).ewm(span=window, adjust=False).mean().rename(f"EMA_{window}")


# ---------------- Momentum Indicators ----------------
def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    close = _close(df)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).rename(f"RSI_{window}")

def roc(df: pd.DataFrame, window: int = 10) -> pd.Series:
    return _close(df).pct_change(periods=window).rename(f"ROC_{window}")


# ---------------- Volatility Indicators ----------------
def bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    close = _close(df)
    sma_val = close.rolling(window).mean()
    std = close.rolling(window).std()
    return pd.DataFrame({
        f"BB_upper_{window}": sma_val + num_std * std,
        f"BB_lower_{window}": sma_val - num_std * std
    })

def rolling_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    ret = returns(df)
    return ret.rolling(window).std().rename(f"volatility_{window}")


# ---------------- Price Action ----------------
def returns(df: pd.DataFrame) -> pd.Series:
    return _close(df).pct_change(fill_method=None).rename("returns")

def log_returns(df: pd.DataFrame) -> pd.Series:
    return np.log(_close(df) / _close(df).shift(1)).rename("log_returns")


# ---------------- Volume-Based ----------------
def obv(df: pd.DataFrame) -> pd.Series:
    close = _close(df)
    volume = df["Volume"]
    obv_val = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i-1]:
            obv_val.iloc[i] = obv_val.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv_val.iloc[i] = obv_val.iloc[i-1] - volume.iloc[i]
        else:
            obv_val.iloc[i] = obv_val.iloc[i-1]
    return obv_val.rename("OBV")

def vroc(df: pd.DataFrame, window: int = 10) -> pd.Series:
    return df["Volume"].pct_change(window).rename(f"VROC_{window}")