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
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).rename(f"RSI_{window}")

def roc(df: pd.DataFrame, window: int = 10) -> pd.Series:
    return _close(df).pct_change(periods=window).rename(f"ROC_{window}")

def macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    close = _close(df)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({
        f"MACD_{fast}_{slow}": macd_line,
        f"MACD_signal_{signal}": signal_line,
        f"MACD_hist_{fast}_{slow}_{signal}": hist
    })

def stochastic(df: pd.DataFrame, k_window=14, d_window=3) -> pd.DataFrame:
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = _close(df)
    lowest_low = low.rolling(k_window).min()
    highest_high = high.rolling(k_window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_window).mean()
    return pd.DataFrame({
        f"STOCH_K_{k_window}": k,
        f"STOCH_D_{k_window}_{d_window}": d
    })


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

def atr(df: pd.DataFrame, window=14) -> pd.Series:
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = _close(df)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean().rename(f"ATR_{window}")


# ---------------- Price Action ----------------
def returns(df: pd.DataFrame) -> pd.Series:
    return _close(df).pct_change(fill_method=None).rename("returns")

def log_returns(df: pd.DataFrame) -> pd.Series:
    return np.log(_close(df) / _close(df).shift(1)).rename("log_returns")


# ---------------- Volume-Based ----------------
def obv(df: pd.DataFrame) -> pd.Series:
    close = _close(df)
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum().rename("OBV")

def vroc(df: pd.DataFrame, window: int = 10) -> pd.Series:
    return df["Volume"].pct_change(window).rename(f"VROC_{window}")

def cmf(df: pd.DataFrame, window=20) -> pd.Series:
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = _close(df)
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    mfm = ((close - low) - (high - close)) / (high - low)
    mfv = mfm * volume
    return (mfv.rolling(window).sum() / volume.rolling(window).sum()).rename(f"CMF_{window}")