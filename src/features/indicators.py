import pandas as pd
import numpy as np

def _close(df):
    return pd.to_numeric(df["Close"], errors="coerce")

def sma(df, window):
    close = _close(df)
    return close.rolling(window=window).mean().rename(f"SMA_{window}")

def ema(df, window):
    close = _close(df)
    return close.ewm(span=window, adjust=False).mean().rename(f"EMA_{window}")

def rsi(df, window=14):
    close = _close(df)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.rename(f"RSI_{window}")

def bollinger_bands(df, window=20, num_std=2):
    close = _close(df)
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()

    upper = sma + num_std * std
    lower = sma - num_std * std

    return pd.DataFrame({
        f"BB_upper_{window}": upper,
        f"BB_lower_{window}": lower
    })

def returns(df):
    close = _close(df)
    return close.pct_change(fill_method=None).rename("returns")

def log_returns(df):
    close = _close(df)
    return np.log(close / close.shift(1)).rename("log_returns")

def rolling_volatility(df, window=20):
    ret = returns(df)
    return ret.rolling(window).std().rename(f"volatility_{window}")