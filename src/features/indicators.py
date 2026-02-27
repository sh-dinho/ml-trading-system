import pandas as pd

def sma(df: pd.DataFrame, window: int):
    return df['Close'].rolling(window=window).mean().rename(f"SMA_{window}")

def ema(df: pd.DataFrame, window: int):
    return df['Close'].ewm(span=window, adjust=False).mean().rename(f"EMA_{window}")

def rsi(df: pd.DataFrame, window: int = 14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).rename(f"RSI_{window}")

def bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2):
    sma = df['Close'].rolling(window).mean()
    std = df['Close'].rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return pd.DataFrame({
        f"BB_upper_{window}": upper,
        f"BB_lower_{window}": lower
    })