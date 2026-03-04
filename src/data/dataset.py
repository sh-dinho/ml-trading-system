class DatasetBuilder:
    def __init__(self, horizon=5):
        self.horizon = horizon

    def build_dataset(self, features_df: pd.DataFrame) -> pd.DataFrame:
        df = features_df.copy()
        
        # v1.1: Standardized Target Creation
        # Shift -5 means today's features are paired with the return 5 days from now
        df["target"] = df["returns"].shift(-self.horizon)
        
        # Clean edges: Indicators (e.g. SMA 200) create NaNs at the start.
        # Target shift creates NaNs at the very end.
        df = df.dropna()
        
        # Ensure no non-numeric columns remain for the ML model
        df = df.select_dtypes(include=[np.number])
        return df