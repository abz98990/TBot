# module_2_features.py
import pandas as pd
import pandas_ta as ta  # pyrefly: ignore [missing-import]
import numpy as np
import os


# ---------------------------------------------------------------------------
# RollingNormalizer
# ---------------------------------------------------------------------------
# Replaces sklearn StandardScaler. Instead of fitting once on training data
# and freezing those global mean/std values forever, this computes mean and
# std over a rolling lookback window of recent candles on every call.
#
# Why this matters:
#   - A StandardScaler fitted when BTC was at $30k encodes that price regime.
#     When BTC moves to $90k, the "normalized" features are way outside the
#     distribution the LSTM was calibrated on — silently killing accuracy.
#   - With rolling normalization, every feature is always expressed relative
#     to its own recent history, so the LSTM sees stationary inputs regardless
#     of the absolute price level or volatility regime.
#
# Drop-in compatibility: the public API mirrors the old scaler so main.py
# and train_offline.py need only tiny changes (documented below).
# ---------------------------------------------------------------------------

class RollingNormalizer:
    """
    Per-feature rolling Z-score normalizer.

    Parameters
    ----------
    window : int
        Number of past candles used to compute mean and std.
        200 is a good default — long enough to be stable, short enough
        to track regime shifts over days/weeks.
    min_periods : int
        Minimum number of valid rows before normalizing kicks in.
        Rows before this threshold are filled with 0.0 (neutral).
    eps : float
        Floor added to std to prevent division-by-zero on flat features.
    """

    def __init__(self, window: int = 200, min_periods: int = 30, eps: float = 1e-8):
        self.window = window
        self.min_periods = min_periods
        self.eps = eps
        # These store the last computed stats so live inference can
        # normalize a single new row using the same window params.
        self._last_mean: pd.Series | None = None
        self._last_std: pd.Series | None = None

    # ------------------------------------------------------------------
    # Core transform
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Normalize `columns` in-place using a rolling window.
        Called during both offline training and the live calibration loop
        (replaces the old is_training=True branch).

        The rolling stats are computed on the full DataFrame passed in,
        so the window anchors naturally to whatever slice of history you
        provide. The last row's stats are cached for single-row inference.
        """
        df = df.copy()

        rolling_mean = df[columns].rolling(window=self.window, min_periods=self.min_periods).mean()
        rolling_std  = df[columns].rolling(window=self.window, min_periods=self.min_periods).std()

        # Cache the final row's stats — used by transform() for live inference
        self._last_mean = rolling_mean.iloc[-1]
        self._last_std  = rolling_std.iloc[-1]

        # Z-score: (x - mean) / (std + eps)
        # fillna(0.0) handles the warm-up rows before min_periods is reached
        normalized = (df[columns] - rolling_mean) / (rolling_std + self.eps)
        df[columns] = normalized.fillna(0.0)

        return df

    def transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Normalize using the cached stats from the last fit_transform call.
        Use this for single-row live inference after the calibration step
        has already run fit_transform on the full recent history.

        If fit_transform hasn't been called yet (cold start), falls back
        to computing stats from the passed-in DataFrame directly.
        """
        df = df.copy()

        if self._last_mean is None or self._last_std is None:
            # Cold start fallback: compute stats from whatever rows we have
            print("[NORMALIZER] Warning: transform() called before fit_transform(). "
                  "Computing stats from current data (cold start).")
            return self.fit_transform(df, columns)

        normalized = (df[columns] - self._last_mean) / (self._last_std + self.eps)
        df[columns] = normalized.fillna(0.0)
        return df

    # ------------------------------------------------------------------
    # Persistence (no-ops — rolling normalizer needs no saved state)
    # ------------------------------------------------------------------
    # The old code saved a fitted StandardScaler to disk with joblib and
    # reloaded it on every bot restart. That's what caused the stale-stats
    # problem. Rolling normalization computes fresh stats on every cycle
    # from the current market data, so there is nothing to persist.
    #
    # These stubs keep call-site compatibility in main.py / train_offline.py
    # so you can remove save/load calls gradually without breaking anything.

    def save(self, filepath: str) -> None:
        """No-op. Rolling normalizer has no state to persist."""
        print(f"[NORMALIZER] save() skipped — rolling normalizer needs no saved state.")

    def load(self, filepath: str) -> None:
        """No-op. Stats are recomputed live on every cycle."""
        print(f"[NORMALIZER] load() skipped — stats are computed fresh each cycle.")


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------

class FeatureEngineer:
    def __init__(self, window_size: int = 60, norm_window: int = 200):
        """
        Parameters
        ----------
        window_size : int
            LSTM lookback in candles (unchanged from v2).
        norm_window : int
            Rolling window for the normalizer. 200 candles ≈ 3 days on 15m,
            ~8 days on 1h. Tune down to 100 for faster regime tracking,
            up to 500 for more stable normalization on daily charts.
        """
        self.window_size = window_size
        self.feature_columns: list[str] = []
        self.normalizer = RollingNormalizer(window=norm_window)

        # ----------------------------------------------------------------
        # Back-compat shim: old code accessed engineer.scaler directly
        # (e.g. main.py line 227: engineer.scaler.transform(...))
        # Pointing .scaler at the normalizer means that call still works
        # without editing main.py right now — but you should migrate it
        # to engineer.normalize_data() in the next cleanup pass.
        # ----------------------------------------------------------------
        self.scaler = self.normalizer

    # ------------------------------------------------------------------
    # Scaler persistence shims (keep old call-sites working)
    # ------------------------------------------------------------------

    def save_scaler(self, filepath: str = "models/scaler.pkl") -> None:
        """Shim for train_offline.py compatibility. Rolling normalizer has no file to save."""
        self.normalizer.save(filepath)

    def load_scaler(self, filepath: str = "models/scaler.pkl") -> None:
        """Shim for main.py compatibility. Rolling normalizer needs no file to load."""
        self.normalizer.load(filepath)

    # ------------------------------------------------------------------
    # Feature engineering (unchanged from v2)
    # ------------------------------------------------------------------

    def apply_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates stationary technical indicators."""
        print("[ENGINEER] Synthesizing Technical Indicators...")

        # 1. Momentum: RSI (Bounded 0–100, already relative)
        df['RSI_14'] = df.ta.rsi(length=14)

        # 2. Volatility: ATR as % of close (already ratio-based, regime-neutral)
        atr = df.ta.atr(length=14)
        df['ATR_pct'] = atr / df['close']

        # 3. Trend: % distance from 50 EMA (ratio-based, regime-neutral)
        ema_50 = df['close'].ewm(span=50, adjust=False).mean()
        df['EMA_50_dist'] = (df['close'] - ema_50) / ema_50

        # 4. Momentum: MACD line and signal
        #    NOTE: Raw MACD values ARE price-level-dependent (they're in $ units).
        #    The rolling normalizer handles this — but we also divide by close
        #    here as a first-pass ratio transform to help the normalizer converge
        #    faster across coins with very different price magnitudes.
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        df['MACD_12_26_9']  = df['MACD_12_26_9']  / df['close']
        df['MACDs_12_26_9'] = df['MACDs_12_26_9'] / df['close']

        # 5. Trend Strength: ADX (0–100, already bounded)
        adx = df.ta.adx(length=14)
        df = pd.concat([df, adx], axis=1)

        df.dropna(inplace=True)

        # Final feature set — 6 dimensions, matches input_size=6 in ModelEngine
        self.feature_columns = [
            'RSI_14', 'ATR_pct', 'EMA_50_dist',
            'MACD_12_26_9', 'MACDs_12_26_9', 'ADX_14'
        ]
        return df

    def engineer_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Binary forward label: 1 if close 4 candles ahead > current close, else 0.
        """
        print("[ENGINEER] Calculating Forward Classification Labels (Target)...")
        df['target_class'] = (df['close'].shift(-4) > df['close']).astype(int)
        df.dropna(inplace=True)
        return df

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Apply rolling Z-score normalization to self.feature_columns.

        The `is_training` flag is kept for call-site compatibility but no
        longer changes behaviour — rolling normalization is identical for
        training and live inference. Both paths call fit_transform() so the
        cached stats always reflect the most recent window of data passed in.

        Migration note for main.py:
          OLD (broken):
            df_train = engineer.normalize_data(df_train, is_training=False)
            ...
            df_infer[cols] = engineer.scaler.transform(df_infer[cols])  # stale stats!

          NEW (correct):
            df_train = engineer.normalize_data(df_train)   # updates cached stats
            ...
            df_infer = engineer.normalize_data(df_infer)   # uses same fresh stats
        """
        print("[ENGINEER] Applying Rolling Z-Score Normalization "
              f"(window={self.normalizer.window} candles)...")

        if not self.feature_columns:
            raise RuntimeError(
                "feature_columns is empty. Call apply_technical_indicators() first."
            )

        # Always fit_transform — this updates the cached mean/std to the
        # latest window in whatever df is passed in.
        df = self.normalizer.fit_transform(df, self.feature_columns)
        return df

    # ------------------------------------------------------------------
    # Tensor shaping (unchanged from v2)
    # ------------------------------------------------------------------

    def create_3d_tensor(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Transforms the 2D DataFrame into a 3D NumPy array for the LSTM.
        Returns X (Features Tensor) and y (Target Array).
        """
        print(f"[ENGINEER] Shaping 3D Tensor (Window Size: {self.window_size})...")

        feature_data = df[self.feature_columns].values
        target_data  = df['target_class'].values

        X, y = [], []
        for i in range(self.window_size, len(df)):
            X.append(feature_data[i - self.window_size: i])
            y.append(target_data[i])

        X = np.array(X)
        y = np.array(y)

        print(f"[SUCCESS] Tensor X shape: {X.shape} | Target y shape: {y.shape}")
        return X, y