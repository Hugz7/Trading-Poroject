# =============================================================================
# COT SIGNAL ENGINE — Calcul des 3 signaux + agrégation
# =============================================================================
# Chaque signal retourne un score entre -1.0 (très bearish) et +1.0 (très bullish).
# Le signal composite est une moyenne pondérée de ces 3 scores.
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class COTSignalEngine:
    """
    Calcule les 3 signaux COT indépendants et les combine en un signal composite.

    Signaux :
        1. OLS Regression — relation empirique Δpositioning → return J+7
        2. COT Momentum   — positioning vs sa moyenne mobile
        3. Z-score        — positionnement normalisé × direction du flux
    """

    def __init__(self, config: dict):
        self.cfg = config

    # ------------------------------------------------------------------
    # SIGNAL 1 — OLS Regression
    # ------------------------------------------------------------------

    def signal_ols_regression(
        self,
        net_pos: pd.Series,
        prices: pd.Series,
        asset: str,
    ) -> pd.Series:
        """
        Régression linéaire glissante : le changement de positioning de cette
        semaine prédit-il le return de prix de la semaine prochaine ?

        Logique :
          X = variation hebdomadaire du net positioning (Δnet_pos)
          Y = return du prix la semaine suivante (forward, décalé de -1)
          → Régression OLS sur une fenêtre glissante de 52 semaines
          → Score = return prédit normalisé par 2× l'écart-type des returns

        Score entre -1 (bearish) et +1 (bullish).
        """
        lookback   = self.cfg["lookback_weeks"]
        delta_pos  = net_pos.diff()
        fwd_return = prices.pct_change().shift(-1)

        scores = pd.Series(index=net_pos.index, dtype=float, name=f"{asset}_ols_score")

        for i in range(lookback, len(delta_pos)):
            x_window = delta_pos.iloc[i - lookback : i].dropna()
            y_window = fwd_return.iloc[i - lookback : i].reindex(x_window.index).dropna()
            x_window = x_window.reindex(y_window.index)

            if len(x_window) < 20:
                continue

            model = LinearRegression().fit(x_window.values.reshape(-1, 1), y_window.values)

            current_delta = delta_pos.iloc[i]
            if pd.isna(current_delta):
                continue

            predicted_return = model.predict([[current_delta]])[0]
            return_std = y_window.std()

            scores.iloc[i] = (
                np.clip(predicted_return / (2 * return_std), -1, 1)
                if return_std > 0 else 0.0
            )

        return scores

    # ------------------------------------------------------------------
    # SIGNAL 2 — COT Momentum
    # ------------------------------------------------------------------

    def signal_cot_momentum(self, net_pos: pd.Series, asset: str) -> pd.Series:
        """
        Momentum du positioning : les hedge funds augmentent-ils leurs positions
        dans la même direction que leur tendance récente ?

        Logique :
          MA       = moyenne mobile du net positioning sur N semaines (défaut: 4)
          momentum = net_pos - MA  (positif = au-dessus de la tendance → bullish)
          → Normalisé par l'écart-type glissant sur 52 semaines

        Score entre -1 (bearish) et +1 (bullish).
        """
        ma_pos      = net_pos.rolling(window=self.cfg["momentum_ma_weeks"]).mean()
        momentum    = net_pos - ma_pos
        rolling_std = momentum.rolling(window=self.cfg["lookback_weeks"]).std()

        scores = (momentum / rolling_std).clip(-1, 1)
        scores.name = f"{asset}_momentum_score"
        return scores

    # ------------------------------------------------------------------
    # SIGNAL 3 — Z-score
    # ------------------------------------------------------------------

    def signal_zscore(self, net_pos: pd.Series, asset: str) -> pd.Series:
        """
        Z-score du positioning × direction du changement courant.

        Logique :
          z         = (net_pos - mean_52w) / std_52w
          direction = signe du Δnet_pos (augmente ou réduit cette semaine ?)
          score     = z × direction

          Exemples :
            z très positif + ils ajoutent  → fort BULL (momentum haussier)
            z très positif + ils réduisent → BEAR  (risque de dénouement)
            z très négatif + ils ajoutent  → BULL  (short covering)

        Score normalisé entre -1 et +1 (clip à ±3σ puis /3).
        """
        rolling_mean = net_pos.rolling(window=self.cfg["lookback_weeks"]).mean()
        rolling_std  = net_pos.rolling(window=self.cfg["lookback_weeks"]).std()
        z_score      = (net_pos - rolling_mean) / rolling_std
        direction    = np.sign(net_pos.diff())

        scores = (z_score * direction).clip(-3, 3) / 3
        scores.name = f"{asset}_zscore_score"
        return scores

    # ------------------------------------------------------------------
    # AGRÉGATION — Signal Composite
    # ------------------------------------------------------------------

    def aggregate_signal(
        self,
        ols_score: pd.Series,
        momentum_score: pd.Series,
        zscore_score: pd.Series,
        asset: str,
    ) -> pd.DataFrame:
        """
        Combine les 3 scores en un signal composite pondéré.

        Score composite = w_ols × OLS + w_mom × Momentum + w_z × Z-score

        Dérive ensuite :
          - confidence_pct : |score| × 100  (0% = incertain, 100% = signal fort)
          - signal         : BULL / BEAR / HOLD selon les seuils de config

        Returns:
            DataFrame avec colonnes : composite_score, confidence_pct, signal,
            ols_score, momentum_score, zscore_score
        """
        w = self.cfg["weights"]
        composite = (
            w["ols"]      * ols_score.fillna(0)
            + w["momentum"] * momentum_score.fillna(0)
            + w["zscore"]   * zscore_score.fillna(0)
        )
        composite.name = f"{asset}_composite_score"

        confidence = (composite.abs() * 100).round(1)
        confidence.name = f"{asset}_confidence_pct"

        bull_th = self.cfg["bull_threshold"]
        bear_th = self.cfg["bear_threshold"]

        def classify(score):
            if pd.isna(score):
                return "N/A"
            if score > bull_th:
                return "BULL"
            elif score < bear_th:
                return "BEAR"
            return "HOLD"

        signal = composite.map(classify)
        signal.name = f"{asset}_signal"

        return pd.DataFrame({
            "composite_score": composite,
            "confidence_pct":  confidence,
            "signal":          signal,
            "ols_score":       ols_score,
            "momentum_score":  momentum_score,
            "zscore_score":    zscore_score,
        })
