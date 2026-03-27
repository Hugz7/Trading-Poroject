# =============================================================================
# MOCK DATA GENERATOR — Fallback sans Bloomberg
# =============================================================================
# Génère des données synthétiques cohérentes avec la réalité du marché carbone.
# Utilisé automatiquement si Bloomberg est inaccessible.
#
# Hypothèses de calibration :
#   EUA : prix de départ ~65 €/t, volatilité annuelle ~35%, tendance haussière
#   UKA : prix de départ ~45 £/t, volatilité annuelle ~40%, plus volatile
#   Net positioning : processus mean-reverting (Ornstein-Uhlenbeck) autour de 0
#     avec des cycles lents simulant les rotations de hedge funds
# =============================================================================

import pandas as pd
import numpy as np


class MockDataGenerator:
    """
    Génère des séries temporelles synthétiques mais réalistes pour tester
    le signal sans accès Bloomberg.

    Les prix suivent un mouvement brownien géométrique (GBM) — standard en finance.
    Le positioning suit un processus d'Ornstein-Uhlenbeck (mean-reverting) avec
    des chocs ponctuels pour simuler des rotations de marché.
    """

    def __init__(self, start_date: str, end_date: str, seed: int = 42):
        """
        Args:
            start_date : "YYYY-MM-DD"
            end_date   : "YYYY-MM-DD"
            seed       : graine aléatoire pour des résultats reproductibles
        """
        self.rng   = np.random.default_rng(seed)
        self.dates = pd.date_range(start=start_date, end=end_date, freq="W-FRI")
        self.n     = len(self.dates)

    def _gbm_prices(self, s0: float, mu: float, sigma: float) -> pd.Series:
        """
        Génère une série de prix via un mouvement brownien géométrique (GBM).

        Formule discrète hebdomadaire :
            S(t+1) = S(t) × exp((μ - σ²/2)·Δt + σ·√Δt·ε)
        où ε ~ N(0,1) et Δt = 1/52 (une semaine en fraction d'année)

        Args:
            s0    : prix initial (€ ou £ par tonne)
            mu    : drift annuel (ex: 0.04 = +4%/an en tendance)
            sigma : volatilité annuelle (ex: 0.35 = 35%/an)
        """
        dt = 1 / 52
        shocks = self.rng.standard_normal(self.n - 1)
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
        prices = s0 * np.exp(np.concatenate([[0], np.cumsum(log_returns)]))
        return pd.Series(prices, index=self.dates, name="price")

    def _ou_positioning(
        self,
        mean: float,
        std: float,
        theta: float,
        n_shocks: int = 6,
    ) -> pd.Series:
        """
        Génère un net positioning via un processus d'Ornstein-Uhlenbeck (OU).

        L'OU est le modèle classique pour des données mean-reverting : les hedge
        funds reviennent tendanciellement vers une position nette neutre.

        Formule discrète :
            pos(t+1) = pos(t) + θ·(mean - pos(t)) + σ·ε
        où θ est la vitesse de retour à la moyenne (ex: 0.07 = retour lent)

        Des chocs ponctuels simulent les rotations brutales (event-driven).

        Args:
            mean     : niveau moyen de long terme du positioning
            std      : amplitude (ordre de grandeur en nombre de contrats)
            theta    : vitesse de mean-reversion (0 = aucune, 1 = instantanée)
            n_shocks : nombre de chocs ponctuels à injecter
        """
        pos = np.zeros(self.n)
        pos[0] = mean + self.rng.normal(0, std * 0.3)
        sigma_ou = std * 0.12

        for t in range(1, self.n):
            pos[t] = pos[t - 1] + theta * (mean - pos[t - 1]) + self.rng.normal(0, sigma_ou)

        # Chocs ponctuels (rotations soudaines de marché)
        for idx in self.rng.integers(52, self.n, size=n_shocks):
            pos[idx] += self.rng.normal(0, std * 0.5)

        return pd.Series(pos, index=self.dates, name="net_pos")

    def generate(self) -> dict:
        """
        Génère l'ensemble des données pour EUA et UKA.

        Paramètres calibrés sur les données historiques réelles 2019-2024 :
          EUA : ~65€/t, vol ~35%/an, positioning entre -150k et +150k contrats
          UKA : ~45£/t, vol ~40%/an, marché moins liquide donc plus volatile

        Returns:
            dict {"EUA": {...}, "UKA": {...}} avec pour chaque asset :
                "prices", "mm_long", "mm_short", "net_pos"
        """
        # (prix_initial, drift_annuel, vol_annuelle, pos_mean, pos_std, ou_theta)
        specs = {
            "EUA": (65.0, 0.04, 0.35,  20_000, 100_000, 0.07),
            "UKA": (45.0, 0.02, 0.40, -10_000,  80_000, 0.09),
        }

        data = {}
        for asset, (s0, mu, sigma, pos_mean, pos_std, theta) in specs.items():
            prices  = self._gbm_prices(s0, mu, sigma)
            net_pos = self._ou_positioning(pos_mean, pos_std, theta)

            # Décomposition du net positioning en longs et shorts cohérents
            # (toujours positifs : on part d'une base fixe et on ajoute le net)
            base   = pos_std * 1.5
            longs  = base + net_pos.clip(lower=0)
            shorts = base - net_pos.clip(upper=0)

            data[asset] = {
                "prices":   prices,
                "mm_long":  longs.rename("mm_long"),
                "mm_short": shorts.rename("mm_short"),
                "net_pos":  net_pos,
            }

        return data
