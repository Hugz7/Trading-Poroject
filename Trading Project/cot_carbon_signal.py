# =============================================================================
# COT SENSITIVITY — CARBON MARKET TRADING SIGNAL
# =============================================================================
# Objectif : Utiliser les données Commitment of Traders (COT) sur les marchés
# carbone (EUA & UKA) pour anticiper les mouvements de prix à horizon 1 semaine.
#
# Signaux calculés :
#   1. OLS Regression     — relation historique Δpositioning → return semaine suivante
#   2. COT Momentum       — tendance du net positioning vs sa moyenne mobile
#   3. Z-score            — positionnement relatif à l'historique + direction du changement
#
# Output : BULL / BEAR / HOLD + indice de confiance (0-100%) + Dashboard Plotly
#
# Dépendances à installer :
#   pip install blpapi pandas numpy scikit-learn plotly scipy
# =============================================================================


try:
    import blpapi                      # Bloomberg API (optionnel — fallback si absent)
    BLPAPI_AVAILABLE = True
except ImportError:
    BLPAPI_AVAILABLE = False

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================
# Modifie cette section pour ajuster les tickers ou les paramètres du signal.
# Pour trouver les bons tickers COT sur Bloomberg :
#   → Tape le ticker du future (ex: MOc1 Comdty) puis lance COT <GO>
#   → Ou cherche "ICE COT EUA" dans la search bar Bloomberg

CONFIG = {

    # --- Tickers Prix ---
    # MOc1 Comdty  = ICE EUA (EU Allowances) front-month futures
    # UKAc1 Comdty = ICE UKA (UK Allowances) front-month futures
    "price_tickers": {
        "EUA": "MOc1 Comdty",
        "UKA": "UKAc1 Comdty",
    },

    # --- Tickers COT (Managed Money = hedge funds & asset managers) ---
    # Ces tickers Bloomberg donnent les positions longues/courtes des
    # Managed Money traders, qui sont le segment le plus "directionnel".
    # ⚠️  Vérifie ces tickers sur ton terminal Bloomberg avec COT <GO>
    #     Les noms peuvent légèrement varier selon la version Bloomberg.
    "cot_tickers": {
        "EUA": {
            "mm_long":  "MOAOMML Index",   # EUA Managed Money Longs
            "mm_short": "MOAOMMS Index",   # EUA Managed Money Shorts
        },
        "UKA": {
            "mm_long":  "UKAOMML Index",   # UKA Managed Money Longs
            "mm_short": "UKAOMMS Index",   # UKA Managed Money Shorts
        },
    },

    # --- Paramètres des signaux ---
    "lookback_weeks":     52,    # Fenêtre glissante pour la régression OLS et le z-score
    "momentum_ma_weeks":   4,    # Moyenne mobile du COT Momentum (en semaines)
    "zscore_entry":        1.5,  # Seuil z-score pour signaler un extrême (ex: 1.5σ)

    # --- Poids des 3 signaux dans l'agrégation finale (doivent sommer à 1.0) ---
    "weights": {
        "ols":      0.40,   # Plus grande pondération : lien empirique direct prix/positioning
        "momentum": 0.35,   # Tendance du flux de positioning
        "zscore":   0.25,   # Contexte de positionnement extrême
    },

    # --- Seuils du signal composite pour la décision finale ---
    "bull_threshold":  0.20,   # Score composite > 0.20 → BULL
    "bear_threshold": -0.20,   # Score composite < -0.20 → BEAR
                               # Entre les deux → HOLD

    # --- Connexion Bloomberg ---
    "bbg_host": "localhost",
    "bbg_port": 8194,

    # --- Période historique ---
    "start_date": "2019-01-01",   # Début de l'historique (format YYYY-MM-DD)
}


# =============================================================================
# SECTION 2 — MOCK DATA GENERATOR (fallback sans Bloomberg)
# =============================================================================
# Génère des données synthétiques cohérentes avec la réalité du marché carbone.
# Utilisé automatiquement si Bloomberg est inaccessible.
#
# Hypothèses de calibration :
#   EUA : prix de départ ~65 €/t, volatilité annuelle ~35%, tendance légèrement haussière
#   UKA : prix de départ ~45 £/t, volatilité annuelle ~40%, plus volatile que l'EUA
#   Net positioning : processus mean-reverting (Ornstein-Uhlenbeck) autour de 0
#     avec des cycles lents simulant les rotations de hedge funds

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
            seed       : graine aléatoire pour la reproductibilité des résultats
        """
        self.rng   = np.random.default_rng(seed)
        self.dates = pd.date_range(start=start_date, end=end_date, freq="W-FRI")
        self.n     = len(self.dates)

    def _gbm_prices(
        self,
        s0: float,
        mu: float,
        sigma: float,
    ) -> pd.Series:
        """
        Génère une série de prix via un mouvement brownien géométrique.

        Formule discrète (hebdomadaire) :
            S(t+1) = S(t) × exp((μ - σ²/2)·Δt + σ·√Δt·ε)
        où ε ~ N(0,1) et Δt = 1/52 (une semaine)

        Args:
            s0    : prix initial
            mu    : drift annuel (tendance, ex: 0.05 = +5%/an)
            sigma : volatilité annuelle (ex: 0.35 = 35%/an)
        """
        dt = 1 / 52  # pas de temps hebdomadaire
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
        Génère un net positioning via un processus d'Ornstein-Uhlenbeck.

        L'OU est le modèle classique pour des données mean-reverting comme
        le positioning COT : les hedge funds ont tendance à revenir vers
        une position nette neutre sur le long terme.

        Formule discrète :
            pos(t+1) = pos(t) + θ·(mean - pos(t)) + σ·ε
        où θ est la vitesse de retour à la moyenne (ex: 0.08 = retour lent)

        Des chocs ponctuels sont ajoutés pour simuler des rotations brutales
        de marché (event-driven positioning).

        Args:
            mean     : niveau moyen de long terme du positioning
            std      : amplitude du positioning (en nombre de contrats, milliers)
            theta    : vitesse de mean-reversion (0=aucune, 1=instantanée)
            n_shocks : nombre de chocs ponctuels à injecter
        """
        pos = np.zeros(self.n)
        pos[0] = mean + self.rng.normal(0, std * 0.3)

        sigma_ou = std * 0.12  # bruit hebdomadaire

        for t in range(1, self.n):
            drift   = theta * (mean - pos[t - 1])
            noise   = self.rng.normal(0, sigma_ou)
            pos[t]  = pos[t - 1] + drift + noise

        # Injection de chocs ponctuels (rotations soudaines de marché)
        shock_indices = self.rng.integers(52, self.n, size=n_shocks)
        for idx in shock_indices:
            pos[idx] += self.rng.normal(0, std * 0.5)

        return pd.Series(pos, index=self.dates, name="net_pos")

    def generate(self) -> dict:
        """
        Génère l'ensemble des données pour EUA et UKA.

        Paramètres calibrés sur les données historiques réelles 2019-2024 :
          - EUA : ~65€/t de départ, vol ~35%/an, légère tendance haussière
          - UKA : ~45£/t de départ, vol ~40%/an (marché moins liquide)
          - Positioning : entre -150k et +150k contrats nets

        Returns:
            dict avec clés "EUA" et "UKA", chacun contenant
            "prices", "mm_long", "mm_short", "net_pos"
        """
        data = {}

        specs = {
            # asset : (prix_initial, drift_annuel, vol_annuelle, pos_mean, pos_std, ou_theta)
            "EUA": (65.0,  0.04, 0.35,   20_000, 100_000, 0.07),
            "UKA": (45.0,  0.02, 0.40,  -10_000,  80_000, 0.09),
        }

        for asset, (s0, mu, sigma, pos_mean, pos_std, theta) in specs.items():
            prices  = self._gbm_prices(s0, mu, sigma)
            net_pos = self._ou_positioning(pos_mean, pos_std, theta)

            # On décompose le net positioning en longs et shorts de façon cohérente :
            # longs  = base fixe + portion positive du net  (toujours > 0)
            # shorts = base fixe - portion négative du net  (toujours > 0)
            base   = pos_std * 1.5           # niveau de base des positions brutes
            longs  = base + net_pos.clip(lower=0)
            shorts = base - net_pos.clip(upper=0)

            data[asset] = {
                "prices":  prices,
                "mm_long":  longs.rename("mm_long"),
                "mm_short": shorts.rename("mm_short"),
                "net_pos":  net_pos,
            }

        return data


# =============================================================================
# SECTION 3 — BLOOMBERG DATA FETCHER
# =============================================================================

class BloombergDataFetcher:
    """
    Gère la connexion Bloomberg et le téléchargement des données historiques.
    Utilise la fonction BDH (Bloomberg Data History) pour les séries temporelles.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.session = None

    def connect(self) -> bool:
        """Ouvre une session Bloomberg. Retourne True si la connexion réussit."""
        session_options = blpapi.SessionOptions()
        session_options.setServerHost(self.host)
        session_options.setServerPort(self.port)

        self.session = blpapi.Session(session_options)

        if not self.session.start():
            print("❌  Impossible de démarrer la session Bloomberg.")
            return False

        if not self.session.openService("//blp/refdata"):
            print("❌  Impossible d'ouvrir le service Bloomberg refdata.")
            return False

        print("✅  Connexion Bloomberg établie.")
        return True

    def get_historical_data(
        self,
        tickers: list,
        fields: list,
        start_date: str,
        end_date: str,
        periodicity: str = "WEEKLY",
    ) -> pd.DataFrame:
        """
        Télécharge les données historiques Bloomberg via BDH.

        Args:
            tickers     : Liste de tickers Bloomberg (ex: ["MOc1 Comdty"])
            fields      : Liste de champs (ex: ["PX_LAST"])
            start_date  : Date de début "YYYYMMDD"
            end_date    : Date de fin   "YYYYMMDD"
            periodicity : "DAILY", "WEEKLY", "MONTHLY"

        Returns:
            DataFrame avec MultiIndex colonnes (ticker, field)
        """
        ref_service = self.session.getService("//blp/refdata")
        request = ref_service.createRequest("HistoricalDataRequest")

        # Ajout des tickers et champs à la requête
        for ticker in tickers:
            request.getElement("securities").appendValue(ticker)
        for field in fields:
            request.getElement("fields").appendValue(field)

        # Paramètres temporels
        request.set("startDate", start_date.replace("-", ""))
        request.set("endDate",   end_date.replace("-", ""))
        request.set("periodicitySelection", periodicity)

        self.session.sendRequest(request)

        # Collecte des réponses Bloomberg (peut arriver en plusieurs messages)
        all_data = {ticker: {} for ticker in tickers}

        while True:
            event = self.session.nextEvent(timeout=5000)

            for msg in event:
                if msg.hasElement("securityData"):
                    security_data = msg.getElement("securityData")
                    ticker = security_data.getElementAsString("security")
                    field_data = security_data.getElement("fieldData")

                    for i in range(field_data.numValues()):
                        row = field_data.getValueAsElement(i)
                        date = row.getElementAsDatetime("date")
                        date_str = datetime(date.year, date.month, date.day)

                        for field in fields:
                            if row.hasElement(field):
                                if field not in all_data[ticker]:
                                    all_data[ticker][field] = {}
                                all_data[ticker][field][date_str] = row.getElementAsFloat(field)

            # Bloomberg signale la fin des données avec RESPONSE (pas PARTIAL_RESPONSE)
            if event.eventType() == blpapi.Event.RESPONSE:
                break

        # Construction du DataFrame final
        dfs = []
        for ticker in tickers:
            for field in fields:
                if field in all_data[ticker]:
                    s = pd.Series(all_data[ticker][field], name=(ticker, field))
                    dfs.append(s)

        if not dfs:
            print("⚠️  Aucune donnée reçue de Bloomberg.")
            return pd.DataFrame()

        df = pd.concat(dfs, axis=1)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df

    def disconnect(self):
        """Ferme proprement la session Bloomberg."""
        if self.session:
            self.session.stop()
            print("Session Bloomberg fermée.")


# =============================================================================
# SECTION 3 — COT SIGNAL ENGINE
# =============================================================================

class COTSignalEngine:
    """
    Calcule les 3 signaux COT et les agrège en un signal composite.

    Chaque signal retourne un score entre -1.0 (très bearish) et +1.0 (très bullish).
    Le signal composite est une moyenne pondérée de ces 3 scores.
    """

    def __init__(self, config: dict):
        self.cfg = config

    # ------------------------------------------------------------------
    # Utilitaire : Calcul du Net Positioning (Longs - Shorts)
    # ------------------------------------------------------------------

    def compute_net_positioning(self, df_cot: pd.DataFrame, asset: str) -> pd.Series:
        """
        Net positioning = Managed Money Longs - Managed Money Shorts.
        Mesure l'exposition nette des hedge funds : positif = net long, négatif = net short.
        """
        long_col  = (self.cfg["cot_tickers"][asset]["mm_long"],  "PX_LAST")
        short_col = (self.cfg["cot_tickers"][asset]["mm_short"], "PX_LAST")

        net = df_cot[long_col] - df_cot[short_col]
        net.name = f"{asset}_net_positioning"
        return net

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
        Régression linéaire glissante : est-ce que le changement de positioning
        de cette semaine prédit le return de prix de la semaine prochaine ?

        Logique :
          - X = variation hebdomadaire du net positioning (Δnet_pos)
          - Y = return du prix la semaine suivante (décalé de -1 semaine)
          - On estime la régression sur une fenêtre de 52 semaines
          - La prédiction utilise le Δnet_pos de la semaine courante

        Output : score entre -1 et +1
          - Positif → le positioning actuel prédit une hausse de prix
          - Négatif → le positioning actuel prédit une baisse de prix
        """
        lookback = self.cfg["lookback_weeks"]
        delta_pos = net_pos.diff()                    # Δ net positioning semaine sur semaine
        fwd_return = prices.pct_change().shift(-1)    # Return semaine suivante (forward)

        scores = pd.Series(index=net_pos.index, dtype=float, name=f"{asset}_ols_score")

        for i in range(lookback, len(delta_pos)):
            # Fenêtre glissante de 52 semaines
            x_window = delta_pos.iloc[i - lookback : i].dropna()
            y_window = fwd_return.iloc[i - lookback : i].reindex(x_window.index).dropna()
            x_window = x_window.reindex(y_window.index)

            if len(x_window) < 20:  # Minimum 20 observations pour la régression
                continue

            # Ajustement OLS (y = α + β·x)
            X = x_window.values.reshape(-1, 1)
            Y = y_window.values
            model = LinearRegression().fit(X, Y)

            # Prédiction du return attendu avec le Δpositioning courant
            current_delta = delta_pos.iloc[i]
            if pd.isna(current_delta):
                continue

            predicted_return = model.predict([[current_delta]])[0]

            # Normalisation du score : on mappe le return prédit → [-1, +1]
            # en utilisant les quantiles 10%/90% de la distribution des returns
            return_std = y_window.std()
            if return_std > 0:
                score = np.clip(predicted_return / (2 * return_std), -1, 1)
            else:
                score = 0.0

            scores.iloc[i] = score

        return scores

    # ------------------------------------------------------------------
    # SIGNAL 2 — COT Momentum
    # ------------------------------------------------------------------

    def signal_cot_momentum(self, net_pos: pd.Series, asset: str) -> pd.Series:
        """
        Momentum du positionnement : les hedge funds augmentent-ils leurs positions
        dans la même direction que leur tendance récente ?

        Logique :
          - MA = moyenne mobile du net positioning sur N semaines (défaut: 4)
          - momentum = net_pos - MA  (positif si au-dessus de la MA → tendance haussière)
          - On normalise ce momentum par son écart-type historique

        Output : score entre -1 et +1
          - Positif → positioning en hausse par rapport à sa moyenne → BULL
          - Négatif → positioning en baisse par rapport à sa moyenne → BEAR
        """
        ma_weeks = self.cfg["momentum_ma_weeks"]
        lookback = self.cfg["lookback_weeks"]

        ma_pos = net_pos.rolling(window=ma_weeks).mean()
        momentum = net_pos - ma_pos

        # Normalisation par l'écart-type glissant
        rolling_std = momentum.rolling(window=lookback).std()
        scores = (momentum / rolling_std).clip(-1, 1)
        scores.name = f"{asset}_momentum_score"

        return scores

    # ------------------------------------------------------------------
    # SIGNAL 3 — Z-score
    # ------------------------------------------------------------------

    def signal_zscore(self, net_pos: pd.Series, asset: str) -> pd.Series:
        """
        Z-score du positionnement : où se situent les hedge funds dans leur
        fourchette historique, et dans quelle direction bougent-ils ?

        Logique :
          - z = (net_pos - mean) / std  sur une fenêtre glissante de 52 semaines
          - direction = signe du Δnet_pos (est-ce qu'ils augmentent ou réduisent ?)
          - score = z_score × direction
            → Si z très positif ET ils ajoutent (direction +) : fort BULL
            → Si z très positif ET ils réduisent (direction -) : BEAR (dénouement)
            → Si z très négatif ET ils ajoutent (direction +) : BULL (covering)

        Output : score entre -1 et +1
        """
        lookback = self.cfg["lookback_weeks"]

        rolling_mean = net_pos.rolling(window=lookback).mean()
        rolling_std  = net_pos.rolling(window=lookback).std()
        z_score      = (net_pos - rolling_mean) / rolling_std

        # Direction du changement de positioning cette semaine
        direction = np.sign(net_pos.diff())

        # Score = z normalisé × direction du flux
        raw_score = z_score * direction
        scores = raw_score.clip(-3, 3) / 3  # Normalisation en [-1, +1]
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

        Score composite = w1 * OLS + w2 * Momentum + w3 * Z-score

        Ensuite on dérive :
          - Direction : BULL / BEAR / HOLD selon les seuils configurés
          - Confiance  : |score composite| × 100 (0% = incertain, 100% = fort signal)
        """
        w = self.cfg["weights"]
        composite = (
            w["ols"]      * ols_score.fillna(0)
            + w["momentum"] * momentum_score.fillna(0)
            + w["zscore"]   * zscore_score.fillna(0)
        )
        composite.name = f"{asset}_composite_score"

        # Indice de confiance = valeur absolue du score composite (0 → 100%)
        confidence = (composite.abs() * 100).round(1)
        confidence.name = f"{asset}_confidence_pct"

        # Décision directionnelle
        bull_th = self.cfg["bull_threshold"]
        bear_th = self.cfg["bear_threshold"]

        def classify(score):
            if pd.isna(score):
                return "N/A"
            if score > bull_th:
                return "BULL"
            elif score < bear_th:
                return "BEAR"
            else:
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


# =============================================================================
# SECTION 4 — DASHBOARD PLOTLY
# =============================================================================

class SignalDashboard:
    """
    Génère un dashboard interactif Plotly avec :
      - Panneau 1 : Prix EUA et UKA
      - Panneau 2 : Net Positioning (Managed Money)
      - Panneau 3 : Score composite + seuils BULL/BEAR
      - Panneau 4 : Décomposition des 3 sous-scores
      - Panneau 5 : Signal final + confiance (tableau)
    """

    # Couleurs du dashboard
    BULL_COLOR   = "#26a69a"   # Vert-teal
    BEAR_COLOR   = "#ef5350"   # Rouge
    HOLD_COLOR   = "#ffa726"   # Orange
    NEUTRAL_COLOR = "#90a4ae"  # Gris-bleu

    def plot_dashboard(
        self,
        prices_eua: pd.Series,
        prices_uka: pd.Series,
        net_pos_eua: pd.Series,
        net_pos_uka: pd.Series,
        results_eua: pd.DataFrame,
        results_uka: pd.DataFrame,
        using_mock: bool = False,
    ):
        """
        Crée et affiche le dashboard complet dans le navigateur.
        Chaque asset (EUA, UKA) a sa propre colonne de sous-graphiques.
        """

        # Layout : 5 lignes × 2 colonnes (EUA gauche, UKA droite)
        fig = make_subplots(
            rows=5, cols=2,
            shared_xaxes=True,
            subplot_titles=[
                "EUA — Prix (€/tonne)", "UKA — Prix (£/tonne)",
                "EUA — Net Positioning (Managed Money)", "UKA — Net Positioning (Managed Money)",
                "EUA — Score Composite", "UKA — Score Composite",
                "EUA — Décomposition des Signaux", "UKA — Décomposition des Signaux",
                "EUA — Signal Final", "UKA — Signal Final",
            ],
            vertical_spacing=0.06,
            horizontal_spacing=0.08,
        )

        # Rendu de chaque asset dans sa colonne
        for col_idx, (asset, prices, net_pos, results) in enumerate([
            ("EUA", prices_eua, net_pos_eua, results_eua),
            ("UKA", prices_uka, net_pos_uka, results_uka),
        ], start=1):
            self._add_price_panel(fig, prices, asset, row=1, col=col_idx)
            self._add_positioning_panel(fig, net_pos, asset, row=2, col=col_idx)
            self._add_composite_panel(fig, results, asset, row=3, col=col_idx)
            self._add_decomposition_panel(fig, results, asset, row=4, col=col_idx)
            self._add_signal_panel(fig, results, prices, asset, row=5, col=col_idx)

        # Titre et mise en forme globale
        last_date   = results_eua.index[-1].strftime("%d %b %Y")
        mock_suffix = "   ⚠️  DONNÉES SYNTHÉTIQUES — MODE DEMO" if using_mock else ""
        fig.update_layout(
            title=dict(
                text=f"COT Sensitivity — Carbon Market Signal   |   Données au {last_date}{mock_suffix}",
                font=dict(size=20, color="white"),
                x=0.5,
            ),
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            font=dict(color="#e0e0e0", family="Arial"),
            height=1600,
            showlegend=True,
            legend=dict(
                bgcolor="rgba(0,0,0,0.4)",
                bordercolor="#444",
                font=dict(color="white"),
            ),
            hovermode="x unified",
        )

        # Axes
        fig.update_xaxes(gridcolor="#2a2a4a", zeroline=False)
        fig.update_yaxes(gridcolor="#2a2a4a", zeroline=True, zerolinecolor="#444")

        # Affichage dans le navigateur
        fig.show()
        print("\n✅  Dashboard affiché dans le navigateur.")

    # ------------------------------------------------------------------
    # Sous-panneaux privés
    # ------------------------------------------------------------------

    def _add_price_panel(self, fig, prices, asset, row, col):
        """Panneau 1 : Courbe de prix."""
        fig.add_trace(
            go.Scatter(
                x=prices.index, y=prices.values,
                name=f"{asset} Prix",
                line=dict(color="#4fc3f7", width=2),
                showlegend=True,
            ),
            row=row, col=col,
        )

    def _add_positioning_panel(self, fig, net_pos, asset, row, col):
        """Panneau 2 : Net positioning avec zones colorées long/short."""
        fig.add_trace(
            go.Scatter(
                x=net_pos.index, y=net_pos.values,
                name=f"{asset} Net Pos.",
                fill="tozeroy",
                line=dict(color="#ab47bc", width=1.5),
                fillcolor="rgba(171,71,188,0.2)",
            ),
            row=row, col=col,
        )

    def _add_composite_panel(self, fig, results, asset, row, col):
        """Panneau 3 : Score composite avec bandes BULL/BEAR."""
        score = results["composite_score"]
        bull_th = CONFIG["bull_threshold"]
        bear_th = CONFIG["bear_threshold"]

        # Bande BULL
        fig.add_hrect(y0=bull_th, y1=1.0, fillcolor="rgba(38,166,154,0.15)",
                      line_width=0, row=row, col=col)
        # Bande BEAR
        fig.add_hrect(y0=-1.0, y1=bear_th, fillcolor="rgba(239,83,80,0.15)",
                      line_width=0, row=row, col=col)

        # Courbe du score
        fig.add_trace(
            go.Scatter(
                x=score.index, y=score.values,
                name=f"{asset} Score",
                line=dict(color="#fff176", width=2),
            ),
            row=row, col=col,
        )

        # Lignes de seuil
        for th, color in [(bull_th, self.BULL_COLOR), (bear_th, self.BEAR_COLOR)]:
            fig.add_hline(y=th, line_dash="dot", line_color=color,
                          line_width=1, row=row, col=col)

    def _add_decomposition_panel(self, fig, results, asset, row, col):
        """Panneau 4 : Les 3 sous-scores superposés."""
        styles = [
            ("ols_score",      "#4db6ac", "OLS"),
            ("momentum_score", "#ffb74d", "Momentum"),
            ("zscore_score",   "#ce93d8", "Z-score"),
        ]
        for col_name, color, label in styles:
            fig.add_trace(
                go.Scatter(
                    x=results.index, y=results[col_name].values,
                    name=f"{asset} {label}",
                    line=dict(color=color, width=1.5, dash="solid"),
                ),
                row=row, col=col,
            )

    def _add_signal_panel(self, fig, results, prices, asset, row, col):
        """
        Panneau 5 : Superposition du prix avec des marqueurs colorés
        indiquant le signal de la semaine (BULL=vert, BEAR=rouge, HOLD=orange).
        """
        color_map = {"BULL": self.BULL_COLOR, "BEAR": self.BEAR_COLOR, "HOLD": self.HOLD_COLOR}

        for signal_value, color in color_map.items():
            mask = results["signal"] == signal_value
            dates_filtered = results.index[mask]
            prices_filtered = prices.reindex(dates_filtered)

            if len(dates_filtered) == 0:
                continue

            fig.add_trace(
                go.Scatter(
                    x=dates_filtered,
                    y=prices_filtered.values,
                    mode="markers",
                    name=f"{asset} {signal_value}",
                    marker=dict(color=color, size=7, symbol="circle"),
                    customdata=results.loc[mask, "confidence_pct"].values,
                    hovertemplate=(
                        f"<b>{signal_value}</b><br>"
                        "Date: %{x}<br>"
                        "Prix: %{y:.2f}<br>"
                        "Confiance: %{customdata:.1f}%<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )


# =============================================================================
# SECTION 5 — AFFICHAGE DU SIGNAL COURANT
# =============================================================================

def print_current_signal(asset: str, results: pd.DataFrame):
    """Affiche le signal le plus récent dans la console de manière lisible."""
    latest = results.dropna(subset=["signal"]).iloc[-1]
    date = results.dropna(subset=["signal"]).index[-1].strftime("%d %b %Y")

    signal     = latest["signal"]
    confidence = latest["confidence_pct"]
    composite  = latest["composite_score"]
    ols        = latest["ols_score"]
    momentum   = latest["momentum_score"]
    zscore     = latest["zscore_score"]

    icons = {"BULL": "🟢", "BEAR": "🔴", "HOLD": "🟡"}
    icon = icons.get(signal, "⚪")

    print(f"\n{'='*55}")
    print(f"  {icon}  {asset} — Signal COT au {date}")
    print(f"{'='*55}")
    print(f"  Signal     : {signal}")
    print(f"  Confiance  : {confidence:.1f}%")
    print(f"  Score comp.: {composite:+.3f}")
    print(f"  ├─ OLS     : {ols:+.3f}  (poids {CONFIG['weights']['ols']*100:.0f}%)")
    print(f"  ├─ Momentum: {momentum:+.3f}  (poids {CONFIG['weights']['momentum']*100:.0f}%)")
    print(f"  └─ Z-score : {zscore:+.3f}  (poids {CONFIG['weights']['zscore']*100:.0f}%)")
    print(f"{'='*55}\n")


# =============================================================================
# SECTION 6 — FONCTION PRINCIPALE
# =============================================================================

def _load_bloomberg_data(start_date: str, end_date: str) -> dict:
    """
    Tente de charger les données réelles via Bloomberg.
    Retourne un dict {"EUA": {...}, "UKA": {...}} au même format que MockDataGenerator,
    ou lève une exception si la connexion échoue.
    """
    bbg = BloombergDataFetcher(CONFIG["bbg_host"], CONFIG["bbg_port"])
    if not bbg.connect():
        raise ConnectionError("Connexion Bloomberg impossible.")

    price_tickers = list(CONFIG["price_tickers"].values())
    df_prices = bbg.get_historical_data(
        tickers=price_tickers,
        fields=["PX_LAST"],
        start_date=start_date,
        end_date=end_date,
        periodicity="WEEKLY",
    )

    cot_tickers_all = [
        CONFIG["cot_tickers"]["EUA"]["mm_long"],
        CONFIG["cot_tickers"]["EUA"]["mm_short"],
        CONFIG["cot_tickers"]["UKA"]["mm_long"],
        CONFIG["cot_tickers"]["UKA"]["mm_short"],
    ]
    df_cot = bbg.get_historical_data(
        tickers=cot_tickers_all,
        fields=["PX_LAST"],
        start_date=start_date,
        end_date=end_date,
        periodicity="WEEKLY",
    )
    bbg.disconnect()

    data = {}
    for asset in ("EUA", "UKA"):
        prices    = df_prices[(CONFIG["price_tickers"][asset], "PX_LAST")]
        mm_long   = df_cot[(CONFIG["cot_tickers"][asset]["mm_long"],  "PX_LAST")]
        mm_short  = df_cot[(CONFIG["cot_tickers"][asset]["mm_short"], "PX_LAST")]
        net_pos   = mm_long - mm_short

        data[asset] = {
            "prices":   prices,
            "mm_long":  mm_long,
            "mm_short": mm_short,
            "net_pos":  net_pos,
        }
    return data


def _print_mock_warning():
    """Avertissement console bien visible indiquant que les données sont synthétiques."""
    bar = "!" * 60
    print(f"\n{bar}")
    print("  ⚠️   ATTENTION — DONNÉES SYNTHÉTIQUES (MODE DEMO)")
    print(f"{bar}")
    print("  Bloomberg est inaccessible ou blpapi n'est pas installé.")
    print("  Les données affichées sont GÉNÉRÉES ALÉATOIREMENT et ne")
    print("  reflètent PAS les prix ou positions réels du marché.")
    print("  Ce mode sert uniquement à tester la logique du signal.")
    print(f"{bar}\n")


def main():
    """
    Point d'entrée du script.
    Tente Bloomberg → bascule automatiquement sur données mock si indisponible.
    """

    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = CONFIG["start_date"]
    using_mock = False

    print(f"\n📡  COT Carbon Signal — {start_date} → {end_date}")
    print("=" * 55)

    # ----------------------------------------------------------------
    # Étape 1 : Acquisition des données (Bloomberg ou Mock)
    # ----------------------------------------------------------------
    raw_data = None

    # On tente Bloomberg seulement si blpapi est installé
    if BLPAPI_AVAILABLE:
        print("\n🔌  Tentative de connexion Bloomberg...")
        try:
            raw_data = _load_bloomberg_data(start_date, end_date)
            print("✅  Données Bloomberg chargées avec succès.")
        except Exception as e:
            print(f"⚠️   Bloomberg indisponible ({e})")

    # Fallback : données synthétiques
    if raw_data is None:
        if not BLPAPI_AVAILABLE:
            print("⚠️   blpapi non installé — passage en mode démo.")
        print("🎲  Génération de données synthétiques cohérentes...")
        mock = MockDataGenerator(start_date=start_date, end_date=end_date)
        raw_data = mock.generate()
        using_mock = True

    # ----------------------------------------------------------------
    # Étape 2 : Calcul des signaux COT
    # ----------------------------------------------------------------
    print("\n⚙️   Calcul des signaux COT...")
    engine  = COTSignalEngine(CONFIG)
    results = {}

    for asset in ("EUA", "UKA"):
        prices  = raw_data[asset]["prices"]
        net_pos = raw_data[asset]["net_pos"]

        # Alignement des séries (dates communes uniquement)
        common_idx      = prices.index.intersection(net_pos.index)
        prices_aligned  = prices.reindex(common_idx)
        net_pos_aligned = net_pos.reindex(common_idx)

        # Calcul des 3 sous-signaux
        ols_score      = engine.signal_ols_regression(net_pos_aligned, prices_aligned, asset)
        momentum_score = engine.signal_cot_momentum(net_pos_aligned, asset)
        zscore_score   = engine.signal_zscore(net_pos_aligned, asset)

        # Agrégation en signal composite
        result_df = engine.aggregate_signal(ols_score, momentum_score, zscore_score, asset)

        results[asset] = {
            "df":      result_df,
            "prices":  prices_aligned,
            "net_pos": net_pos_aligned,
        }

        print_current_signal(asset, result_df)

    # Avertissement données mock APRÈS les signaux pour qu'il soit bien visible
    if using_mock:
        _print_mock_warning()

    # ----------------------------------------------------------------
    # Étape 3 : Dashboard
    # ----------------------------------------------------------------
    print("📊  Génération du dashboard...")
    dashboard = SignalDashboard()
    dashboard.plot_dashboard(
        prices_eua  = results["EUA"]["prices"],
        prices_uka  = results["UKA"]["prices"],
        net_pos_eua = results["EUA"]["net_pos"],
        net_pos_uka = results["UKA"]["net_pos"],
        results_eua = results["EUA"]["df"],
        results_uka = results["UKA"]["df"],
        using_mock  = using_mock,
    )


# =============================================================================
# LANCEMENT DU SCRIPT
# =============================================================================

if __name__ == "__main__":
    main()
