# =============================================================================
# CONFIG — COT Sensitivity Carbon Market
# =============================================================================
# Tous les paramètres du signal sont ici. C'est le seul fichier à modifier
# pour ajuster les tickers, les poids, ou les seuils de décision.
#
# Pour trouver les bons tickers COT sur Bloomberg :
#   → Tape le ticker du future (ex: MOc1 Comdty) puis COT <GO>
#   → Ou cherche "ICE COT EUA" dans la search bar Bloomberg
# =============================================================================

try:
    import blpapi
    BLPAPI_AVAILABLE = True
except ImportError:
    BLPAPI_AVAILABLE = False


CONFIG = {

    # --- Tickers Prix ---
    # MOc1 Comdty  = ICE EUA (EU Allowances) front-month futures
    # UKAc1 Comdty = ICE UKA (UK Allowances) front-month futures
    "price_tickers": {
        "EUA": "MOc1 Comdty",
        "UKA": "UKAc1 Comdty",
    },

    # --- Tickers COT (Managed Money = hedge funds & asset managers) ---
    # ⚠️  Vérifie ces tickers sur ton terminal Bloomberg avec COT <GO>
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
    "lookback_weeks":    52,    # Fenêtre glissante pour OLS et z-score
    "momentum_ma_weeks":  4,    # Moyenne mobile du COT Momentum (semaines)
    "zscore_entry":       1.5,  # Seuil z-score pour signaler un extrême (σ)

    # --- Poids des 3 signaux (doivent sommer à 1.0) ---
    "weights": {
        "ols":      0.40,   # Lien empirique direct positioning → prix
        "momentum": 0.35,   # Tendance du flux de positioning
        "zscore":   0.25,   # Contexte de positionnement extrême
    },

    # --- Seuils du signal composite ---
    "bull_threshold":  0.20,   # Score > +0.20 → BULL
    "bear_threshold": -0.20,   # Score < -0.20 → BEAR  (entre les deux → HOLD)

    # --- Connexion Bloomberg ---
    "bbg_host": "localhost",
    "bbg_port": 8194,

    # --- Période historique ---
    "start_date": "2019-01-01",   # Format YYYY-MM-DD
}
