# =============================================================================
# CONFIG — COT Sensitivity Carbon Market
# =============================================================================

try:
    import blpapi
    BLPAPI_AVAILABLE = True
except ImportError:
    BLPAPI_AVAILABLE = False


CONFIG = {

    # --- Tickers Prix Bloomberg ---
    # MO1 Comdty  = EUA Generic 1st Future (ICE, source ICE Index)
    # UKE1 Comdty = UKA Generic 1st Future (ICE)
    "price_tickers": {
        "EUA": "MO1 Comdty",
        "UKA": "UKE1 Comdty",
    },

    # --- COT : géré via fichiers Excel dans cot_data/ ---
    # Exporte chaque vendredi depuis Bloomberg :
    #   MO1 Comdty  COT → Export → BDP → sauvegarder sous cot_data/cot_eua.xlsx
    #   UKE1 Comdty COT → Export → BDP → sauvegarder sous cot_data/cot_uka.xlsx
    # Pour accumuler l'historique, renomme les fichiers :
    #   cot_eua_20260327.xlsx, cot_eua_20260320.xlsx, etc.

    # --- Paramètres des signaux ---
    "lookback_weeks":    52,
    "momentum_ma_weeks":  4,
    "zscore_entry":       1.5,

    # --- Poids des 3 signaux (doivent sommer à 1.0) ---
    "weights": {
        "ols":      0.40,
        "momentum": 0.35,
        "zscore":   0.25,
    },

    # --- Seuils du signal composite ---
    "bull_threshold":  0.20,
    "bear_threshold": -0.20,

    # --- Connexion Bloomberg ---
    "bbg_host": "localhost",
    "bbg_port": 8194,

    # --- Période historique ---
    "start_date": "2019-01-01",
}