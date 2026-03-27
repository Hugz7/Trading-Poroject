# =============================================================================
# MAIN — COT Sensitivity Carbon Market
# =============================================================================
# Point d'entrée du signal. Lance :
#   1. Acquisition des données (Bloomberg ou données mock si inaccessible)
#   2. Calcul des 3 signaux COT + agrégation
#   3. Affichage console du signal courant
#   4. Dashboard Plotly interactif
#
# Lancement :
#   python main.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

from config import CONFIG, BLPAPI_AVAILABLE
from data.mock import MockDataGenerator
from signals.engine import COTSignalEngine
from dashboard import SignalDashboard
from utils import print_current_signal, print_mock_warning

if BLPAPI_AVAILABLE:
    from data.bloomberg import load_bloomberg_data


def main():

    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = CONFIG["start_date"]
    using_mock = False

    print(f"\n📡  COT Carbon Signal — {start_date} → {end_date}")
    print("=" * 55)

    # ----------------------------------------------------------------
    # Étape 1 : Acquisition des données (Bloomberg ou Mock)
    # ----------------------------------------------------------------
    raw_data = None

    if BLPAPI_AVAILABLE:
        print("\n🔌  Tentative de connexion Bloomberg...")
        try:
            raw_data = load_bloomberg_data(start_date, end_date)
            print("✅  Données Bloomberg chargées avec succès.")
        except Exception as e:
            print(f"⚠️   Bloomberg indisponible ({e})")

    if raw_data is None:
        if not BLPAPI_AVAILABLE:
            print("⚠️   blpapi non installé — passage en mode démo.")
        print("🎲  Génération de données synthétiques cohérentes...")
        raw_data   = MockDataGenerator(start_date=start_date, end_date=end_date).generate()
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

        # Alignement sur les dates communes
        common_idx      = prices.index.intersection(net_pos.index)
        prices_aligned  = prices.reindex(common_idx)
        net_pos_aligned = net_pos.reindex(common_idx)

        ols_score      = engine.signal_ols_regression(net_pos_aligned, prices_aligned, asset)
        momentum_score = engine.signal_cot_momentum(net_pos_aligned, asset)
        zscore_score   = engine.signal_zscore(net_pos_aligned, asset)

        result_df = engine.aggregate_signal(ols_score, momentum_score, zscore_score, asset)

        results[asset] = {
            "df":      result_df,
            "prices":  prices_aligned,
            "net_pos": net_pos_aligned,
        }

        print_current_signal(asset, result_df)

    # Avertissement mock affiché après les signaux pour rester visible
    if using_mock:
        print_mock_warning()

    # ----------------------------------------------------------------
    # Étape 3 : Dashboard
    # ----------------------------------------------------------------
    print("📊  Génération du dashboard...")
    SignalDashboard().plot_dashboard(
        prices_eua  = results["EUA"]["prices"],
        prices_uka  = results["UKA"]["prices"],
        net_pos_eua = results["EUA"]["net_pos"],
        net_pos_uka = results["UKA"]["net_pos"],
        results_eua = results["EUA"]["df"],
        results_uka = results["UKA"]["df"],
        using_mock  = using_mock,
    )


if __name__ == "__main__":
    main()
