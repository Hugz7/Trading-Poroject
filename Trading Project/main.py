# =============================================================================
# MAIN — COT Sensitivity Carbon Market
# =============================================================================
# Stratégie de données :
#   PRIX    → Bloomberg BDH (MO1 Comdty, UKE1 Comdty) — automatique
#   COT     → Fichiers Excel exportés manuellement depuis Bloomberg chaque vendredi
#             (MO1 Comdty COT → Export → BDP → cot_data/cot_eua.xlsx)
#             Fallback → données mock si fichiers absents
#
# Lancement :
#   python main.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

from config import CONFIG, BLPAPI_AVAILABLE
from data.mock import MockDataGenerator
from data.cot_manual import load_cot_history, cot_files_available
from signals.engine import COTSignalEngine
from dashboard import SignalDashboard
from utils import print_current_signal, print_mock_warning

if BLPAPI_AVAILABLE:
    from bloomberg import load_bloomberg_prices


def main():

    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = CONFIG["start_date"]

    print(f"\n📡  COT Carbon Signal — {start_date} → {end_date}")
    print("=" * 55)

    # ----------------------------------------------------------------
    # Étape 1 : Prix — Bloomberg si disponible, sinon mock
    # ----------------------------------------------------------------
    bbg_prices = {}
    using_mock_prices = False

    if BLPAPI_AVAILABLE:
        print("\n🔌  Chargement des prix Bloomberg...")
        try:
            bbg_prices = load_bloomberg_prices(start_date, end_date)
            print("✅  Prix Bloomberg chargés.")
        except Exception as e:
            print(f"⚠️   Prix Bloomberg indisponibles ({e})")
            using_mock_prices = True
    else:
        using_mock_prices = True

    # ----------------------------------------------------------------
    # Étape 2 : COT — fichiers Excel si disponibles, sinon mock
    # ----------------------------------------------------------------
    cot_available = cot_files_available()
    using_mock_cot = not any(cot_available.values())

    if not using_mock_cot:
        print("\n📋  Chargement des données COT (fichiers Excel Bloomberg)...")
    else:
        print("\n⚠️   Aucun fichier COT trouvé dans cot_data/ — passage en mode mock.")
        print("     → Exporte depuis Bloomberg : MO1 Comdty COT → Export → BDP")
        print("     → Sauvegarde sous : cot_data/cot_eua.xlsx")
        print("     → Idem pour UKE1 Comdty → cot_data/cot_uka.xlsx")

    # ----------------------------------------------------------------
    # Étape 3 : Construction des données par asset
    # ----------------------------------------------------------------
    using_mock = using_mock_prices or using_mock_cot
    mock_data = MockDataGenerator(start_date=start_date, end_date=end_date).generate() if using_mock else None

    raw_data = {}
    for asset in ("EUA", "UKA"):

        # Prix
        if asset in bbg_prices:
            prices = bbg_prices[asset]
        else:
            prices = mock_data[asset]["prices"]

        # COT
        if cot_available.get(asset, False):
            try:
                cot      = load_cot_history(asset, start_date, end_date)
                mm_long  = cot["mm_long"]
                mm_short = cot["mm_short"]
                net_pos  = cot["net_pos"]
            except Exception as e:
                print(f"⚠️   Erreur lecture COT {asset} ({e}) — fallback mock.")
                mm_long  = mock_data[asset]["mm_long"]
                mm_short = mock_data[asset]["mm_short"]
                net_pos  = mock_data[asset]["net_pos"]
        else:
            mm_long  = mock_data[asset]["mm_long"]
            mm_short = mock_data[asset]["mm_short"]
            net_pos  = mock_data[asset]["net_pos"]

        raw_data[asset] = {
            "prices":   prices,
            "mm_long":  mm_long,
            "mm_short": mm_short,
            "net_pos":  net_pos,
        }

    # ----------------------------------------------------------------
    # Étape 4 : Calcul des signaux COT
    # ----------------------------------------------------------------
    print("\n⚙️   Calcul des signaux COT...")
    engine  = COTSignalEngine(CONFIG)
    results = {}

    for asset in ("EUA", "UKA"):
        prices  = raw_data[asset]["prices"]
        net_pos = raw_data[asset]["net_pos"]

        common_idx      = prices.index.intersection(net_pos.index)
        prices_aligned  = prices.reindex(common_idx)
        net_pos_aligned = net_pos.reindex(common_idx)

        ols_score      = engine.signal_ols_regression(net_pos_aligned, prices_aligned, asset)
        momentum_score = engine.signal_cot_momentum(net_pos_aligned, asset)
        zscore_score   = engine.signal_zscore(net_pos_aligned, asset)
        result_df      = engine.aggregate_signal(ols_score, momentum_score, zscore_score, asset)

        results[asset] = {
            "df":      result_df,
            "prices":  prices_aligned,
            "net_pos": net_pos_aligned,
        }

        print_current_signal(asset, result_df)

    if using_mock:
        print_mock_warning()

    # ----------------------------------------------------------------
    # Étape 5 : Dashboard
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