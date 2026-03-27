# =============================================================================
# UTILS — Affichage console
# =============================================================================

import pandas as pd
from config import CONFIG


def print_current_signal(asset: str, results: pd.DataFrame):
    """
    Affiche le signal le plus récent dans la console.
    Montre le signal, la confiance, le score composite, et la contribution
    de chacun des 3 sous-signaux.
    """
    valid   = results.dropna(subset=["signal"])
    latest  = valid.iloc[-1]
    date    = valid.index[-1].strftime("%d %b %Y")

    signal     = latest["signal"]
    confidence = latest["confidence_pct"]
    composite  = latest["composite_score"]
    ols        = latest["ols_score"]
    momentum   = latest["momentum_score"]
    zscore     = latest["zscore_score"]

    icons = {"BULL": "🟢", "BEAR": "🔴", "HOLD": "🟡"}
    icon  = icons.get(signal, "⚪")
    w     = CONFIG["weights"]

    print(f"\n{'='*55}")
    print(f"  {icon}  {asset} — Signal COT au {date}")
    print(f"{'='*55}")
    print(f"  Signal     : {signal}")
    print(f"  Confiance  : {confidence:.1f}%")
    print(f"  Score comp.: {composite:+.3f}")
    print(f"  ├─ OLS     : {ols:+.3f}  (poids {w['ols']*100:.0f}%)")
    print(f"  ├─ Momentum: {momentum:+.3f}  (poids {w['momentum']*100:.0f}%)")
    print(f"  └─ Z-score : {zscore:+.3f}  (poids {w['zscore']*100:.0f}%)")
    print(f"{'='*55}\n")


def print_mock_warning():
    """
    Avertissement bien visible dans la console indiquant que les données
    affichées sont synthétiques et ne reflètent pas le marché réel.
    """
    bar = "!" * 60
    print(f"\n{bar}")
    print("  ⚠️   ATTENTION — DONNÉES SYNTHÉTIQUES (MODE DEMO)")
    print(f"{bar}")
    print("  Bloomberg est inaccessible ou blpapi n'est pas installé.")
    print("  Les données affichées sont GÉNÉRÉES ALÉATOIREMENT et ne")
    print("  reflètent PAS les prix ou positions réels du marché.")
    print("  Ce mode sert uniquement à tester la logique du signal.")
    print(f"{bar}\n")
