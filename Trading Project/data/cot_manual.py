# =============================================================================
# COT MANUAL READER — Lecture des exports Bloomberg BDP
# =============================================================================
# Lit les fichiers Excel exportés manuellement depuis Bloomberg :
#   MO1 Comdty COT → Export → BDP → sauvegarder sous "cot_eua.xlsx"
#   UKE1 Comdty COT → Export → BDP → sauvegarder sous "cot_uka.xlsx"
#
# Ces fichiers doivent être placés dans le dossier "cot_data/" du projet.
# À mettre à jour chaque vendredi après la publication des données COT.
#
# Structure attendue du fichier Excel (export BDP Bloomberg) :
#   Col 0 : Catégorie (Investment Firms, Investment Funds, etc.)
#   Col 1-3 : Risk Reducing (Position, Chg, % OI)
#   Col 4-6 : Other Activities (Position, Chg, % OI)  ← on utilise ça
#   Col 7-9 : Total (Position, Chg, % OI)
#
# On utilise "Investment Firms or Credit Institutions" — Other Activities
# car c'est le segment le plus purement spéculatif/directionnel.
# =============================================================================

import os
import re
from datetime import datetime, date

import openpyxl
import pandas as pd
import numpy as np


# Dossier où l'utilisateur dépose les fichiers COT exportés de Bloomberg
COT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "cot_data")

# Noms de fichiers attendus
COT_FILES = {
    "EUA": "cot_eua.xlsx",
    "UKA": "cot_uka.xlsx",
}

# Catégorie trader à utiliser comme proxy "Managed Money"
TRADER_CATEGORY = "Investment Firms or Credit Institutions"

# Colonnes dans le fichier BDP :
# Index 4 = Other Activities Long/Short (le plus directionnel)
# Index 7 = Total Long/Short
COL_LONG  = 4   # Other Activities — Position (Long row)
COL_SHORT = 4   # Other Activities — Position (Short row)


def _parse_value(v):
    """Convertit une valeur Bloomberg en float. Retourne NaN si invalide."""
    if v is None:
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and v.startswith('#'):
        return np.nan
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def read_cot_snapshot(filepath: str) -> dict:
    """
    Lit un fichier Excel BDP Bloomberg et extrait les positions
    Long et Short de la catégorie Investment Firms (Other Activities).

    Retourne un dict avec :
        "mm_long"  : float — positions longues Investment Firms
        "mm_short" : float — positions courtes Investment Firms
        "net_pos"  : float — net positioning (long - short)
        "date"     : date — date du fichier (date de modification)
    """
    wb = openpyxl.load_workbook(filepath, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))

    mm_long  = np.nan
    mm_short = np.nan
    current_category = None

    for row in rows:
        label = row[0]

        # Détection de la catégorie
        if isinstance(label, str) and label.strip():
            if label.strip() not in ("Long", "Short", "Net"):
                current_category = label.strip()
                continue

        # Extraction des positions Investment Firms
        if current_category and TRADER_CATEGORY.lower() in current_category.lower():
            if isinstance(label, str):
                if label.strip() == "Long":
                    mm_long = _parse_value(row[COL_LONG])
                elif label.strip() == "Short":
                    mm_short = _parse_value(row[COL_SHORT])

    file_date = date.fromtimestamp(os.path.getmtime(filepath))

    return {
        "mm_long":  mm_long,
        "mm_short": mm_short,
        "net_pos":  mm_long - mm_short if not (np.isnan(mm_long) or np.isnan(mm_short)) else np.nan,
        "date":     file_date,
    }


def _extract_date_from_filename(filename: str):
    """
    Tente d'extraire une date depuis le nom de fichier.
    Formats supportés : cot_eua_20260327.xlsx, cot_eua_2026-03-27.xlsx
    Retourne None si pas de date trouvée.
    """
    match = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})', filename)
    if match:
        s = match.group(1).replace('-', '').replace('_', '')
        try:
            return datetime.strptime(s, '%Y%m%d').date()
        except ValueError:
            pass
    return None


def load_cot_history(asset: str, start_date: str, end_date: str) -> dict:
    """
    Charge l'historique COT pour un asset depuis le dossier cot_data/.

    Cherche tous les fichiers correspondant à l'asset dans cot_data/ :
        cot_eua.xlsx          → snapshot le plus récent
        cot_eua_20260327.xlsx → snapshot daté

    Construit une série temporelle hebdomadaire en utilisant les snapshots
    disponibles et en interpolant entre les points connus.

    Retourne un dict avec "prices" (None ici), "mm_long", "mm_short", "net_pos"
    sous forme de pd.Series indexées par date.

    Si aucun fichier n'est trouvé, lève FileNotFoundError.
    """
    if not os.path.exists(COT_DATA_DIR):
        raise FileNotFoundError(
            f"Dossier '{COT_DATA_DIR}' introuvable. "
            f"Crée le dossier et dépose-y les fichiers COT exportés de Bloomberg."
        )

    prefix = f"cot_{asset.lower()}"
    snapshots = {}

    # Parcours de tous les fichiers COT de l'asset
    for fname in sorted(os.listdir(COT_DATA_DIR)):
        if not fname.lower().startswith(prefix):
            continue
        if not fname.lower().endswith('.xlsx'):
            continue

        fpath = os.path.join(COT_DATA_DIR, fname)
        snapshot = read_cot_snapshot(fpath)

        # Date du snapshot : depuis le nom de fichier ou date de modification
        snap_date = _extract_date_from_filename(fname) or snapshot["date"]

        # Aligne sur le vendredi le plus proche (les COT sont publiés le vendredi)
        snap_date = _nearest_friday(snap_date)
        snapshots[snap_date] = snapshot

    if not snapshots:
        raise FileNotFoundError(
            f"Aucun fichier COT trouvé pour {asset} dans '{COT_DATA_DIR}'.\n"
            f"Attendu : '{prefix}.xlsx' ou '{prefix}_YYYYMMDD.xlsx'.\n"
            f"Exporte depuis Bloomberg : {asset} COT → Export → BDP."
        )

    # Construction de l'index hebdomadaire
    idx = pd.date_range(start=start_date, end=end_date, freq="W-FRI")

    # Série de net positioning à partir des snapshots connus
    net_series   = pd.Series(dtype=float, index=idx)
    long_series  = pd.Series(dtype=float, index=idx)
    short_series = pd.Series(dtype=float, index=idx)

    for snap_date, snap in snapshots.items():
        ts = pd.Timestamp(snap_date)
        if ts in net_series.index:
            net_series[ts]   = snap["net_pos"]
            long_series[ts]  = snap["mm_long"]
            short_series[ts] = snap["mm_short"]

    # Interpolation linéaire entre les snapshots connus
    net_series   = net_series.interpolate(method="time", limit_direction="both")
    long_series  = long_series.interpolate(method="time", limit_direction="both")
    short_series = short_series.interpolate(method="time", limit_direction="both")

    n_snapshots = len(snapshots)
    print(f"  📋  {asset} COT : {n_snapshots} snapshot(s) chargé(s) depuis '{COT_DATA_DIR}'")
    if n_snapshots == 1:
        print(f"  ⚠️   Un seul snapshot disponible — les signaux OLS et Z-score")
        print(f"       seront moins fiables. Accumule des exports hebdomadaires")
        print(f"       pour améliorer la précision du signal.")

    return {
        "mm_long":  long_series.rename("mm_long"),
        "mm_short": short_series.rename("mm_short"),
        "net_pos":  net_series.rename("net_pos"),
    }


def _nearest_friday(d: date) -> date:
    """Retourne le vendredi le plus proche d'une date donnée."""
    weekday = d.weekday()  # 0=lundi, 4=vendredi
    if weekday <= 4:
        return d + pd.Timedelta(days=4 - weekday)
    else:  # samedi ou dimanche → vendredi suivant
        return d + pd.Timedelta(days=4 - weekday + 7)


def cot_files_available() -> dict:
    """
    Vérifie quels fichiers COT sont disponibles dans cot_data/.
    Retourne un dict {"EUA": bool, "UKA": bool}.
    """
    available = {}
    for asset in ("EUA", "UKA"):
        prefix = f"cot_{asset.lower()}"
        found = False
        if os.path.exists(COT_DATA_DIR):
            for fname in os.listdir(COT_DATA_DIR):
                if fname.lower().startswith(prefix) and fname.lower().endswith('.xlsx'):
                    found = True
                    break
        available[asset] = found
    return available
