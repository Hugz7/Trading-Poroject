# =============================================================================
# BLOOMBERG DATA FETCHER — COT MiFID via blpapi
# =============================================================================
# Utilise blpapi (BDH) avec les champs COT MiFID pour récupérer les positions
# Investment Firms (proxy Managed Money) sur EUA et UKA.
#
# Champs COT MiFID Bloomberg :
#   COT_MIFID_IF_LONG  = Investment Firms Long positions
#   COT_MIFID_IF_SHORT = Investment Firms Short positions
#
# Si ces champs échouent, on tente un fallback avec d'autres champs COT
# disponibles sur le terminal (COT_NET_POSITION, etc.)
# =============================================================================

import pandas as pd
from datetime import datetime

from config import CONFIG, BLPAPI_AVAILABLE

if BLPAPI_AVAILABLE:
    import blpapi


class BloombergDataFetcher:

    def __init__(self):
        self.session = None

    def connect(self) -> bool:
        session_options = blpapi.SessionOptions()
        session_options.setServerHost(CONFIG["bbg_host"])
        session_options.setServerPort(CONFIG["bbg_port"])
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
        Retourne un DataFrame avec MultiIndex colonnes (ticker, field).
        """
        ref_service = self.session.getService("//blp/refdata")
        request = ref_service.createRequest("HistoricalDataRequest")

        for ticker in tickers:
            request.getElement("securities").appendValue(ticker)
        for field in fields:
            request.getElement("fields").appendValue(field)

        request.set("startDate", start_date.replace("-", ""))
        request.set("endDate",   end_date.replace("-", ""))
        request.set("periodicitySelection", periodicity)

        self.session.sendRequest(request)

        all_data = {ticker: {field: {} for field in fields} for ticker in tickers}

        while True:
            event = self.session.nextEvent(timeout=10000)

            for msg in event:
                if msg.hasElement("securityData"):
                    sec_data   = msg.getElement("securityData")
                    ticker     = sec_data.getElementAsString("security")
                    field_data = sec_data.getElement("fieldData")

                    for i in range(field_data.numValues()):
                        row  = field_data.getValueAsElement(i)
                        date = row.getElementAsDatetime("date")
                        dt   = datetime(date.year, date.month, date.day)

                        for field in fields:
                            if row.hasElement(field):
                                all_data[ticker][field][dt] = row.getElementAsFloat(field)

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        dfs = []
        for ticker in tickers:
            for field in fields:
                if all_data[ticker][field]:
                    s = pd.Series(all_data[ticker][field], name=(ticker, field))
                    dfs.append(s)

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, axis=1)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df

    def disconnect(self):
        if self.session:
            self.session.stop()
            print("Session Bloomberg fermée.")


def load_bloomberg_data(start_date: str, end_date: str) -> dict:
    """
    Charge les données Bloomberg pour EUA et UKA :
      - Prix hebdomadaires (PX_LAST)
      - Positions COT MiFID Investment Firms (Long + Short)

    Retourne un dict {"EUA": {...}, "UKA": {...}}.
    Lève une exception si la connexion échoue ou si les données sont vides.
    """
    bbg = BloombergDataFetcher()
    if not bbg.connect():
        raise ConnectionError("Connexion Bloomberg impossible.")

    data = {}

    for asset in ("EUA", "UKA"):
        cot_cfg = CONFIG["cot_tickers"][asset]
        ticker  = cot_cfg["ticker"]
        f_long  = cot_cfg["mm_long"]
        f_short = cot_cfg["mm_short"]

        print(f"  📥  Chargement {asset} ({ticker})...")

        # --- Prix ---
        df_price = bbg.get_historical_data(
            tickers=[ticker],
            fields=["PX_LAST"],
            start_date=start_date,
            end_date=end_date,
            periodicity="WEEKLY",
        )

        if df_price.empty or (ticker, "PX_LAST") not in df_price.columns:
            bbg.disconnect()
            raise ValueError(
                f"Aucune donnée de prix reçue pour {asset} ({ticker}). "
                f"Vérifie le ticker dans config.py."
            )

        prices = df_price[(ticker, "PX_LAST")].dropna()

        # --- COT MiFID ---
        df_cot = bbg.get_historical_data(
            tickers=[ticker],
            fields=[f_long, f_short],
            start_date=start_date,
            end_date=end_date,
            periodicity="WEEKLY",
        )

        if df_cot.empty:
            bbg.disconnect()
            raise ValueError(
                f"Aucune donnée COT reçue pour {asset} ({ticker}). \n"
                f"Champs tentés : {f_long}, {f_short}. \n"
                f"Vérifie les champs COT disponibles sur le terminal : "
                f"{ticker} COT <GO> → Open in BQL."
            )

        mm_long  = df_cot[(ticker, f_long)].dropna()
        mm_short = df_cot[(ticker, f_short)].dropna()

        # --- Alignement ---
        common_idx = prices.index.intersection(mm_long.index).intersection(mm_short.index)
        if len(common_idx) == 0:
            bbg.disconnect()
            raise ValueError(f"Aucune date commune prix/COT pour {asset}.")

        prices   = prices.reindex(common_idx)
        mm_long  = mm_long.reindex(common_idx)
        mm_short = mm_short.reindex(common_idx)

        data[asset] = {
            "prices":   prices,
            "mm_long":  mm_long,
            "mm_short": mm_short,
            "net_pos":  mm_long - mm_short,
        }

        print(f"  ✅  {asset} : {len(common_idx)} semaines "
              f"({common_idx[0].date()} → {common_idx[-1].date()})")

    bbg.disconnect()
    return data