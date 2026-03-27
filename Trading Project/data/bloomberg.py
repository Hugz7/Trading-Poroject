# =============================================================================
# BLOOMBERG DATA FETCHER
# =============================================================================
# Gère la connexion Bloomberg (blpapi) et le téléchargement des données
# historiques via BDH (Bloomberg Data History).
# =============================================================================

import pandas as pd
from datetime import datetime

# blpapi est importé conditionnellement depuis config pour éviter les erreurs
# si la librairie n'est pas installée.
from config import CONFIG, BLPAPI_AVAILABLE

if BLPAPI_AVAILABLE:
    import blpapi


class BloombergDataFetcher:
    """
    Gère la connexion Bloomberg et le téléchargement des données historiques.
    Utilise la fonction BDH (Bloomberg Data History) pour les séries temporelles.
    """

    def __init__(self):
        self.session = None

    def connect(self) -> bool:
        """Ouvre une session Bloomberg. Retourne True si la connexion réussit."""
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

        Args:
            tickers     : Liste de tickers Bloomberg (ex: ["MOc1 Comdty"])
            fields      : Liste de champs (ex: ["PX_LAST"])
            start_date  : Date de début "YYYY-MM-DD"
            end_date    : Date de fin   "YYYY-MM-DD"
            periodicity : "DAILY", "WEEKLY", "MONTHLY"

        Returns:
            DataFrame avec MultiIndex colonnes (ticker, field)
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


def load_bloomberg_data(start_date: str, end_date: str) -> dict:
    """
    Charge les données Bloomberg pour EUA et UKA (prix + COT).

    Retourne un dict {"EUA": {...}, "UKA": {...}} avec pour chaque asset :
        "prices"   : pd.Series hebdomadaire des prix
        "mm_long"  : pd.Series des positions longues Managed Money
        "mm_short" : pd.Series des positions courtes Managed Money
        "net_pos"  : pd.Series du net positioning (longs - shorts)

    Lève une ConnectionError si Bloomberg est inaccessible.
    """
    bbg = BloombergDataFetcher()
    if not bbg.connect():
        raise ConnectionError("Connexion Bloomberg impossible.")

    # Prix
    price_tickers = list(CONFIG["price_tickers"].values())
    df_prices = bbg.get_historical_data(
        tickers=price_tickers,
        fields=["PX_LAST"],
        start_date=start_date,
        end_date=end_date,
    )

    # COT
    cot_tickers = [
        CONFIG["cot_tickers"]["EUA"]["mm_long"],
        CONFIG["cot_tickers"]["EUA"]["mm_short"],
        CONFIG["cot_tickers"]["UKA"]["mm_long"],
        CONFIG["cot_tickers"]["UKA"]["mm_short"],
    ]
    df_cot = bbg.get_historical_data(
        tickers=cot_tickers,
        fields=["PX_LAST"],
        start_date=start_date,
        end_date=end_date,
    )
    bbg.disconnect()

    data = {}
    for asset in ("EUA", "UKA"):
        prices   = df_prices[(CONFIG["price_tickers"][asset], "PX_LAST")]
        mm_long  = df_cot[(CONFIG["cot_tickers"][asset]["mm_long"],  "PX_LAST")]
        mm_short = df_cot[(CONFIG["cot_tickers"][asset]["mm_short"], "PX_LAST")]

        data[asset] = {
            "prices":   prices,
            "mm_long":  mm_long,
            "mm_short": mm_short,
            "net_pos":  mm_long - mm_short,
        }

    return data
