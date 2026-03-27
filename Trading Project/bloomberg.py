# =============================================================================
# BLOOMBERG DATA FETCHER — Prix uniquement via blpapi BDH
# =============================================================================
# Les données COT MiFID ne sont pas accessibles via BDH (uniquement via BQL).
# Ce module se charge uniquement des prix hebdomadaires EUA et UKA.
#
# Tickers prix :
#   EUA : MO1 Comdty  (EU Carbon Allowance — ICE, source ICE Index)
#   UKA : UKE1 Comdty (UK Carbon Allowance — ICE)
# =============================================================================

import pandas as pd
from datetime import datetime

from config import CONFIG, BLPAPI_AVAILABLE

if BLPAPI_AVAILABLE:
    import blpapi

PRICE_TICKERS = {
    "EUA": "MO1 Comdty",
    "UKA": "UKE1 Comdty",
}


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

    def get_historical_data(self, tickers, fields, start_date, end_date, periodicity="WEEKLY"):
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
                        d    = row.getElementAsDatetime("date")
                        dt   = datetime(d.year, d.month, d.day)
                        for field in fields:
                            if row.hasElement(field):
                                all_data[ticker][field][dt] = row.getElementAsFloat(field)
            if event.eventType() == blpapi.Event.RESPONSE:
                break

        dfs = []
        for ticker in tickers:
            for field in fields:
                if all_data[ticker][field]:
                    dfs.append(pd.Series(all_data[ticker][field], name=(ticker, field)))

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, axis=1)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def disconnect(self):
        if self.session:
            self.session.stop()


def load_bloomberg_prices(start_date: str, end_date: str) -> dict:
    """
    Charge les prix hebdomadaires EUA et UKA depuis Bloomberg BDH.
    Retourne {"EUA": pd.Series, "UKA": pd.Series}.
    """
    bbg = BloombergDataFetcher()
    if not bbg.connect():
        raise ConnectionError("Connexion Bloomberg impossible.")

    prices = {}
    for asset, ticker in PRICE_TICKERS.items():
        df = bbg.get_historical_data([ticker], ["PX_LAST"], start_date, end_date)
        if df.empty or (ticker, "PX_LAST") not in df.columns:
            bbg.disconnect()
            raise ValueError(f"Aucun prix reçu pour {asset} ({ticker}).")
        s = df[(ticker, "PX_LAST")].dropna()
        s.index = pd.to_datetime(s.index)
        prices[asset] = s
        print(f"  ✅  {asset} prix : {len(s)} semaines ({s.index[0].date()} → {s.index[-1].date()})")

    bbg.disconnect()
    return prices