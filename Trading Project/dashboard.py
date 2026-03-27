# =============================================================================
# DASHBOARD — Interactive Plotly visualisation
# =============================================================================
# Layout: 6 rows × 2 columns (EUA left, UKA right)
#   1. Price
#   2. Net Positioning (Managed Money)
#   3. Composite score + BULL/BEAR thresholds
#   4. Sub-signal decomposition (OLS / Momentum / Z-score)
#   5. Final signal overlaid on price (colour-coded markers)
#   6. Written interpretation table (full width)
# =============================================================================

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import CONFIG


class SignalDashboard:

    BULL_COLOR = "#26a69a"   # teal-green
    BEAR_COLOR = "#ef5350"   # red
    HOLD_COLOR = "#ffa726"   # orange

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
        Builds and opens the full dashboard in the browser.
        Each asset (EUA, UKA) has its own column of charts.
        A written interpretation table spans the full width at the bottom.
        """

        # Row 6 is a full-width table (colspan=2) — no shared x-axis needed for it.
        fig = make_subplots(
            rows=6, cols=2,
            shared_xaxes=True,
            specs=[
                [{"type": "xy"},    {"type": "xy"}],
                [{"type": "xy"},    {"type": "xy"}],
                [{"type": "xy"},    {"type": "xy"}],
                [{"type": "xy"},    {"type": "xy"}],
                [{"type": "xy"},    {"type": "xy"}],
                [{"type": "table", "colspan": 2}, None],  # interpretation row
            ],
            # Relative row heights: 5 chart rows + 1 taller table row
            row_heights=[0.14, 0.14, 0.14, 0.14, 0.14, 0.30],
            subplot_titles=[
                "EUA — Price (€/t)",                        "UKA — Price (£/t)",
                "EUA — Net Positioning (Managed Money)",     "UKA — Net Positioning (Managed Money)",
                "EUA — Composite Score",                     "UKA — Composite Score",
                "EUA — Signal Decomposition",                "UKA — Signal Decomposition",
                "EUA — Final Signal",                        "UKA — Final Signal",
                "Signal Interpretation",                     # table title (colspan cell)
            ],
            vertical_spacing=0.04,
            horizontal_spacing=0.08,
        )

        # --- Render chart rows for each asset ---
        for col_idx, (asset, prices, net_pos, results) in enumerate([
            ("EUA", prices_eua, net_pos_eua, results_eua),
            ("UKA", prices_uka, net_pos_uka, results_uka),
        ], start=1):
            self._add_price_panel(fig, prices, asset, row=1, col=col_idx)
            self._add_positioning_panel(fig, net_pos, asset, row=2, col=col_idx)
            self._add_composite_panel(fig, results, asset, row=3, col=col_idx)
            self._add_decomposition_panel(fig, results, asset, row=4, col=col_idx)
            self._add_signal_panel(fig, results, prices, asset, row=5, col=col_idx)

        # --- Interpretation table (row 6, full width) ---
        self._add_interpretation_table(fig, results_eua, results_uka, row=6, col=1)

        # --- Global layout ---
        last_date   = results_eua.index[-1].strftime("%d %b %Y")
        mock_suffix = "   ⚠️  SYNTHETIC DATA — DEMO MODE" if using_mock else ""

        fig.update_layout(
            title=dict(
                text=f"COT Sensitivity — Carbon Market Signal   |   Data as of {last_date}{mock_suffix}",
                font=dict(size=20, color="white"),
                x=0.5,
            ),
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            font=dict(color="#e0e0e0", family="Arial"),
            height=2000,
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#444", font=dict(color="white")),
            hovermode="x unified",
        )
        fig.update_xaxes(gridcolor="#2a2a4a", zeroline=False)
        fig.update_yaxes(gridcolor="#2a2a4a", zeroline=True, zerolinecolor="#444")

        fig.show()
        print("\n✅  Dashboard displayed in browser.")

    # ==========================================================================
    # Chart panels
    # ==========================================================================

    def _add_price_panel(self, fig, prices, asset, row, col):
        fig.add_trace(
            go.Scatter(x=prices.index, y=prices.values, name=f"{asset} Price",
                       line=dict(color="#4fc3f7", width=2), showlegend=True),
            row=row, col=col,
        )

    def _add_positioning_panel(self, fig, net_pos, asset, row, col):
        fig.add_trace(
            go.Scatter(x=net_pos.index, y=net_pos.values, name=f"{asset} Net Pos.",
                       fill="tozeroy", line=dict(color="#ab47bc", width=1.5),
                       fillcolor="rgba(171,71,188,0.2)"),
            row=row, col=col,
        )

    def _add_composite_panel(self, fig, results, asset, row, col):
        score   = results["composite_score"]
        bull_th = CONFIG["bull_threshold"]
        bear_th = CONFIG["bear_threshold"]

        fig.add_hrect(y0=bull_th, y1=1.0,  fillcolor="rgba(38,166,154,0.15)", line_width=0, row=row, col=col)
        fig.add_hrect(y0=-1.0, y1=bear_th, fillcolor="rgba(239,83,80,0.15)",  line_width=0, row=row, col=col)
        fig.add_trace(
            go.Scatter(x=score.index, y=score.values, name=f"{asset} Score",
                       line=dict(color="#fff176", width=2)),
            row=row, col=col,
        )
        for th, color in [(bull_th, self.BULL_COLOR), (bear_th, self.BEAR_COLOR)]:
            fig.add_hline(y=th, line_dash="dot", line_color=color, line_width=1, row=row, col=col)

    def _add_decomposition_panel(self, fig, results, asset, row, col):
        for col_name, color, label in [
            ("ols_score",      "#4db6ac", "OLS"),
            ("momentum_score", "#ffb74d", "Momentum"),
            ("zscore_score",   "#ce93d8", "Z-score"),
        ]:
            fig.add_trace(
                go.Scatter(x=results.index, y=results[col_name].values,
                           name=f"{asset} {label}", line=dict(color=color, width=1.5)),
                row=row, col=col,
            )

    def _add_signal_panel(self, fig, results, prices, asset, row, col):
        for signal_value, color in [
            ("BULL", self.BULL_COLOR),
            ("BEAR", self.BEAR_COLOR),
            ("HOLD", self.HOLD_COLOR),
        ]:
            mask            = results["signal"] == signal_value
            dates_filtered  = results.index[mask]
            prices_filtered = prices.reindex(dates_filtered)
            if len(dates_filtered) == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=dates_filtered, y=prices_filtered.values,
                    mode="markers", name=f"{asset} {signal_value}",
                    marker=dict(color=color, size=7, symbol="circle"),
                    customdata=results.loc[mask, "confidence_pct"].values,
                    hovertemplate=(
                        f"<b>{signal_value}</b><br>"
                        "Date: %{x}<br>Price: %{y:.2f}<br>"
                        "Confidence: %{customdata:.1f}%<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )

    # ==========================================================================
    # Interpretation table (row 6)
    # ==========================================================================

    def _add_interpretation_table(self, fig, results_eua, results_uka, row, col):
        """
        Full-width Plotly table with a plain-English interpretation of the
        current signal for each asset. Covers:
          - Overall signal verdict
          - OLS regression reading
          - COT Momentum reading
          - Z-score reading
          - Synthesised market view
        """
        interp_eua = self._build_interpretation(results_eua, "EUA")
        interp_uka = self._build_interpretation(results_uka, "UKA")

        # Row labels
        row_labels = [
            "Overall Signal",
            "OLS Regression  (40%)",
            "COT Momentum  (35%)",
            "Z-score  (25%)",
            "Market View",
        ]

        # Cell content per asset
        def cells(interp):
            return [
                interp["signal_str"],
                interp["ols_text"],
                interp["momentum_text"],
                interp["zscore_text"],
                interp["market_view"],
            ]

        fig.add_trace(
            go.Table(
                columnwidth=[180, 420, 420],
                header=dict(
                    values=["<b>Signal Component</b>", "<b>EUA — EU Allowances</b>", "<b>UKA — UK Allowances</b>"],
                    fill_color="#0f3460",
                    font=dict(color="white", size=13),
                    align="center",
                    height=36,
                    line_color="#2a2a4a",
                ),
                cells=dict(
                    values=[row_labels, cells(interp_eua), cells(interp_uka)],
                    fill_color=[
                        # Row label column: dark slate
                        ["#1a1a3e"] * 5,
                        # EUA column: alternate rows for readability
                        ["#16213e", "#1a1a2e", "#16213e", "#1a1a2e", "#0d1b2a"],
                        # UKA column: same alternation
                        ["#16213e", "#1a1a2e", "#16213e", "#1a1a2e", "#0d1b2a"],
                    ],
                    font=dict(color=["#a0b4c8", "#e0e0e0", "#e0e0e0"], size=12),
                    align=["center", "left", "left"],
                    height=68,
                    line_color="#2a2a4a",
                ),
            ),
            row=row, col=col,
        )

    # --------------------------------------------------------------------------
    # Interpretation logic — converts numeric scores into English sentences
    # --------------------------------------------------------------------------

    def _build_interpretation(self, results: pd.DataFrame, asset: str) -> dict:
        """
        Reads the latest row of the results DataFrame and returns a dict of
        plain-English strings for each signal component.
        """
        valid   = results.dropna(subset=["composite_score"])
        latest  = valid.iloc[-1]

        signal     = latest["signal"]
        confidence = latest["confidence_pct"]
        ols        = latest["ols_score"]
        momentum   = latest["momentum_score"]
        zscore     = latest["zscore_score"]

        return {
            "signal_str":    self._fmt_signal(signal, confidence),
            "ols_text":      self._interpret_ols(ols),
            "momentum_text": self._interpret_momentum(momentum),
            "zscore_text":   self._interpret_zscore(zscore),
            "market_view":   self._market_view(signal, confidence, asset),
        }

    @staticmethod
    def _fmt_signal(signal: str, confidence: float) -> str:
        """Formats the signal line with an icon and confidence percentage."""
        icons = {"BULL": "▲  BULL", "BEAR": "▼  BEAR", "HOLD": "◆  HOLD"}
        return f"{icons.get(signal, signal)}  —  {confidence:.1f}% confidence"

    @staticmethod
    def _interpret_ols(score: float) -> str:
        """
        Reads the OLS regression score and returns a plain-English explanation.
        The OLS score reflects how well current positioning changes predict
        next-week price returns, based on a rolling 52-week regression.
        """
        if score > 0.35:
            return (
                "Strong bullish signal from the regression model. "
                "Current speculative positioning flows have historically been "
                "a reliable predictor of price gains over the following week. "
                "The predicted return is significantly above zero."
            )
        elif score > 0.12:
            return (
                "Mild bullish signal from the regression model. "
                "Positioning changes point to modest upside over the next week, "
                "though the predictive relationship is not particularly strong."
            )
        elif score > -0.12:
            return (
                "Neutral. The rolling regression finds no meaningful link between "
                "current positioning changes and next-week price direction. "
                "The predicted return is near zero."
            )
        elif score > -0.35:
            return (
                "Mild bearish signal from the regression model. "
                "Current positioning flows point to modest price weakness "
                "over the next week based on historical patterns."
            )
        else:
            return (
                "Strong bearish signal from the regression model. "
                "Current speculative positioning flows have historically preceded "
                "price declines in the following week. "
                "The predicted return is significantly below zero."
            )

    @staticmethod
    def _interpret_momentum(score: float) -> str:
        """
        Reads the COT Momentum score and returns a plain-English explanation.
        Momentum measures whether net positioning is above or below its
        4-week moving average — i.e., is speculative interest growing?
        """
        if score > 0.35:
            return (
                "Strong bullish momentum. Net Managed Money positioning is "
                "well above its 4-week moving average and accelerating. "
                "Speculative buyers are actively building exposure."
            )
        elif score > 0.12:
            return (
                "Mild bullish momentum. Net positioning is modestly above its "
                "4-week trend, suggesting a gradual increase in speculative interest."
            )
        elif score > -0.12:
            return (
                "Neutral momentum. Net positioning is tracking near its 4-week "
                "average with no clear directional drift. "
                "The speculative community is holding steady."
            )
        elif score > -0.35:
            return (
                "Mild bearish momentum. Net positioning is modestly below its "
                "4-week trend. Speculative interest is fading at the margin."
            )
        else:
            return (
                "Strong bearish momentum. Net Managed Money positioning is "
                "significantly below its 4-week moving average and declining. "
                "Speculative selling pressure is building."
            )

    @staticmethod
    def _interpret_zscore(score: float) -> str:
        """
        Reads the Z-score signal and returns a plain-English explanation.
        The Z-score combines the normalised positioning level (vs. 52-week
        history) with the direction of the latest change — capturing whether
        a stretched position is being extended or unwound.
        """
        if score > 0.35:
            return (
                "Positioning is elevated relative to the past year and funds "
                "are actively adding. This combination of a stretched long and "
                "continued buying momentum is a bullish signal."
            )
        elif score > 0.12:
            return (
                "Positioning is slightly above historical norms with continued "
                "positive flow. Mildly supportive for prices."
            )
        elif score > -0.12:
            return (
                "Positioning sits within its normal historical range and shows "
                "no extreme crowding or short-covering dynamic. Neutral signal."
            )
        elif score > -0.35:
            return (
                "Positioning dynamics are mildly bearish — either a stretched "
                "long position is beginning to be reduced, or a growing short "
                "is being added to."
            )
        else:
            return (
                "Bearish positioning signal. A crowded long is being unwound "
                "at scale, or an extreme short position is being extended. "
                "Both dynamics typically precede price weakness."
            )

    def _market_view(self, signal: str, confidence: float, asset: str) -> str:
        """
        Synthesises all three signals into a single market view paragraph.
        Tone adjusts based on signal direction and confidence level.
        """
        if signal == "BULL":
            if confidence >= 65:
                return (
                    f"Strong bullish conviction on {asset}. The weight of positioning "
                    f"evidence — regression, momentum, and historical levels — all lean "
                    f"in favour of upside over the coming week. Managed Money appears "
                    f"to be accumulating, which has historically preceded price gains. "
                    f"Risk/reward favours long exposure at current levels."
                )
            elif confidence >= 40:
                return (
                    f"Moderate bullish bias on {asset}. The majority of positioning "
                    f"indicators tilt in favour of upside, though conviction is "
                    f"not uniform across all three models. A cautious long position "
                    f"may be warranted, with close monitoring for confirmation."
                )
            else:
                return (
                    f"Weak bullish lean on {asset}. Positioning shows a slight tilt "
                    f"to the upside but signals are thin and mixed. Not enough "
                    f"evidence to act aggressively — monitor for a stronger setup "
                    f"before adding directional exposure."
                )
        elif signal == "BEAR":
            if confidence >= 65:
                return (
                    f"Strong bearish conviction on {asset}. Positioning dynamics "
                    f"across models consistently point to downside risk. Speculative "
                    f"outflows or crowded-long unwinding could pressure prices lower "
                    f"over the next week. Risk/reward favours defensive or short positioning."
                )
            elif confidence >= 40:
                return (
                    f"Moderate bearish bias on {asset}. The balance of positioning "
                    f"evidence leans negative for prices over the next week. Consider "
                    f"reducing long exposure or initiating a light short position "
                    f"with a defined stop."
                )
            else:
                return (
                    f"Weak bearish lean on {asset}. A slight tilt to the downside "
                    f"in positioning data, but conviction is low. Avoid aggressive "
                    f"shorts — wait for more decisive positioning signals before acting."
                )
        else:  # HOLD
            return (
                f"No actionable directional signal on {asset}. Positioning models "
                f"are mixed or balanced, with bullish and bearish signals offsetting "
                f"one another. The recommended stance is flat. Monitor the weekly "
                f"COT release for a shift in speculative flows that could trigger "
                f"a clearer directional setup."
            )
