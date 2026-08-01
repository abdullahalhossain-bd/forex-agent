# visualization/chart.py
# ============================================================
# Day 6 — Chart Visualization Engine
# AI Trader-এর Visual Interface — Plotly Interactive Chart
# ============================================================

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ChartEngine:
    """
    AI Trader-এর visualization engine।

    TradingView-এর মতো interactive chart:
    - Candlestick price chart
    - MA overlay (SMA 20, 50, 200)
    - Support & Resistance lines
    - Pattern annotations
    - RSI subplot
    - MACD subplot
    - Volume subplot
    """

    def __init__(self, symbol="EUR/USDT", timeframe="15m"):
        self.symbol    = symbol
        self.timeframe = timeframe

    # ─────────────────────────────────────────────
    # MAIN CHART — সব একসাথে
    # ─────────────────────────────────────────────

    def create_full_chart(
        self,
        df,
        support_zones    = None,
        resistance_zones = None,
        patterns_df      = None,
        show             = True,
        save_html        = None,
    ):
        """
        Full professional trading chart তৈরি করো।

        Parameters:
            df               : OHLCV + indicator DataFrame
            support_zones    : list of {'center': price, 'touches': n}
            resistance_zones : list of {'center': price, 'touches': n}
            patterns_df      : DataFrame with 'pattern', 'engulfing', 'star_pattern' columns
            show             : browser-এ দেখাবে কিনা
            save_html        : HTML file-এ save করবে (path দিলে)
        """
        support_zones    = support_zones    or []
        resistance_zones = resistance_zones or []

        # 4 row subplot: Price | RSI | MACD | Volume
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.55, 0.15, 0.15, 0.15],
            subplot_titles=(
                f"{self.symbol} — {self.timeframe}",
                "RSI (14)",
                "MACD",
                "Volume",
            ),
        )

        # ── Row 1: Candlestick ──
        self._add_candlesticks(fig, df)
        self._add_moving_averages(fig, df)
        self._add_sr_levels(fig, df, support_zones, resistance_zones)

        if patterns_df is not None:
            self._add_pattern_annotations(fig, patterns_df)

        # ── Row 2: RSI ──
        self._add_rsi(fig, df)

        # ── Row 3: MACD ──
        self._add_macd(fig, df)

        # ── Row 4: Volume ──
        self._add_volume(fig, df)

        # ── Layout ──
        self._apply_layout(fig)

        if save_html:
            fig.write_html(save_html)
            print(f"💾 Chart saved: {save_html}")

        if show:
            fig.show()
            print("📊 Chart opened in browser")

        return fig

    # ─────────────────────────────────────────────
    # CHART COMPONENTS
    # ─────────────────────────────────────────────

    def _add_candlesticks(self, fig, df):
        """Candlestick chart — price action"""
        fig.add_trace(
            go.Candlestick(
                x     = df.index,
                open  = df['open'],
                high  = df['high'],
                low   = df['low'],
                close = df['close'],
                name  = self.symbol,
                increasing_line_color = '#26a69a',   # green
                decreasing_line_color = '#ef5350',   # red
            ),
            row=1, col=1,
        )

    def _add_moving_averages(self, fig, df):
        """MA lines — trend direction"""
        ma_config = [
            ('sma_20',  '#f39c12', '1px', 'SMA 20'),
            ('sma_50',  '#3498db', '1px', 'SMA 50'),
            ('sma_200', '#9b59b6', '2px', 'SMA 200'),
            ('ema_9',   '#e74c3c', '1px', 'EMA 9'),
        ]
        for col, color, width, name in ma_config:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x    = df.index,
                        y    = df[col],
                        name = name,
                        line = dict(color=color, width=1),
                        opacity = 0.8,
                    ),
                    row=1, col=1,
                )

    def _add_sr_levels(self, fig, df, support_zones, resistance_zones):
        """Support & Resistance horizontal lines"""
        # Support — সবুজ
        for zone in support_zones[:5]:   # top 5 strongest
            strength = min(zone.get('touches', 1), 5)
            fig.add_hline(
                y                = zone['center'],
                line_color       = 'rgba(38, 166, 154, 0.6)',
                line_width       = 1 + strength * 0.3,
                line_dash        = 'dash',
                annotation_text  = f"S {zone['center']:.5f} ({'★'*strength})",
                annotation_font_color = '#26a69a',
                row=1, col=1,
            )

        # Resistance — লাল
        for zone in resistance_zones[:5]:
            strength = min(zone.get('touches', 1), 5)
            fig.add_hline(
                y                = zone['center'],
                line_color       = 'rgba(239, 83, 80, 0.6)',
                line_width       = 1 + strength * 0.3,
                line_dash        = 'dash',
                annotation_text  = f"R {zone['center']:.5f} ({'★'*strength})",
                annotation_font_color = '#ef5350',
                row=1, col=1,
            )

    def _add_pattern_annotations(self, fig, df):
        """Candlestick pattern annotations — chart-এ mark করো"""
        pattern_cols = ['pattern', 'engulfing', 'star_pattern']
        colors = {
            'bullish': '#26a69a',
            'bearish': '#ef5350',
            'neutral': '#f39c12',
        }
        bullish_patterns = {
            'hammer', 'bullish_engulfing', 'morning_star', 'bullish_pin_bar'
        }
        bearish_patterns = {
            'shooting_star', 'bearish_engulfing', 'evening_star', 'bearish_pin_bar'
        }

        for col in pattern_cols:
            if col not in df.columns:
                continue
            for idx, row in df.iterrows():
                pat = row[col]
                if pat == 'none' or pd.isna(pat):
                    continue

                if pat in bullish_patterns:
                    color, ay, symbol = colors['bullish'], 40, '▲'
                elif pat in bearish_patterns:
                    color, ay, symbol = colors['bearish'], -40, '▼'
                else:
                    color, ay, symbol = colors['neutral'], -40, '◆'

                fig.add_annotation(
                    x          = idx,
                    y          = row['low'] if pat in bullish_patterns else row['high'],
                    text       = f"{symbol} {pat.replace('_', ' ').title()}",
                    showarrow  = True,
                    arrowhead  = 2,
                    arrowcolor = color,
                    font       = dict(size=9, color=color),
                    ay         = ay,
                    row=1, col=1,
                )

    def _add_rsi(self, fig, df):
        """RSI subplot"""
        if 'rsi' not in df.columns:
            return

        fig.add_trace(
            go.Scatter(
                x    = df.index,
                y    = df['rsi'],
                name = 'RSI',
                line = dict(color='#f39c12', width=1.5),
            ),
            row=2, col=1,
        )
        # Overbought / Oversold lines
        for level, color, label in [(70, 'red', 'OB'), (30, 'green', 'OS')]:
            fig.add_hline(
                y=level, line_color=color,
                line_dash='dot', line_width=1,
                annotation_text=label,
                row=2, col=1,
            )

    def _add_macd(self, fig, df):
        """MACD subplot"""
        if 'macd' not in df.columns:
            return

        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['macd'],
                name='MACD', line=dict(color='#3498db', width=1.5),
            ),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['macd_signal'],
                name='Signal', line=dict(color='#e74c3c', width=1.5),
            ),
            row=3, col=1,
        )
        # Histogram
        if 'macd_hist' in df.columns:
            colors = ['#26a69a' if v >= 0 else '#ef5350'
                      for v in df['macd_hist']]
            fig.add_trace(
                go.Bar(
                    x=df.index, y=df['macd_hist'],
                    name='Histogram', marker_color=colors, opacity=0.6,
                ),
                row=3, col=1,
            )

    def _add_volume(self, fig, df):
        """Volume subplot"""
        if 'volume' not in df.columns:
            return

        colors = [
            '#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i]
            else '#ef5350'
            for i in range(len(df))
        ]
        fig.add_trace(
            go.Bar(
                x=df.index, y=df['volume'],
                name='Volume', marker_color=colors, opacity=0.7,
            ),
            row=4, col=1,
        )

    def _apply_layout(self, fig):
        """Dark theme layout — TradingView-এর মতো"""
        fig.update_layout(
            title           = f"🤖 AI Trader — {self.symbol} {self.timeframe}",
            height          = 900,
            paper_bgcolor   = '#131722',
            plot_bgcolor    = '#131722',
            font_color      = '#d1d4dc',
            xaxis_rangeslider_visible = False,
            legend = dict(
                orientation = 'h',
                yanchor     = 'bottom',
                y           = 1.02,
                xanchor     = 'right',
                x           = 1,
            ),
            margin = dict(l=60, r=60, t=80, b=40),
        )
        # Dark grid for all axes
        for axis in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2',
                     'xaxis3', 'yaxis3', 'xaxis4', 'yaxis4']:
            fig.update_layout(**{
                axis: dict(
                    gridcolor   = '#2a2e39',
                    zerolinecolor = '#2a2e39',
                )
            })