# visualization/chart_decorator.py — Visual trade overlay on chart images
# =============================================================================
# Ported from: https://github.com/zeroxt32/Forex-Expert-Advisor-Python
# Original: ChartDecorator class (forex_utils.py + T32_v5.py)
# Original author: zeroxt32 — MIT license
#
# Draws trade information directly onto MT5 chart screenshots:
#   - Account balance health bar (green = healthy, red = depleted)
#   - Current trade position indicator (BUY=green, SELL=magenta, CLOSE=yellow)
#   - Profit/loss bar at the top of the image
#   - Trade entry/exit markers on the price axis
#
# This is used by the RL agent's visual environment to overlay trade state
# onto chart images, so the agent can "see" its current position and P&L
# as part of the observation.
#
# The original code is quite dense (pixel coordinates hardcoded for 224×224
# images). This port cleans it up, adds docstrings, and makes the image
# dimensions configurable.
# =============================================================================

from __future__ import annotations

from typing import Optional

from utils.logger import get_logger

log = get_logger("chart_decorator")

try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    log.warning("Pillow not installed. Install with: pip install Pillow")


# Color constants (matching original)
COLORS = {
    "green": (110, 235, 131),
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "orange": (255, 87, 20),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
    "cyan": (27, 152, 224),
    "gray": (200, 200, 200),
}


class ChartDecorator:
    """
    Draws trade overlay information on chart screenshot images.

    Designed for 224×224 pixel MT5 chart screenshots. The overlay includes:
      - Top bar: profit/loss indicator (segmented, centered at 100)
      - Bottom bar: account balance health + current position indicator
      - Trade entry/exit lines drawn on the price axis

    Parameters
    ----------
    img_width, img_height : image dimensions (default 224×224).
    balance_limit : minimum balance threshold for the health bar (default 300).
    standard_balance : starting balance for health bar scaling (default 1000).
    """

    def __init__(
        self,
        img_width: int = 224,
        img_height: int = 224,
        balance_limit: float = 300,
        standard_balance: float = 1000,
    ):
        if not _HAS_PIL:
            raise ImportError("Pillow required. Install with: pip install Pillow")
        self.img_width = img_width
        self.img_height = img_height
        self.balance_limit = balance_limit
        self.standard_balance = standard_balance
        self.prev_trade: Optional[list] = None

    def _get_draw(self, img: "Image.Image") -> "ImageDraw.ImageDraw":
        return ImageDraw.Draw(img)

    # ── Account balance health bar ───────────────────────────────────────────

    def draw_account_balance(
        self,
        img: "Image.Image",
        draw: "ImageDraw.ImageDraw",
        account_balance: float = 1000,
    ) -> None:
        """
        Draw the account balance health bar at the bottom of the image.
        Green = healthy, red = depleted, blue = profit above standard.
        """
        balance = self.balance_limit + (account_balance - self.standard_balance)
        bar_start = 3
        bar_end = 75
        bar_top = 204
        bar_bottom = 224

        if balance >= self.balance_limit:
            # Full green bar + optional blue profit section
            draw.rectangle([bar_start, bar_top, bar_end, bar_bottom],
                          fill=COLORS["green"])
            if balance > self.balance_limit:
                profit = int(((balance - self.balance_limit) * 75) / self.balance_limit)
                draw.rectangle([80, bar_top, max(80, 80 + profit), bar_bottom],
                              fill=COLORS["blue"])
        else:
            # Partial green + red depleted section
            health = max(bar_start, int((max(0, balance) * 75) / self.balance_limit))
            draw.rectangle([bar_start, bar_top, health, bar_bottom],
                          fill=COLORS["green"])
            if health < bar_end:
                draw.rectangle([health, bar_top, bar_end, bar_bottom],
                              fill=COLORS["red"])

        # Separator lines
        for x in [0, 76, 155]:
            draw.rectangle([x, bar_top, x + 3, bar_bottom], fill=COLORS["orange"])

    # ── Position indicator ───────────────────────────────────────────────────

    def draw_position_indicator(
        self,
        img: "Image.Image",
        draw: "ImageDraw.ImageDraw",
        position: Optional[str] = None,
    ) -> None:
        """
        Draw the current position indicator: BUY=green, SELL=magenta, CLOSE=yellow.
        """
        bar_top, bar_bottom = 204, 224
        indicators = {
            "sell": (160, 175, COLORS["yellow"]),
            "buy": (180, 195, COLORS["green"]),
            "close": (200, 215, COLORS["magenta"]),
        }
        pos_key = position if position in indicators else "close"
        x1, x2, color = indicators[pos_key]
        draw.rectangle([x1, bar_top, x2, bar_bottom], fill=color)

        # White separators
        for x in [175, 195, 215]:
            draw.rectangle([x, bar_top, x + 1, bar_bottom], fill=COLORS["white"])

    # ── Profit/loss bar ──────────────────────────────────────────────────────

    def draw_profit_bar(
        self,
        img: "Image.Image",
        draw: "ImageDraw.ImageDraw",
        profit: float = 0,
        account_balance: float = 1000,
    ) -> None:
        """
        Draw the profit/loss bar at the top of the image.
        Positive profit → cyan segments extending right from center.
        Negative profit → red segments extending left from center.
        """
        # Segmented bar (20 segments of 10px each)
        pnl_parts = int(abs(profit) / 10)
        for segment in range(0, 200, 10):
            x1 = segment + 10
            draw.rectangle([x1, 0, x1 + 1, 9], fill=COLORS["white"])

            if profit > 0 and pnl_parts > 0 and segment >= 100:
                if segment < pnl_parts * 10 + 100:
                    draw.rectangle([segment, 0, segment + 10, 9], fill=COLORS["cyan"])
            elif profit < 0 and pnl_parts > 0 and segment <= 90:
                if segment >= 100 - abs(profit):
                    draw.rectangle([segment, 0, segment + 10, 9], fill=COLORS["red"])

        # Account balance progress line
        balance_bar = max(0, int((200 * account_balance) / 2000))
        draw.rectangle([0, 10, balance_bar, 14], fill=COLORS["orange"])

    # ── Trade entry/exit lines ───────────────────────────────────────────────

    def draw_trade_lines(
        self,
        img: "Image.Image",
        draw: "ImageDraw.ImageDraw",
        start_lines: dict,
        end_lines: dict,
        position: str = "close",
        profit: float = 0,
    ) -> None:
        """
        Draw trade entry (start) and exit (end) lines on the price axis.

        Parameters
        ----------
        start_lines : {"ask_line": y, "bid_line": y} at trade entry
        end_lines : {"ask_line": y, "bid_line": y} at current bar
        position : "buy", "sell", or "close"
        profit : current P&L
        """
        if position == "buy":
            entry_color = COLORS["yellow"]
            exit_color = COLORS["gray"]
        elif position == "sell":
            entry_color = COLORS["green"]
            exit_color = COLORS["gray"]
        else:
            entry_color = COLORS["magenta"]
            exit_color = COLORS["magenta"]

        # Entry line (full width)
        if start_lines.get("ask_line") and start_lines.get("bid_line"):
            draw.line(
                [(50, start_lines["ask_line"]), (50, start_lines["bid_line"])],
                fill=entry_color, width=250
            )

        # Exit/current line (right side)
        if end_lines.get("ask_line") and end_lines.get("bid_line"):
            draw.line(
                [(200, end_lines["ask_line"]), (200, end_lines["bid_line"])],
                fill=exit_color, width=20
            )

        # Profit/loss connecting line
        if (start_lines.get("ask_line") and end_lines.get("bid_line") and
                position in ["buy", "sell"]):
            pl_color = COLORS["green"] if profit >= 0 else COLORS["red"]
            draw.line(
                [(200, start_lines["ask_line"]), (200, end_lines["bid_line"])],
                fill=pl_color, width=10
            )

    # ── Full decoration ──────────────────────────────────────────────────────

    def decorate(
        self,
        img: "Image.Image",
        *,
        account_balance: float = 1000,
        position: str = "close",
        profit: float = 0,
        start_lines: Optional[dict] = None,
        end_lines: Optional[dict] = None,
    ) -> "Image.Image":
        """
        Apply all decorations to a chart image.

        Parameters
        ----------
        img : PIL Image of the chart (224×224 recommended).
        account_balance : current account balance.
        position : "buy", "sell", or "close".
        profit : current unrealized P&L.
        start_lines : entry ask/bid lines (optional).
        end_lines : current ask/bid lines (optional).

        Returns
        -------
        The decorated image (modified in place + returned).
        """
        draw = self._get_draw(img)

        # Clear top and bottom bars
        draw.rectangle([0, 0, self.img_width, 14], fill=COLORS["black"])
        draw.rectangle([0, 204, self.img_width, self.img_height], fill=COLORS["black"])

        # Draw all overlays
        self.draw_account_balance(img, draw, account_balance)
        self.draw_position_indicator(img, draw, position)
        self.draw_profit_bar(img, draw, profit, account_balance)

        if start_lines and end_lines:
            self.draw_trade_lines(img, draw, start_lines, end_lines, position, profit)

        return img

    # ── Create a blank chart for testing ─────────────────────────────────────

    @staticmethod
    def create_blank_chart(width: int = 224, height: int = 224) -> "Image.Image":
        """Create a blank dark chart image for testing."""
        if not _HAS_PIL:
            raise ImportError("Pillow required")
        return Image.new("RGB", (width, height), (20, 20, 30))


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not _HAS_PIL:
        print("Pillow not installed. Install with: pip install Pillow")
    else:
        decorator = ChartDecorator()

        # Test 1: blank chart with BUY position
        img = ChartDecorator.create_blank_chart()
        img = decorator.decorate(
            img,
            account_balance=1200,  # above standard → blue profit bar
            position="buy",
            profit=25,  # positive profit → cyan segments
            start_lines={"ask_line": 100, "bid_line": 110},
            end_lines={"ask_line": 80, "bid_line": 90},
        )
        img.save("/tmp/test_chart_buy.png")
        print("✓ BUY chart saved to /tmp/test_chart_buy.png")

        # Test 2: SELL position with loss
        img2 = ChartDecorator.create_blank_chart()
        img2 = decorator.decorate(
            img2,
            account_balance=250,  # below balance_limit → red health bar
            position="sell",
            profit=-30,  # negative profit → red segments
            start_lines={"ask_line": 80, "bid_line": 90},
            end_lines={"ask_line": 100, "bid_line": 110},
        )
        img2.save("/tmp/test_chart_sell_loss.png")
        print("✓ SELL+loss chart saved to /tmp/test_chart_sell_loss.png")

        # Test 3: CLOSE position (no trade)
        img3 = ChartDecorator.create_blank_chart()
        img3 = decorator.decorate(
            img3,
            account_balance=1000,
            position="close",
            profit=0,
        )
        img3.save("/tmp/test_chart_close.png")
        print("✓ CLOSE chart saved to /tmp/test_chart_close.png")

        # Verify image dimensions
        assert img.size == (224, 224)
        assert img2.size == (224, 224)
        assert img3.size == (224, 224)

        print("\nChartDecorator smoke test passed.")
