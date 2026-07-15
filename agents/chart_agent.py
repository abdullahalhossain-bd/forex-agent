# agents/chart_agent.py
from playwright.sync_api import sync_playwright
import yfinance as yf
import numpy as np
import time

class ChartAgent:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        self.current_price = None
        self.support_levels = []
        self.resistance_levels = []

    def start(self, headless=False):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=headless,
            args=["--start-maximized"]
        )
        self.page = self.browser.new_page(
            viewport={"width": 1600, "height": 900}
        )
        print("✅ Browser started")

    # ─────────────────────────────────────────
    # STEP 1: S/R CALCULATION (data থেকে)
    # ─────────────────────────────────────────
    def calculate_sr_levels(self, symbol="EURUSD=X", period="5d", interval="15m"):
        print(f"\n📊 Calculating S/R for {symbol}...")
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            
            highs  = df['High'].values.flatten()
            lows   = df['Low'].values.flatten()
            closes = df['Close'].values.flatten()
            
            self.current_price = float(closes[-1])

            # Pivot point দিয়ে S/R বের করা
            support    = []
            resistance = []

            for i in range(3, len(closes) - 3):
                # Pivot Low = Support
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                        lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    support.append(round(float(lows[i]), 5))

                # Pivot High = Resistance
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                        highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    resistance.append(round(float(highs[i]), 5))

            def cluster(levels, tol=0.0004):
                if not levels:
                    return []
                levels = sorted(set(levels))
                out = [levels[0]]
                for l in levels[1:]:
                    if abs(l - out[-1]) > tol:
                        out.append(l)
                return out

            all_support    = cluster(support)
            all_resistance = cluster(resistance)

            # Current price এর নিচে = support, উপরে = resistance
            self.support_levels    = [x for x in all_support    if x < self.current_price][-3:]
            self.resistance_levels = [x for x in all_resistance if x > self.current_price][:3]

            print(f"   Current Price : {self.current_price:.5f}")
            print(f"   Support       : {self.support_levels}")
            print(f"   Resistance    : {self.resistance_levels}")

        except Exception as e:
            print(f"❌ S/R Error: {e}")

    # ─────────────────────────────────────────
    # STEP 2: TRADINGVIEW OPEN
    # ─────────────────────────────────────────
    def open_tradingview(self, symbol="EURUSD"):
        url = f"https://www.tradingview.com/chart/?symbol=FX:{symbol}"
        print(f"\n📡 Opening TradingView: {symbol}")
        self.page.goto(url, wait_until="domcontentloaded")
        time.sleep(7)
        # Dialog/popup বন্ধ করো
        self.page.keyboard.press("Escape")
        time.sleep(1)
        print("✅ TradingView loaded")

    # ─────────────────────────────────────────
    # STEP 3: TIMEFRAME
    # ─────────────────────────────────────────
    def change_timeframe(self, timeframe="15"):
        try:
            btn = self.page.locator(f'button[data-value="{timeframe}"]').first
            btn.wait_for(timeout=5000)
            btn.click()
            time.sleep(2)
            print(f"✅ Timeframe: {timeframe}m")
        except Exception as e:
            print(f"⚠️ Timeframe error: {e}")

    # ─────────────────────────────────────────
    # STEP 4: INDICATOR ADD
    # ─────────────────────────────────────────
    def add_indicator(self, name):
        try:
            print(f"🔧 Adding: {name}")
            # Indicators button
            self.page.locator('button[data-name="indicators"]').first.click()
            time.sleep(2.5)

            # Search box
            search = self.page.locator('input[data-role="search"]').first
            search.fill(name)
            time.sleep(2)

            # প্রথম result click
            result = self.page.locator('[class*="itemRow"]').first
            result.click()
            time.sleep(1.5)

            # Close dialog
            self.page.keyboard.press("Escape")
            time.sleep(1)
            print(f"   ✅ {name} added")

        except Exception as e:
            print(f"   ⚠️ {name} failed: {e}")

    # ─────────────────────────────────────────
    # STEP 5: S/R DRAW (Price Scale থেকে)
    # ─────────────────────────────────────────
    def _get_chart_price_range(self):
        """
        Chart এর visible price range বের করো
        Price scale এর top ও bottom text read করে
        """
        try:
            # Price scale labels
            labels = self.page.locator(
                '[class*="priceScale"] [class*="labelRow"]'
            ).all_text_contents()
            
            prices = []
            for label in labels:
                try:
                    prices.append(float(label.replace(',', '')))
                except Exception:
                    pass
            
            if len(prices) >= 2:
                return min(prices), max(prices)
        except Exception:
            pass
        
        # Fallback: current price ± 0.5%
        p = self.current_price
        return p * 0.995, p * 1.005

    def _price_to_y(self, price, box, price_min, price_max):
        """Price কে chart এর Y pixel position এ convert করো"""
        ratio = (price_max - price) / (price_max - price_min)
        ratio = max(0.05, min(0.95, ratio))
        return box['y'] + ratio * box['height']

    def draw_sr_levels(self):
        """
        TradingView এ সরাসরি S/R lines draw করো
        Horizontal Line tool ব্যবহার করবে
        """
        try:
            # Chart bounding box
            chart = self.page.locator('.chart-container').first
            box   = chart.bounding_box()
            if not box:
                print("❌ Chart container not found")
                return

            price_min, price_max = self._get_chart_price_range()
            chart_cx = box['x'] + box['width'] * 0.5

            def draw_line(price, label):
                try:
                    # 'H' shortcut = Horizontal Line tool TradingView এ
                    self.page.keyboard.press("h")
                    time.sleep(0.8)

                    y = self._price_to_y(price, box, price_min, price_max)
                    self.page.mouse.click(chart_cx, y)
                    time.sleep(0.8)
                    self.page.keyboard.press("Escape")
                    time.sleep(0.5)
                    print(f"   ✅ {label}: {price:.5f}")

                except Exception as e:
                    print(f"   ⚠️ {label} draw error: {e}")

            print("\n📐 Drawing Support levels (green)...")
            for lvl in self.support_levels:
                draw_line(lvl, "Support")
                time.sleep(0.5)

            print("📐 Drawing Resistance levels (red)...")
            for lvl in self.resistance_levels:
                draw_line(lvl, "Resistance")
                time.sleep(0.5)

            print("✅ All S/R levels drawn!")

        except Exception as e:
            print(f"❌ Draw S/R error: {e}")

    def close(self):
        input("\n⏸️  Enter চাপো browser বন্ধ করতে...")
        self.browser.close()
        self.playwright.stop()
        print("🔴 Browser closed")