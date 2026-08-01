# alerts/mt5_alert_engine.py
# ============================================================
# MT5-Style Alert Condition Engine
# ============================================================
# MT5 User Guide Page 29 — Alert Editor condition types.
# Implements a simple rule engine over:
#   {Bid, Ask, Last, Volume, Time} × {>, <, =} × {threshold value}
#
# 9 condition types (per book):
#   1. BID >   — Bid price greater than value
#   2. BID <   — Bid price less than value
#   3. ASK >   — Ask price greater than value
#   4. ASK <   — Ask price less than value
#   5. LAST >  — Last traded price greater than value
#   6. LAST <  — Last traded price less than value
#   7. VOLUME > — Volume greater than value
#   8. VOLUME < — Volume less than value
#   9. TIME =   — Server time equals value
#
# Actions (per book):
#   - Sound: play alert sound
#   - File: write to file
#   - Mail: send email
#   - Notification: push notification
#
# Additional fields:
#   - Timeout: snooze interval (seconds) after first fire
#   - Maximum Retries: number of times alert repeats
#   - Expiration: optional expiry datetime
# ============================================================

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Callable, Any
from enum import Enum

log = logging.getLogger(__name__)


# ─── Enums ────────────────────────────────────────────────────

class AlertField(str, Enum):
    BID    = "BID"
    ASK    = "ASK"
    LAST   = "LAST"
    VOLUME = "VOLUME"
    TIME   = "TIME"


class AlertOperator(str, Enum):
    GREATER = ">"
    LESS    = "<"
    EQUAL   = "="


class AlertAction(str, Enum):
    SOUND         = "Sound"
    FILE          = "File"
    MAIL          = "Mail"
    NOTIFICATION  = "Notification"


# ─── Dataclass ────────────────────────────────────────────────

@dataclass
class AlertCondition:
    """Single MT5-style alert condition."""
    name: str
    symbol: str
    field: AlertField           # BID, ASK, LAST, VOLUME, TIME
    operator: AlertOperator     # >, <, =
    value: float                # threshold value
    action: AlertAction = AlertAction.NOTIFICATION
    source: str = ""            # sound file / file path / email
    timeout_sec: int = 0        # snooze interval (0 = no snooze)
    max_retries: int = 1        # number of times to repeat
    expiration: Optional[datetime] = None
    enabled: bool = True

    # Internal state
    _last_fired: Optional[datetime] = field(default=None, repr=False)
    _fire_count: int = field(default=0, repr=False)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "field": self.field.value,
            "operator": self.operator.value,
            "value": self.value,
            "action": self.action.value,
            "source": self.source,
            "timeout_sec": self.timeout_sec,
            "max_retries": self.max_retries,
            "expiration": self.expiration.isoformat() if self.expiration else None,
            "enabled": self.enabled,
            "fire_count": self._fire_count,
        }


@dataclass
class AlertResult:
    """Result of an alert check."""
    alert_name: str
    fired: bool
    reason: str
    current_value: Optional[float] = None
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "alert_name": self.alert_name,
            "fired": bool(self.fired),
            "reason": self.reason,
            "current_value": self.current_value,
            "timestamp": self.timestamp,
        }


# ─── Alert Engine ─────────────────────────────────────────────

class MT5AlertEngine:
    """
    MT5-style alert condition engine.

    Implements the 9 condition types from MT5 User Guide Page 29:
      {Bid, Ask, Last, Volume} × {>, <} + {Time} × {=}

    Usage:
        engine = MT5AlertEngine()
        engine.add_alert(AlertCondition(
            name="EURUSD_bid_above_1.09",
            symbol="EURUSD",
            field=AlertField.BID,
            operator=AlertOperator.GREATER,
            value=1.0900,
            action=AlertAction.NOTIFICATION,
        ))
        results = engine.check_all(tick_data={"EURUSD": {"bid": 1.0910, "ask": 1.0912}})
    """

    def __init__(self):
        self._alerts: List[AlertCondition] = []
        self._action_handlers: dict[AlertAction, Callable] = {
            AlertAction.SOUND: self._default_sound_handler,
            AlertAction.FILE: self._default_file_handler,
            AlertAction.MAIL: self._default_mail_handler,
            AlertAction.NOTIFICATION: self._default_notification_handler,
        }

    # ═══════════════════════════════════════════════════════════
    # ALERT MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    def add_alert(self, alert: AlertCondition) -> None:
        """Register a new alert condition."""
        self._alerts.append(alert)
        log.info(f"[AlertEngine] Added alert '{alert.name}' for {alert.symbol}: "
                 f"{alert.field.value} {alert.operator.value} {alert.value}")

    def remove_alert(self, name: str) -> bool:
        """Remove an alert by name."""
        before = len(self._alerts)
        self._alerts = [a for a in self._alerts if a.name != name]
        removed = len(self._alerts) < before
        if removed:
            log.info(f"[AlertEngine] Removed alert '{name}'")
        return removed

    def list_alerts(self) -> List[dict]:
        """List all registered alerts."""
        return [a.to_dict() for a in self._alerts]

    def set_action_handler(self, action: AlertAction, handler: Callable) -> None:
        """Override the default action handler for a given action type."""
        self._action_handlers[action] = handler

    # ═══════════════════════════════════════════════════════════
    # CONDITION CHECKING
    # ═══════════════════════════════════════════════════════════

    def check_all(self, tick_data: dict, server_time: Optional[datetime] = None) -> List[AlertResult]:
        """
        Check all alerts against current tick data.

        Args:
            tick_data: {symbol: {"bid": float, "ask": float, "last": float, "volume": float}}
            server_time: current server time (defaults to UTC now)

        Returns:
            List of AlertResult for alerts that fired.
        """
        if server_time is None:
            server_time = datetime.now(timezone.utc)

        results = []
        for alert in self._alerts:
            if not alert.enabled:
                continue

            # Check expiration
            if alert.expiration and server_time > alert.expiration:
                alert.enabled = False
                log.info(f"[AlertEngine] Alert '{alert.name}' expired")
                continue

            # Check max retries
            if alert._fire_count >= alert.max_retries:
                continue

            # Check timeout (snooze)
            if alert._last_fired and alert.timeout_sec > 0:
                elapsed = (server_time - alert._last_fired).total_seconds()
                if elapsed < alert.timeout_sec:
                    continue

            # Get current value for the alert's symbol
            symbol_data = tick_data.get(alert.symbol, {})
            result = self._check_condition(alert, symbol_data, server_time)

            if result.fired:
                alert._last_fired = server_time
                alert._fire_count += 1
                self._execute_action(alert)
                results.append(result)

        return results

    def _check_condition(
        self, alert: AlertCondition, symbol_data: dict, server_time: datetime
    ) -> AlertResult:
        """Check a single alert condition against current data."""
        field_name = alert.field.value.lower()

        # Handle TIME = specially (uses server_time, not tick data)
        if alert.field == AlertField.TIME:
            # Time equality check — compare hour:minute
            target_time = str(alert.value)
            current_time = server_time.strftime("%H:%M")
            fired = current_time == target_time
            return AlertResult(
                alert_name=alert.name,
                fired=fired,
                reason=f"TIME = {alert.value} (current: {current_time})",
                current_value=float(server_time.timestamp()),
                timestamp=server_time.isoformat(),
            )

        # For price/volume fields, get current value from tick data
        current_value = symbol_data.get(field_name)
        if current_value is None:
            return AlertResult(
                alert_name=alert.name,
                fired=False,
                reason=f"No {field_name} data for {alert.symbol}",
                timestamp=server_time.isoformat(),
            )

        # Apply operator
        if alert.operator == AlertOperator.GREATER:
            fired = current_value > alert.value
        elif alert.operator == AlertOperator.LESS:
            fired = current_value < alert.value
        elif alert.operator == AlertOperator.EQUAL:
            fired = abs(current_value - alert.value) < 1e-6
        else:
            fired = False

        reason = (
            f"{alert.field.value} {alert.operator.value} {alert.value} "
            f"(current: {current_value:.5f}) → {'FIRED' if fired else 'not fired'}"
        )

        return AlertResult(
            alert_name=alert.name,
            fired=fired,
            reason=reason,
            current_value=current_value,
            timestamp=server_time.isoformat(),
        )

    # ═══════════════════════════════════════════════════════════
    # ACTION EXECUTION
    # ═══════════════════════════════════════════════════════════

    def _execute_action(self, alert: AlertCondition) -> None:
        """Execute the alert's configured action."""
        handler = self._action_handlers.get(alert.action)
        if handler:
            try:
                handler(alert)
            except Exception as e:
                log.error(f"[AlertEngine] Action handler error for '{alert.name}': {e}")
        else:
            log.warning(f"[AlertEngine] No handler for action {alert.action}")

    def _default_sound_handler(self, alert: AlertCondition) -> None:
        log.info(f"[AlertEngine] 🔔 SOUND: {alert.name} (file: {alert.source or 'default'})")

    def _default_file_handler(self, alert: AlertCondition) -> None:
        log.info(f"[AlertEngine] 📄 FILE: {alert.name} → {alert.source}")
        if alert.source:
            try:
                with open(alert.source, "a") as f:
                    f.write(f"{datetime.now(timezone.utc).isoformat()} | {alert.name}\n")
            except Exception as e:
                log.error(f"[AlertEngine] File write failed: {e}")

    def _default_mail_handler(self, alert: AlertCondition) -> None:
        log.info(f"[AlertEngine] 📧 MAIL: {alert.name} → {alert.source}")

    def _default_notification_handler(self, alert: AlertCondition) -> None:
        log.info(f"[AlertEngine] 🔔 NOTIFICATION: {alert.name}")
        # Try to send via Telegram if available
        try:
            from alerts.telegram_bot import TelegramNotifier
            notifier = TelegramNotifier()
            import asyncio
            asyncio.run(notifier.send_message(
                f"🔔 *MT5 Alert Fired*\n"
                f"  Alert: `{alert.name}`\n"
                f"  Symbol: `{alert.symbol}`\n"
                f"  Condition: `{alert.field.value} {alert.operator.value} {alert.value}`\n"
                f"  Action: `{alert.action.value}`"
            ))
        except Exception:
            pass  # Telegram not configured — just log

    # ═══════════════════════════════════════════════════════════
    # RESET / STATUS
    # ═══════════════════════════════════════════════════════════

    def reset_all(self) -> None:
        """Reset all alert fire counts + last_fired times."""
        for alert in self._alerts:
            alert._fire_count = 0
            alert._last_fired = None
            alert.enabled = True
        log.info(f"[AlertEngine] Reset {len(self._alerts)} alert(s)")

    def get_status(self) -> dict:
        """Get engine status summary."""
        return {
            "total_alerts": len(self._alerts),
            "enabled": sum(1 for a in self._alerts if a.enabled),
            "expired": sum(1 for a in self._alerts if not a.enabled),
            "fired_total": sum(a._fire_count for a in self._alerts),
            "alerts": [a.to_dict() for a in self._alerts],
        }


# ============================================================
# Convenience: Create standard MT5-style alerts
# ============================================================

def create_price_alert(
    name: str, symbol: str, field: str, operator: str, value: float,
    action: str = "Notification", source: str = "",
) -> AlertCondition:
    """Quick helper to create a price-based alert.

    Args:
        name: alert name
        symbol: e.g., "EURUSD"
        field: "BID", "ASK", "LAST", "VOLUME", "TIME"
        operator: ">", "<", "="
        value: threshold value
        action: "Sound", "File", "Mail", "Notification"
        source: sound file / file path / email
    """
    return AlertCondition(
        name=name,
        symbol=symbol,
        field=AlertField(field.upper()),
        operator=AlertOperator(operator),
        value=value,
        action=AlertAction(action),
        source=source,
    )


# ============================================================
# CLI entry
# ============================================================

if __name__ == "__main__":
    engine = MT5AlertEngine()

    # Add alerts (MT5 User Guide Page 29 — 9 condition types)
    engine.add_alert(create_price_alert(
        "EURUSD_bid_above_1.09", "EURUSD", "BID", ">", 1.0900
    ))
    engine.add_alert(create_price_alert(
        "EURUSD_ask_below_1.08", "EURUSD", "ASK", "<", 1.0800
    ))
    engine.add_alert(create_price_alert(
        "GBPUSD_volume_high", "GBPUSD", "VOLUME", ">", 10000
    ))

    # Simulate tick data
    tick_data = {
        "EURUSD": {"bid": 1.0910, "ask": 1.0912, "last": 1.0911, "volume": 5000},
        "GBPUSD": {"bid": 1.2500, "ask": 1.2502, "last": 1.2501, "volume": 15000},
    }

    results = engine.check_all(tick_data)
    print(f"\n{'='*50}")
    print(f"  MT5 Alert Engine — {len(results)} alert(s) fired")
    print(f"{'='*50}")
    for r in results:
        print(f"\n  🔔 {r.alert_name}")
        print(f"     {r.reason}")
    print(f"\n{'='*50}")
