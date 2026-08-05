"""orchestrator/trading_sessions.py
Lightweight Sessions manager used by core boot wiring.
Provides a minimal `Sessions` API (callable) delegating to utils.session.SessionAnalyzer.
This restores the legacy module expected by core/_orphan_integration.py.
"""
from datetime import datetime
from utils.session import SessionAnalyzer

class Sessions:
    """Compatibility wrapper exposing the simple session detection API.

    Usage:
        s = Sessions()
        s()                # returns current session info dict
        s(datetime_obj)    # returns session info for that datetime
        s.get_current_session(dt)
    """
    def __init__(self):
        self._analyzer = SessionAnalyzer()

    def get_current_session(self, dt: datetime = None) -> dict:
        return self._analyzer.get_current_session(dt)

    def __call__(self, arg: datetime = None) -> dict:
        return self.get_current_session(arg)
