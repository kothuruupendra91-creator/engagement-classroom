"""
events.py – detects noteworthy moments during a session and returns
(event_type, detail) tuples that are written to the database.

EventDetector.update(smooth_pct, hands, phones, now, q_active)
    → list[(kind: str, detail: str)]
"""
from __future__ import annotations

import config


class EventDetector:
    """
    Stateful detector: call update() on every scoring cycle.
    Events are emitted at most once per EVENT_MIN_GAP_SEC seconds
    (except hand/phone transitions, which are edge-triggered).
    """

    _DELTA = config.EVENT_PCT_DELTA    # 15 %
    _GAP   = config.EVENT_MIN_GAP_SEC  # 5 s

    def __init__(self):
        self._last_pct:      float | None = None
        self._prev_phones:   int  = 0
        self._hand_signaled: bool = False   # True while hands are up → no repeat
        self._last_ts: dict[str, float] = {}

    # ------------------------------------------------------------------
    def update(
        self,
        smooth_pct:  float,
        hands:       int,
        phones:      int,
        now:         float,
        q_active:    bool,
    ) -> list[tuple[str, str]]:
        """
        Parameters
        ----------
        smooth_pct : rolling-average class engagement %
        hands      : number of raised hands this frame
        phones     : number of phones detected this frame
        now        : current time (time.time())
        q_active   : is the question window open?

        Returns
        -------
        list of (event_type, detail) pairs to be logged.
        """
        events: list[tuple[str, str]] = []

        # ── 1. engagement change ──────────────────────────────────────
        if self._last_pct is not None:
            delta = smooth_pct - self._last_pct

            def _gap_ok(key: str) -> bool:
                return now - self._last_ts.get(key, -(self._GAP + 1)) >= self._GAP

            if delta <= -self._DELTA and _gap_ok("engagement_drop"):
                events.append((
                    "engagement_drop",
                    f"Engagement fell from {self._last_pct:.0f}% to {smooth_pct:.0f}%",
                ))
                self._last_ts["engagement_drop"] = now

            elif delta >= self._DELTA and _gap_ok("engagement_rise"):
                events.append((
                    "engagement_rise",
                    f"Engagement rose from {self._last_pct:.0f}% to {smooth_pct:.0f}%",
                ))
                self._last_ts["engagement_rise"] = now

        self._last_pct = smooth_pct

        # ── 2. hand raise (edge: off→on while question is open) ───────
        if hands > 0 and q_active and not self._hand_signaled:
            events.append(("hand_raised", f"{hands} hand(s) raised"))
            self._hand_signaled = True
        if hands == 0:
            self._hand_signaled = False   # reset so the next Q window fires

        # ── 3. phone detection (edge: off→on and on→off) ─────────────
        if phones > 0 and self._prev_phones == 0:
            events.append(("phone_detected", "Phone visible in classroom"))
        elif phones == 0 and self._prev_phones > 0:
            events.append(("phone_cleared", "Phone no longer visible"))
        self._prev_phones = phones

        return events
