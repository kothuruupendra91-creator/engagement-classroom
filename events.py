import config


class EventDetector:
    """Turns the live numbers into time-stamped events for the session history."""

    def __init__(self):
        self.ref_pct = None
        self.last_change_t = 0.0
        self.prev_hands = 0
        self.prev_phones = 0

    def update(self, pct, hands, phones, now, question_active):
        """Return a list of (event_type, detail) for this moment."""
        events = []
        if self.ref_pct is None:
            self.ref_pct = pct
        elif abs(pct - self.ref_pct) >= config.EVENT_PCT_DELTA and \
                now - self.last_change_t >= config.EVENT_MIN_GAP_SEC:
            kind = "engagement_rise" if pct > self.ref_pct else "engagement_drop"
            events.append((kind, f"Class engagement went from {self.ref_pct:.0f}% to {pct:.0f}%"))
            self.ref_pct, self.last_change_t = pct, now

        if hands > self.prev_hands:
            when = "during a question" if question_active else "with no question active"
            events.append(("hand_raised", f"{hands} hand(s) raised, {when}"))
            self.prev_hands = hands

        if phones > self.prev_phones:
            events.append(("phone_detected", f"{phones} phone(s) visible in the classroom"))
        elif phones == 0 and self.prev_phones > 0:
            events.append(("phone_cleared", "No phone visible any more"))
        self.prev_phones = phones
        return events
