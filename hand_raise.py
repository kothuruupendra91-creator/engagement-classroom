import config

L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16


def _vis(lm):
    return 1.0 if getattr(lm, "visibility", None) is None else lm.visibility


def hand_raised(lms):
    """True if either wrist is clearly above its shoulder."""
    for sh, wr in ((L_SHOULDER, L_WRIST), (R_SHOULDER, R_WRIST)):
        s, w = lms[sh], lms[wr]
        if _vis(s) > 0.5 and _vis(w) > 0.5 and w.y < s.y - config.HAND_MARGIN:
            return True
    return False
