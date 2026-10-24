"""
scoring.py – engagement level rules and class percentage.

Pure functions so they can be tested without any models.
"""
from __future__ import annotations


def student_level(
    direction: str,
    phone_detected: bool,
    hand_raised: bool,
    question_active: bool,
) -> str:
    """
    Map a student's current state to an engagement level.

    Priority (highest first)
    ───────────────────────
    1. Hand raised DURING a question window → engaged
    2. Facing away → disengaged
    3. Slightly turned → partial
    4. Facing + phone visible → partial
    5. Facing + no phone → engaged
    """
    if hand_raised and question_active:
        return "engaged"
    if direction == "away":
        return "disengaged"
    if direction == "slight":
        return "partial"
    if phone_detected:
        return "partial"
    return "engaged"


def class_percent(engaged: int, partial: int, total: int) -> float:
    """
    Weighted engagement score:
        (engaged + 0.5 × partial) / total × 100

    Returns 0.0 when there are no students.
    """
    if total == 0:
        return 0.0
    return round((engaged + 0.5 * partial) / total * 100, 1)
