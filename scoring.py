def student_level(head, phone, hand, question_active):
    """head is 'facing', 'slight' or 'away'."""
    if question_active and hand:
        return "engaged"
    if head == "away":
        return "disengaged"
    if phone:
        return "partial"
    if head == "facing":
        return "engaged"
    return "partial"


def class_percent(engaged, partial, total):
    return 0.0 if total == 0 else (engaged + 0.5 * partial) / total * 100
