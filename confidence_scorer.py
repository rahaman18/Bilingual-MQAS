def get_confidence_score(answer):
    return len(answer.split()) / 20.0  # Simple heuristic based on length