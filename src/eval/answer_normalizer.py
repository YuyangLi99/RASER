"""Answer post-processing: type classification + canonicalization."""

import re
from typing import Optional


# Question type patterns
YES_NO_PATTERNS = [
    re.compile(r'^(is|are|was|were|did|do|does|has|have|had|can|could|will|would|should)\b', re.IGNORECASE),
]

DATE_YEAR_PATTERNS = [
    re.compile(r'\b(what year|which year|when was|when did|when were|what date)\b', re.IGNORECASE),
]

COUNT_PATTERNS = [
    re.compile(r'\b(how many|how much|what number|what is the number)\b', re.IGNORECASE),
]


def classify_question_type(question: str) -> str:
    """Classify question into: yes_no, date, count, entity."""
    q = question.strip()
    for pat in YES_NO_PATTERNS:
        if pat.match(q):
            return "yes_no"
    for pat in DATE_YEAR_PATTERNS:
        if pat.search(q):
            return "date"
    for pat in COUNT_PATTERNS:
        if pat.search(q):
            return "count"
    return "entity"


def canonicalize_answer(raw_answer: str, question_type: str) -> str:
    """Canonicalize answer based on question type."""
    if not raw_answer:
        return ""

    answer = raw_answer.strip()

    if question_type == "yes_no":
        return _canonicalize_yes_no(answer)
    elif question_type == "date":
        return _canonicalize_date(answer)
    elif question_type == "count":
        return _canonicalize_count(answer)
    else:
        return _canonicalize_entity(answer)


def _canonicalize_yes_no(answer: str) -> str:
    """Extract yes/no from potentially verbose answer."""
    lower = answer.lower().strip()

    # Direct match
    if lower in ("yes", "no"):
        return lower

    # Check if answer starts with yes/no
    if lower.startswith("yes"):
        return "yes"
    if lower.startswith("no"):
        return "no"

    # Check for affirmative/negative phrases
    affirmative = ["both", "same", "correct", "true", "they are", "it is", "he is", "she is"]
    negative = ["different", "not the same", "false", "incorrect", "neither"]

    for phrase in affirmative:
        if phrase in lower:
            return "yes"
    for phrase in negative:
        if phrase in lower:
            return "no"

    return answer


def _canonicalize_date(answer: str) -> str:
    """Extract year/date from answer."""
    # Try to find a 4-digit year
    year_match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', answer)
    if year_match:
        return year_match.group(1)

    # Try full date patterns
    date_match = re.search(r'\b(\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})\b', answer)
    if date_match:
        return date_match.group(1)

    return answer.strip()


def _canonicalize_count(answer: str) -> str:
    """Extract number from answer."""
    num_match = re.search(r'\b(\d+(?:\.\d+)?)\b', answer)
    if num_match:
        return num_match.group(1)
    return answer.strip()


def _canonicalize_entity(answer: str) -> str:
    """Clean entity answer: remove full sentences, keep shortest meaningful span."""
    answer = answer.strip()

    # If answer is a full sentence, try to extract the key entity
    # Remove common prefixes like "The answer is", "It is", etc.
    prefixes = [
        r'^the answer is\s+',
        r'^it is\s+',
        r'^they are\s+',
        r'^he is\s+',
        r'^she is\s+',
        r'^the\s+.*?\s+is\s+',
    ]
    for prefix in prefixes:
        cleaned = re.sub(prefix, '', answer, flags=re.IGNORECASE).strip()
        if cleaned and len(cleaned) < len(answer):
            answer = cleaned

    # Remove trailing periods
    answer = answer.rstrip('.')

    return answer


def normalize_prediction(raw_answer: str, question: str) -> str:
    """Full pipeline: classify question type, then canonicalize answer."""
    if not raw_answer:
        return ""
    qtype = classify_question_type(question)
    return canonicalize_answer(raw_answer, qtype)
