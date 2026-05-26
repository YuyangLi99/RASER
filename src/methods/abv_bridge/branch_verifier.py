"""Branch Verifier: scores and prunes branches based on evidence quality.

Scoring dimensions:
  1. Novelty: how much new evidence does this branch bring vs one-shot?
  2. Support: does the new evidence mention the bridge entity?
  3. Consistency: does the branch answer align with one-shot answer?
  4. Information gain: Jaccard distance from one-shot evidence.

Actions per branch:
  - KEEP: branch has high novelty + support
  - DROP: branch adds nothing new or contradicts
  - REPAIR: bridge might be wrong, allow one entity replacement
"""

from typing import Dict, List, Any, Tuple


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _entity_mentioned(chunks: List[Dict], entity: str) -> float:
    """Fraction of chunks that mention the bridge entity."""
    if not entity or not chunks:
        return 0.0
    entity_lower = entity.lower()
    count = sum(1 for c in chunks if entity_lower in c.get("text", "").lower())
    return count / len(chunks)


class BranchVerifier:
    """Scores branches and decides keep/drop/repair."""

    def __init__(self, novelty_threshold: float = 0.05,
                 support_threshold: float = 0.05):
        self.novelty_threshold = novelty_threshold
        self.support_threshold = support_threshold

    def verify_branches(self, branches: List[Dict],
                        one_shot_chunk_ids: set) -> List[Dict[str, Any]]:
        """Score each branch and assign action.

        Args:
            branches: from BranchRetriever.retrieve_branches()
            one_shot_chunk_ids: set of chunk_ids from one-shot retrieval

        Returns: list of scored branches with added fields:
            "novelty": float (0-1, fraction of new chunks)
            "support": float (0-1, fraction of chunks mentioning bridge entity)
            "info_gain": float (1 - jaccard overlap with one-shot)
            "action": "keep" | "drop" | "repair"
            "score": float (composite score for ranking)
        """
        scored = []
        for branch in branches:
            chunk_ids = branch["chunk_ids"]
            entity = branch["proposal"].get("bridge_entity", "")
            confidence = branch["proposal"].get("confidence", 0.5)

            # Novelty: fraction of chunks NOT in one-shot
            if chunk_ids:
                new_chunks = chunk_ids - one_shot_chunk_ids
                novelty = len(new_chunks) / len(chunk_ids)
            else:
                novelty = 0.0

            # Support: does new evidence mention the bridge entity?
            support = _entity_mentioned(branch["chunks"], entity)

            # Information gain: 1 - jaccard overlap
            info_gain = 1.0 - _jaccard(chunk_ids, one_shot_chunk_ids)

            # Composite score
            score = (0.4 * novelty + 0.3 * support + 0.2 * info_gain
                     + 0.1 * confidence)

            # Decision: keep if bridge has ANY signal of usefulness
            if support >= self.support_threshold or novelty >= self.novelty_threshold:
                action = "keep"
            elif confidence >= 0.5 and info_gain >= 0.05:
                action = "keep"
            elif confidence < 0.3 and support < self.support_threshold:
                action = "repair"
            else:
                action = "drop"

            scored.append({
                **branch,
                "novelty": novelty,
                "support": support,
                "info_gain": info_gain,
                "score": score,
                "action": action,
            })

        # Sort by score descending
        scored.sort(key=lambda x: -x["score"])
        return scored
